"""Strategy tagging — controlled vocabulary for what each node has tried.

Every successful node carries a ``set[str]`` of strategy tags drawn from
``STRATEGY_VOCAB``. Tags are populated from two sources:

    1. **LLM declaration**: the system prompt requires the model to emit a
       ``STRATEGIES: tag1, tag2, ...`` line after its code block.
    2. **Code inference**: regex patterns scan the code for known constructs
       (``CatBoostClassifier``, ``optuna.create_study``, ``HorizontalFlip``...).

Both sources are unioned and intersected with ``STRATEGY_VOCAB`` so the agent
never has to defend against unknown tags.

The downstream consumers are:

    - **Improve prompts**: receive a per-branch history table of nodes,
      val scores, and *new* strategies added at each step, plus a list of
      *untried* strategies, and a *required* strategy the LLM must apply this
      iteration.
    - **Selector**: counts distinct strategies per branch to compute novelty
      headroom and rotate fairly.
    - **Journal persistence**: tags are serialised as a sorted list.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:  # pragma: no cover
    from .nodes import Journal, SearchNode


# ── Controlled vocabulary ──────────────────────────────────────────────────
# A flat dict so the LLM can be shown the full menu in one block. Format:
#     "category:specific" or "category" (for top-level moves like pseudo_labeling)

STRATEGY_VOCAB: dict[str, str] = {
    # ── Model families ────────────────────────────────────────────────
    "model:catboost":         "CatBoost gradient boosting (handles strings via cat_features)",
    "model:lightgbm":         "LightGBM gradient boosting (requires int-encoded categoricals)",
    "model:xgboost":          "XGBoost gradient boosting (requires one-hot or freq encoding)",
    "model:sklearn_gbm":      "sklearn GradientBoostingClassifier/Regressor",
    "model:sklearn_rf":       "sklearn RandomForest",
    "model:sklearn_extratrees": "sklearn ExtraTreesClassifier/Regressor",
    "model:sklearn_logreg":   "sklearn LogisticRegression",
    "model:sklearn_linear":   "sklearn Ridge/Lasso/Linear/ElasticNet",
    "model:sklearn_svm":      "sklearn SVC/SVR",
    "model:sklearn_knn":      "sklearn KNeighbors",
    "model:nn_mlp":           "Tabular MLP (PyTorch)",
    "model:nn_tabnet":        "TabNet",
    "model:nn_cnn":           "CNN backbone (timm or torchvision)",
    "model:nn_transformer":   "Transformer encoder (HuggingFace)",
    "model:tabpfn":           "TabPFN (very small datasets only)",

    # ── Cross-validation ──────────────────────────────────────────────
    "cv:stratified_5fold":    "StratifiedKFold(n_splits=5)",
    "cv:stratified_10fold":   "StratifiedKFold(n_splits=10)",
    "cv:kfold_5":             "KFold(n_splits=5)",
    "cv:groupkfold":          "GroupKFold",
    "cv:stratified_groupkfold": "StratifiedGroupKFold",
    "cv:timeseries_split":    "TimeSeriesSplit (no shuffle)",
    "cv:single_split":        "Single train_test_split (no CV)",
    "cv:repeated_kfold":      "RepeatedKFold (multi-seed CV)",

    # ── Feature engineering (tabular) ─────────────────────────────────
    "fe:datetime_expansion":  "Datetime → year/month/day/dayofweek/hour",
    "fe:target_encoding_oof": "Out-of-fold target encoding for high-cardinality cats",
    "fe:frequency_encoding":  "Frequency / count encoding",
    "fe:one_hot":             "One-hot encoding (low-cardinality)",
    "fe:label_encoding":      "Label / ordinal encoding via pd.factorize",
    "fe:delimited_split":     "Split multi-part string columns on delimiter (e.g. '/' or '_')",
    "fe:interaction_features": "Pairwise interactions of top features",
    "fe:numeric_binning":     "Numeric → quantile bins",
    "fe:aggregation_groupby": "GroupBy mean/sum/std features",
    "fe:polynomial":          "PolynomialFeatures",
    "fe:row_stats":           "Per-row statistics (sum, mean, nunique, etc.)",
    "fe:lag_features":        "Lag / rolling features (time series)",

    # ── Hyperparameter search ─────────────────────────────────────────
    "hp:fixed_defaults":      "Fixed default hyperparameters from working template",
    "hp:manual_tuning":       "Manually-chosen hyperparameters",
    "hp:optuna":              "Optuna hyperparameter search (≥20 trials)",
    "hp:gridsearch":          "GridSearchCV",
    "hp:randomsearch":        "RandomizedSearchCV",
    "hp:bayesian":            "BayesianOptimization",

    # ── Ensembling / stability ────────────────────────────────────────
    "ensemble:none":          "Single model, single seed",
    "ensemble:seed_averaging": "Average predictions across multiple seeds of the same model",
    "ensemble:cv_fold_averaging": "Average per-fold test predictions (no full-train refit)",
    "ensemble:in_script":     "Multiple model families inside one script",
    "ensemble:stacking":      "Stacking with OOF predictions + meta-model",
    "ensemble:rank_average":  "Rank-average across model probability columns",

    # ── Data tricks ───────────────────────────────────────────────────
    "pseudo_labeling":        "Pseudo-labeling on confident test predictions",
    "data_augmentation":      "Train-time augmentation (vision/audio)",
    "tta:hflip":              "Test-time horizontal flip averaging",
    "tta:multi_view":         "Multi-view / multi-crop TTA",
    "balancing:smote":        "SMOTE oversampling for class imbalance",
    "balancing:class_weights": "class_weight='balanced' or sample_weight",

    # ── Calibration / post-processing ─────────────────────────────────
    "calibration:isotonic":   "Isotonic calibration of probabilities",
    "calibration:platt":      "Platt (sigmoid) calibration",
    "post:threshold_tuning":  "Decision threshold tuning on validation",
    "post:rank_clip":         "Quantile / rank clipping of predictions",

    # ── Optimizers (vision/NLP) ───────────────────────────────────────
    "optimizer:adamw":        "AdamW optimizer",
    "optimizer:sgd_nesterov": "SGD with Nesterov momentum",
    "optimizer:adam":         "Adam optimizer",
    "optimizer:rmsprop":      "RMSprop optimizer",
    "optimizer:lion":         "Lion optimizer",

    # ── Schedulers (vision/NLP) ───────────────────────────────────────
    "scheduler:cosine":       "CosineAnnealingLR",
    "scheduler:onecycle":     "OneCycleLR",
    "scheduler:linear_warmup": "Linear warmup + linear decay",
    "scheduler:reduce_on_plateau": "ReduceLROnPlateau",
}


# ── Code inference patterns ───────────────────────────────────────────────
# (regex, tag). The regexes are deliberately permissive — false positives are
# fine (the LLM declaration is the authoritative source) but false negatives
# would let the LLM "secretly" use a strategy without recording it.

_INFER_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # Models
    (re.compile(r"\bCatBoost(Classifier|Regressor)|catboost\.train\b"), "model:catboost"),
    (re.compile(r"\bimport lightgbm\b|\blgb\.(train|Dataset)\b|\bLGBM(Classifier|Regressor)\b"), "model:lightgbm"),
    (re.compile(r"\bimport xgboost\b|\bxgb\.(train|DMatrix)\b|\bXGB(Classifier|Regressor)\b"), "model:xgboost"),
    (re.compile(r"\bGradientBoosting(Classifier|Regressor)\b"), "model:sklearn_gbm"),
    (re.compile(r"\bRandomForest(Classifier|Regressor)\b"), "model:sklearn_rf"),
    (re.compile(r"\bExtraTrees(Classifier|Regressor)\b"), "model:sklearn_extratrees"),
    (re.compile(r"\bLogisticRegression\b"), "model:sklearn_logreg"),
    (re.compile(r"\b(Ridge|Lasso|LinearRegression|ElasticNet)\("), "model:sklearn_linear"),
    (re.compile(r"\b(SVC|SVR|LinearSVC|LinearSVR)\("), "model:sklearn_svm"),
    (re.compile(r"\bKNeighbors(Classifier|Regressor)\b"), "model:sklearn_knn"),
    (re.compile(r"\btimm\.create_model|torchvision\.models\."), "model:nn_cnn"),
    (re.compile(r"\bAutoModel(ForSequenceClassification)?|transformers\.|HfApi"), "model:nn_transformer"),
    (re.compile(r"\bTabNet"), "model:nn_tabnet"),
    (re.compile(r"\bTabPFN"), "model:tabpfn"),

    # CV
    (re.compile(r"\bStratifiedKFold\("), "cv:stratified_5fold"),
    (re.compile(r"\bStratifiedGroupKFold\("), "cv:stratified_groupkfold"),
    (re.compile(r"\bGroupKFold\("), "cv:groupkfold"),
    (re.compile(r"\bTimeSeriesSplit\("), "cv:timeseries_split"),
    (re.compile(r"\bRepeatedKFold\("), "cv:repeated_kfold"),
    (re.compile(r"\bKFold\("), "cv:kfold_5"),
    (re.compile(r"\btrain_test_split\("), "cv:single_split"),

    # Hyperparameter search
    (re.compile(r"\boptuna\.(create_study|trial)|study\.optimize\("), "hp:optuna"),
    (re.compile(r"\bGridSearchCV\("), "hp:gridsearch"),
    (re.compile(r"\bRandomizedSearchCV\("), "hp:randomsearch"),
    (re.compile(r"\bBayesianOptimization\("), "hp:bayesian"),

    # Feature engineering
    (re.compile(r"target_encod|TargetEncoder|MeanEncoder"), "fe:target_encoding_oof"),
    (re.compile(r"\.value_counts\(\).*\.to_dict\(\)|freq_encod|frequency_encod"), "fe:frequency_encoding"),
    (re.compile(r"\bOneHotEncoder\b|pd\.get_dummies\("), "fe:one_hot"),
    (re.compile(r"\bOrdinalEncoder\b|pd\.factorize\("), "fe:label_encoding"),
    (re.compile(r"str\.split\([^)]*expand\s*=\s*True"), "fe:delimited_split"),
    (re.compile(r"\bPolynomialFeatures\b"), "fe:polynomial"),
    (re.compile(r"\.dt\.(year|month|day|dayofweek|hour|minute|quarter|dayofyear)\b"), "fe:datetime_expansion"),
    (re.compile(r"\.groupby\([^)]+\)\.(agg|mean|sum|std|max|min|count)\("), "fe:aggregation_groupby"),
    (re.compile(r"\bKBinsDiscretizer|qcut\(|pd\.cut\("), "fe:numeric_binning"),
    (re.compile(r"\.shift\(\d+\)|rolling\("), "fe:lag_features"),

    # Ensembling / stability
    (re.compile(r"for.*seed.*in.*\[.*\d+.*,.*\d+"), "ensemble:seed_averaging"),
    (re.compile(r"StackingClassifier|StackingRegressor|oof.*predict|out.of.fold"), "ensemble:stacking"),
    (re.compile(r"\.rank\(.*pct=True"), "ensemble:rank_average"),

    # Data tricks
    (re.compile(r"pseudo[_-]?label"), "pseudo_labeling"),
    (re.compile(r"\bAlbumentations|HorizontalFlip|VerticalFlip|RandomResizedCrop|ShiftScaleRotate"), "data_augmentation"),
    (re.compile(r"torch\.flip\([^)]*dims\s*=\s*\[?-1\]?"), "tta:hflip"),
    (re.compile(r"\bSMOTE\b"), "balancing:smote"),
    (re.compile(r"class_weight\s*=\s*['\"]balanced['\"]|sample_weight\s*="), "balancing:class_weights"),

    # Calibration
    (re.compile(r"IsotonicRegression|CalibratedClassifierCV[^)]*isotonic"), "calibration:isotonic"),
    (re.compile(r"CalibratedClassifierCV[^)]*sigmoid"), "calibration:platt"),

    # Optimizers
    (re.compile(r"\boptim\.AdamW\(|AdamW\("), "optimizer:adamw"),
    (re.compile(r"SGD\([^)]*nesterov\s*=\s*True"), "optimizer:sgd_nesterov"),
    (re.compile(r"\boptim\.Adam\("), "optimizer:adam"),
    (re.compile(r"\bRMSprop\("), "optimizer:rmsprop"),
    (re.compile(r"\bLion\("), "optimizer:lion"),

    # Schedulers
    (re.compile(r"CosineAnnealing(LR|WarmRestarts)|cosine_schedule_with"), "scheduler:cosine"),
    (re.compile(r"OneCycleLR"), "scheduler:onecycle"),
    (re.compile(r"get_linear_schedule_with_warmup|linear_warmup"), "scheduler:linear_warmup"),
    (re.compile(r"ReduceLROnPlateau"), "scheduler:reduce_on_plateau"),
]


# ── Public API ────────────────────────────────────────────────────────────


def infer_strategies_from_code(code: str) -> set[str]:
    """Best-effort regex scan of a script. Returns the set of detected tags."""
    found: set[str] = set()
    if not code:
        return found
    for pattern, tag in _INFER_PATTERNS:
        if pattern.search(code):
            found.add(tag)
    return found


_STRATEGIES_LINE_RE = re.compile(
    r"^[ \t#/-]*STRATEGIES[ \t]*[:=][ \t]*(.+?)$",
    re.MULTILINE | re.IGNORECASE,
)


def parse_strategies_from_response(response_text: str) -> set[str]:
    """Read the 'STRATEGIES: tag1, tag2' line the LLM is asked to emit.

    Tolerant of: leading whitespace / hash / dash, trailing punctuation,
    backticks, single/double quotes, and either ``:`` or ``=`` separator.
    Tags not in the controlled vocabulary are silently dropped.
    """
    if not response_text:
        return set()
    matches = _STRATEGIES_LINE_RE.findall(response_text)
    if not matches:
        return set()
    raw = matches[-1]  # the last STRATEGIES line wins (LLMs sometimes preview)
    tags: set[str] = set()
    for token in raw.split(","):
        t = token.strip().strip("`").strip("'\"").strip(".").strip()
        if t and t in STRATEGY_VOCAB:
            tags.add(t)
    return tags


def collect_strategies(response_text: str, code: str) -> set[str]:
    """Combine LLM-declared + code-inferred strategies. Both sources are valued."""
    return parse_strategies_from_response(response_text) | infer_strategies_from_code(code)


# ── Branch history ────────────────────────────────────────────────────────


@dataclass
class BranchHistoryRow:
    node_id: str
    stage: str           # draft | improve | debug
    val: float | None
    holdout: float | None
    delta: float | None  # change vs previous SUCCESSFUL node in this branch
    new_strategies: set[str]
    is_buggy: bool
    is_suspicious: bool


def build_branch_history(journal: "Journal", branch_root_id: str) -> list[BranchHistoryRow]:
    """Walk the lineage rooted at ``branch_root_id`` and assemble a history.

    Nodes are returned in creation order. ``new_strategies`` is the set of
    tags added by this node compared to the cumulative set of all earlier
    nodes in the same branch.
    """
    nodes_in_branch = [
        n for n in journal
        if (getattr(n, "branch_root_id", None) or n.id) == branch_root_id
    ]
    nodes_in_branch.sort(key=lambda n: n.created_at)
    maximize = True
    for n in nodes_in_branch:
        if n.scores.maximize is not None:
            maximize = n.scores.maximize
            break

    rows: list[BranchHistoryRow] = []
    cumulative: set[str] = set()
    prev_val: float | None = None
    for n in nodes_in_branch:
        node_strats = getattr(n, "strategies", set()) or set()
        new_strats = node_strats - cumulative
        delta = None
        if n.val_score is not None and prev_val is not None:
            delta = (n.val_score - prev_val) if maximize else (prev_val - n.val_score)
        rows.append(BranchHistoryRow(
            node_id=n.id,
            stage=n.stage,
            val=n.val_score,
            holdout=n.holdout_score,
            delta=delta,
            new_strategies=new_strats,
            is_buggy=n.is_buggy,
            is_suspicious=n.is_suspicious,
        ))
        if n.val_score is not None and not n.is_buggy:
            prev_val = n.val_score
        cumulative |= node_strats
    return rows


def cumulative_strategies(rows: list[BranchHistoryRow]) -> set[str]:
    out: set[str] = set()
    for r in rows:
        out |= r.new_strategies
    return out


def branch_distinct_strategies(journal: "Journal", branch_root_id: str) -> set[str]:
    """Return the set of strategies tried anywhere in the branch lineage."""
    out: set[str] = set()
    for n in journal:
        if (getattr(n, "branch_root_id", None) or n.id) == branch_root_id:
            out |= getattr(n, "strategies", set()) or set()
    return out


def render_branch_history_table(rows: list[BranchHistoryRow]) -> str:
    """Markdown table for prompt injection."""
    if not rows:
        return "(no history)"
    lines = [
        "| node | stage   | val      | holdout  | delta    | strategies added                  | flags |",
        "|------|---------|----------|----------|----------|-----------------------------------|-------|",
    ]
    for r in rows:
        val_str = f"{r.val:.4f}" if r.val is not None else "N/A"
        holdout_str = f"{r.holdout:.4f}" if r.holdout is not None else "N/A"
        if r.is_buggy:
            val_str += " (buggy)"
        delta_str = f"{r.delta:+.4f}" if r.delta is not None else "        "
        strats = ", ".join(sorted(r.new_strategies)) if r.new_strategies else "(none new)"
        flags = ",".join(
            flag for flag, active in (("buggy", r.is_buggy), ("suspect", r.is_suspicious))
            if active
        ) or "-"
        lines.append(
            f"| {r.node_id:<4} | {r.stage:<7} | {val_str:<8} | {holdout_str:<8} | {delta_str:<8} | {strats[:33]:<33} | {flags:<5} |"
        )
    return "\n".join(lines)


# ── Untried-list + required-strategy picker ───────────────────────────────


# Per-task ranked menus. Higher-priority strategies appear earlier so the
# required-strategy picker walks them in order.
_TABULAR_MENU = [
    "fe:delimited_split",
    "fe:datetime_expansion",
    "fe:target_encoding_oof",
    "fe:frequency_encoding",
    "fe:interaction_features",
    "fe:aggregation_groupby",
    "fe:row_stats",
    "fe:numeric_binning",
    "hp:optuna",
    "ensemble:seed_averaging",
    "ensemble:cv_fold_averaging",
    "ensemble:in_script",
    "ensemble:stacking",
    "calibration:isotonic",
    "post:threshold_tuning",
    "pseudo_labeling",
    "balancing:class_weights",
]

_VISION_MENU = [
    "data_augmentation",
    "tta:hflip",
    "tta:multi_view",
    "scheduler:cosine",
    "scheduler:onecycle",
    "optimizer:sgd_nesterov",
    "optimizer:lion",
    "ensemble:cv_fold_averaging",
    "ensemble:seed_averaging",
    "pseudo_labeling",
]

_NLP_MENU = [
    "scheduler:linear_warmup",
    "scheduler:cosine",
    "ensemble:seed_averaging",
    "ensemble:cv_fold_averaging",
    "post:threshold_tuning",
    "pseudo_labeling",
]

_AUDIO_MENU = [
    "data_augmentation",
    "tta:multi_view",
    "scheduler:cosine",
    "optimizer:adamw",
    "ensemble:seed_averaging",
    "ensemble:cv_fold_averaging",
    "pseudo_labeling",
]

_TIMESERIES_MENU = [
    "fe:lag_features",
    "fe:datetime_expansion",
    "fe:aggregation_groupby",
    "cv:timeseries_split",
    "hp:optuna",
    "ensemble:seed_averaging",
    "ensemble:cv_fold_averaging",
]


def _append_unique(items: list[str], tag: str) -> None:
    if tag not in items:
        items.append(tag)


def _menu_for(
    task_type: str,
    objective: str = "",
    metric_name: str = "",
) -> list[str]:
    if task_type == "vision":
        menu = list(_VISION_MENU)
    elif task_type == "nlp":
        menu = list(_NLP_MENU)
    elif task_type == "audio":
        menu = list(_AUDIO_MENU)
    elif task_type == "timeseries":
        menu = list(_TIMESERIES_MENU)
    else:
        menu = list(_TABULAR_MENU)

    if objective in {"binary_classification", "multiclass_classification", "classification", "multilabel_classification"}:
        if metric_name == "logloss":
            _append_unique(menu, "calibration:isotonic")
            _append_unique(menu, "calibration:platt")
        if metric_name in {"auc", "ranking_metric", "correlation", "map"}:
            _append_unique(menu, "ensemble:rank_average")
        if metric_name in {"accuracy", "f1", "classification_metric", "overlap_metric"}:
            _append_unique(menu, "post:threshold_tuning")
    if objective == "recommendation":
        _append_unique(menu, "fe:aggregation_groupby")
        _append_unique(menu, "ensemble:rank_average")
    if objective in {"multilabel_classification", "binary_classification"}:
        _append_unique(menu, "balancing:class_weights")
    return menu


def untried_strategies(
    used: set[str],
    task_type: str = "tabular",
    objective: str = "",
    metric_name: str = "",
) -> list[str]:
    """Return the menu items that haven't been used yet (in priority order)."""
    return [s for s in _menu_for(task_type, objective, metric_name) if s not in used]


def pick_required_strategy(
    *,
    fraction_used: float,
    used: set[str],
    task_type: str,
    objective: str = "",
    metric_name: str = "",
) -> str | None:
    """Choose ONE strategy the next improve must apply.

    Time-aware exploration phases:
        Phase 1 (early,  fraction_used <= 0.30): broaden FE
        Phase 2 (mid,    0.30 < fraction_used <= 0.55): hyperparameter search
        Phase 3 (late,   0.55 < fraction_used <= 0.80): ensembling / pseudo-labeling
        Phase 4 (final,  fraction_used > 0.80): only safe small wins

    Returns None if every relevant strategy has already been tried (let the
    LLM choose its own move).
    """
    untried = untried_strategies(
        used,
        task_type=task_type,
        objective=objective,
        metric_name=metric_name,
    )
    if not untried:
        return None

    def _first(matches: Iterable[str]) -> str | None:
        for m in matches:
            if m in untried:
                return m
        return None

    if task_type == "timeseries":
        if fraction_used <= 0.30:
            return _first([
                "cv:timeseries_split",
                "fe:lag_features",
                "fe:datetime_expansion",
                "fe:aggregation_groupby",
            ]) or _first(untried)
        if fraction_used <= 0.55:
            return _first([
                "hp:optuna",
                "fe:lag_features",
                "ensemble:seed_averaging",
            ]) or _first(untried)
        if fraction_used <= 0.80:
            return _first([
                "ensemble:cv_fold_averaging",
                "ensemble:seed_averaging",
            ]) or _first(untried)
        return _first(["ensemble:cv_fold_averaging", "fe:lag_features"]) or _first(untried)

    if task_type == "audio":
        if fraction_used <= 0.30:
            return _first(["data_augmentation", "scheduler:cosine", "optimizer:adamw"]) or _first(untried)
        if fraction_used <= 0.55:
            return _first(["tta:multi_view", "ensemble:seed_averaging"]) or _first(untried)
        if fraction_used <= 0.80:
            return _first(["ensemble:cv_fold_averaging", "pseudo_labeling"]) or _first(untried)
        return _first(["ensemble:cv_fold_averaging", "tta:multi_view"]) or _first(untried)

    # Tabular phase plan
    if task_type not in ("vision", "nlp"):
        if fraction_used <= 0.30:
            # Broaden feature engineering first.
            return _first([
                "fe:delimited_split",
                "fe:datetime_expansion",
                "fe:target_encoding_oof",
                "fe:frequency_encoding",
                "fe:interaction_features",
                "fe:aggregation_groupby",
            ]) or _first(untried)
        if fraction_used <= 0.55:
            # Hyperparameter search is the highest-leverage mid-phase move.
            return _first([
                "hp:optuna",
                "fe:row_stats",
                "fe:numeric_binning",
            ]) or _first(untried)
        if fraction_used <= 0.80:
            # Ensembling and stability tricks.
            late = [
                "ensemble:seed_averaging",
                "ensemble:cv_fold_averaging",
                "ensemble:in_script",
                "ensemble:stacking",
                "pseudo_labeling",
            ]
            if metric_name == "logloss":
                late.extend(["calibration:isotonic", "calibration:platt"])
            elif metric_name in {"auc", "ranking_metric", "correlation", "map"}:
                late.append("ensemble:rank_average")
            else:
                late.append("post:threshold_tuning")
            return _first(late) or _first(untried)
        # Late stage — only safe small wins.
        final_moves = ["ensemble:cv_fold_averaging"]
        if metric_name == "logloss":
            final_moves.extend(["calibration:isotonic", "calibration:platt"])
        elif metric_name in {"accuracy", "f1", "classification_metric", "overlap_metric"}:
            final_moves.append("post:threshold_tuning")
        else:
            final_moves.append("ensemble:seed_averaging")
        return _first(final_moves) or _first(untried)

    # Vision phase plan
    if task_type == "vision":
        if fraction_used <= 0.30:
            return _first(["data_augmentation", "tta:hflip", "scheduler:cosine"]) or _first(untried)
        if fraction_used <= 0.55:
            return _first(["scheduler:onecycle", "optimizer:sgd_nesterov", "tta:multi_view"]) or _first(untried)
        if fraction_used <= 0.80:
            return _first(["ensemble:cv_fold_averaging", "ensemble:seed_averaging", "pseudo_labeling"]) or _first(untried)
        return _first(["tta:hflip"])

    # NLP phase plan
    if fraction_used <= 0.30:
        return _first(["scheduler:linear_warmup", "scheduler:cosine"]) or _first(untried)
    if fraction_used <= 0.55:
        mid = ["ensemble:seed_averaging"]
        if objective not in {"qa", "span_extraction", "seq2seq"}:
            mid.append("post:threshold_tuning")
        return _first(mid) or _first(untried)
    if fraction_used <= 0.80:
        return _first(["ensemble:cv_fold_averaging", "pseudo_labeling"]) or _first(untried)
    late = ["ensemble:cv_fold_averaging"]
    if objective not in {"qa", "span_extraction", "seq2seq"}:
        late.append("post:threshold_tuning")
    return _first(late) or _first(untried)


def render_strategy_menu(
    task_type: str = "tabular",
    objective: str = "",
    metric_name: str = "",
) -> str:
    """Compact list of task-relevant tags for prompt injection."""
    menu = _menu_for(task_type, objective, metric_name)
    parts = [f"`{t}`" for t in menu]
    return ", ".join(parts)
