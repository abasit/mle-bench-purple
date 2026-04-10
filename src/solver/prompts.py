"""All prompts. One LLM call per operator. Lean, precise, no fluff."""

from __future__ import annotations

from textwrap import dedent

from .nodes import SearchNode
from .task_classify import TaskProfile
from .utils import fmt_seconds, truncate_tail


_DEFAULT_PROFILE = TaskProfile()

_METRIC_GUIDANCE: dict[str, str] = {
    "auc": (
        "METRIC TACTICS (AUC/ROC-AUC): Focus on ranking, not calibration. "
        "Use predict_proba() or decision_function(), never hard labels."
    ),
    "logloss": (
        "METRIC TACTICS (LogLoss): Calibration matters. Use probability outputs, "
        "clip them to [1e-15, 1-1e-15], and keep multiclass rows normalized."
    ),
    "accuracy": (
        "METRIC TACTICS (Accuracy): Final hard labels matter. Tune thresholds if the "
        "submission format allows probabilities."
    ),
    "f1": (
        "METRIC TACTICS (F1): Threshold tuning is usually necessary. Prefer "
        "probability-producing models and search thresholds on validation."
    ),
    "error_metric": (
        "METRIC TACTICS (RMSE/MSE/MAE/RMSLE): Preserve the target scale and do not "
        "rank-normalize regression outputs."
    ),
    "correlation": (
        "METRIC TACTICS (Pearson/Spearman): Ranking and monotonic trends matter more "
        "than exact calibration."
    ),
    "map": (
        "METRIC TACTICS (mAP/MAP): Ranking quality matters most. For detection, "
        "post-processing and threshold choices matter almost as much as the model."
    ),
    "ranking_metric": (
        "METRIC TACTICS (NDCG/MRR/Ranking): Preserve per-query ordering and build "
        "validation around the query/group structure."
    ),
    "overlap_metric": (
        "METRIC TACTICS (IoU/Dice/Jaccard): Threshold tuning and post-processing matter."
    ),
    "classification_metric": (
        "METRIC TACTICS (Precision/Recall): These are threshold-sensitive. Pick "
        "thresholds on validation, not by default."
    ),
}


SYSTEM_PROMPT = dedent("""
You are a Kaggle Grandmaster. You write complete, runnable Python scripts.

OUTPUT FORMAT:
1. A ```python block with the full solution.py
2. After the code block: STRATEGIES: tag1, tag2, ...

The script runs as `python solution.py` in a directory with `./input/` containing
competition data. It must produce `./submission.csv` and print these lines:
    FINAL VAL SCORE: <number>
    FINAL HOLDOUT SCORE: <number>
    METRIC DIRECTION: maximize|minimize
If cross-validation is used, also print:
    CV FOLD SCORES: <score1>, <score2>, ...

RULES:
- submission.csv must match sample_submission.csv columns, dtypes, and row count exactly.
- Follow the task route below in order: output contract first, validation design second, leakage controls third, model choice fourth.
- Build a local validation scheme that matches the benchmark split as closely as possible.
- Use group-aware splits when leakage by patient/user/item/video/source is plausible.
- Use time-ordered validation for temporal data. Do not use shuffled or IID CV for temporal tasks.
- Any target-dependent features or encodings must be fit in-fold only.
- Prefer one strong model at a time. Do not ensemble, stack, blend submissions, or average multiple model families.
- SEED = 42 everywhere. Suppress warnings. No pip install.
- CatBoost may train on raw string categoricals via `cat_features`; LightGBM/XGBoost/sklearn models must receive fully numeric matrices.
- Before fitting a numeric-only model, encode every object/category column and assert no object/category dtypes remain in the model matrix.
- Avoid pandas categorical mutation traps: fill missing values before encoding, and do not call `fillna('Unknown')` on a categorical column unless that category already exists.
- Never use pandas nullable BooleanDtype (`astype('boolean')`) for feature columns that you later fill with string sentinels or pass into encoders.
- Before OneHotEncoder, LabelEncoder, or `pd.factorize` on feature columns, cast bool/object/category inputs to string and fill missing with `'missing'` so the encoder sees one uniform type.
- Never wrap training in try/except with a constant fallback; the runner detects and rejects constant-prediction submissions.
- For bool/string targets: `y.map({'True':1,'False':0})`; never `astype(int)` on strings.
- For multiclass probability submissions, preserve sample_submission column order and renormalize each row to sum to 1 after any clipping or post-processing.
- For multilabel probability submissions, treat targets independently; do NOT force rows to sum to 1.
""").strip()


def _profile(task_profile: TaskProfile | None) -> TaskProfile:
    return task_profile or _DEFAULT_PROFILE


def _metric_guidance_block(task_profile: TaskProfile | None) -> str:
    profile = _profile(task_profile)
    metric = profile.metric_name or ""
    if not metric or metric == "unknown":
        return ""
    if metric in _METRIC_GUIDANCE:
        return _METRIC_GUIDANCE[metric]
    for key, advice in _METRIC_GUIDANCE.items():
        if metric in key or key in metric:
            return advice
    return ""


def _session_reuse_block() -> str:
    return (
        "SESSION RUNTIME:\n"
        "- Before this script runs, persistable globals from the parent attempt are restored.\n"
        "- Reuse already-loaded data, features, encoders, and trained models when the approach is still compatible.\n"
        "- Check `SESSION_RESTORED` and `globals()` before retraining expensive components.\n"
        "- If the new approach is incompatible with old state, overwrite the stale globals explicitly."
    )


def _working_baseline_block() -> str:
    return (
        "WORKING BASELINE FIRST:\n"
        "- First priority is a clean run with an exact sample_submission schema and an honest validation score.\n"
        "- Prefer the simplest strong baseline that matches the task route before adding fragile feature engineering.\n"
        "- If the current approach is brittle, simplify the preprocessing or model family instead of stacking more tricks on top."
    )


def _grounding_block() -> str:
    return (
        "GROUNDING RULES:\n"
        "- Use the actual files under ./input/ and the observed columns as the source of truth.\n"
        "- Infer submission columns, dtypes, and row count from sample_submission.csv, not memory.\n"
        "- If anything is ambiguous, inspect the files/columns in code before training rather than guessing."
    )


def _tabular_guidance_block(task_profile: TaskProfile | None) -> str:
    profile = _profile(task_profile)
    if profile.category != "tabular":
        return ""
    return (
        "TABULAR PLAYBOOK:\n"
        "- Audit schema first: identifier-like columns, composite strings, booleans, datetimes, repeated-group keys, missingness patterns, and high-cardinality categoricals.\n"
        "- Add only schema-supported features: missing indicators, safe delimiter splits, frequency/count encodings, row totals or ratios, and group aggregates only when they are leakage-safe.\n"
        "- Model-family contract: CatBoost for raw string categoricals; LightGBM/XGBoost only after integer/one-hot/frequency encoding; sklearn numeric models only after every feature column is numeric.\n"
        "- For numeric-only models, rebuild `X_train/X_valid/X_test` from the raw frames after encoding and assert no `object` or `category` columns remain before fit.\n"
        "- Keep feature dtypes uniform for encoders: convert bool/object/category inputs to string before OneHotEncoder/LabelEncoder/factorize, and do not mix bool with string sentinels.\n"
        "- Avoid `astype('boolean')` on feature columns unless they stay boolean all the way into the model; prefer int mapping or string normalization instead.\n"
        "- When switching model families inside a restored session, overwrite stale matrices and encoders instead of reusing CatBoost-ready or category-mutated state.\n"
        "- Keep preprocessing identical across train and test, and fit any target-aware encoding strictly in-fold.\n"
        "- Stay competition agnostic: infer feature ideas from column structure and distributions, not from hardcoded column names."
    )


def _draft_variant_hint(variant: int, task_profile: TaskProfile | None) -> str:
    profile = _profile(task_profile)
    route = profile.route_id
    route_pools: dict[str, list[str]] = {
        "tabular_binary_prob": [
            "CatBoostClassifier with predict_proba(), stratified validation, missing-indicator features, and probability clipping for the final submission.",
            "LightGBM binary classifier with boolean cleanup, integer/frequency encoding for categoricals, and an explicit assertion that the model matrix contains no object/category dtypes.",
            "XGBoost binary classifier on a fully numeric encoded matrix only, with one-hot/frequency encoding by cardinality and probability outputs from predict_proba().",
        ],
        "tabular_binary_label": [
            "CatBoostClassifier with stratified CV, missingness-aware preprocessing, and threshold tuning on validation before converting probabilities to labels.",
            "LightGBM binary classifier with class balancing, explicit threshold search on validation, and a fully numeric encoded feature matrix.",
            "XGBoost binary classifier with conservative preprocessing, explicit numeric encoding of categoricals, and threshold tuning from validation scores.",
        ],
        "tabular_multiclass_prob": [
            "CatBoostClassifier with loss_function='MultiClass', stratified CV, and submission built from predict_proba() columns aligned to sample_submission order.",
            "LightGBM multiclass model with label encoding, boolean cleanup, explicit class-column alignment, and a numeric-only model matrix.",
            "XGBoost multiclass softprob model with careful label mapping, numeric encoding of categoricals, and row-wise probability renormalization before writing submission.",
        ],
        "tabular_multilabel": [
            "One-vs-rest CatBoost or LightGBM multilabel setup with one model per target column and independent probability outputs.",
            "Independent binary models per target with shared preprocessing, OOF validation per target, and no row normalization.",
            "A multilabel one-vs-rest baseline with CatBoost or XGBoost on fully numeric encoded matrices, class weights for rare targets, and per-target calibration if needed.",
        ],
        "tabular_regression": [
            "CatBoostRegressor with robust tabular preprocessing, KFold CV, and continuous predictions averaged across folds.",
            "LightGBM regressor with datetime/frequency features, boolean cleanup, conservative regularization, and a numeric-only model matrix.",
            "XGBoost regressor with mixed one-hot/frequency encoding, row statistics where justified by the schema, and no object/category columns at fit time.",
        ],
        "tabular_multioutput_regression": [
            "One model per target column with shared preprocessing and continuous outputs aligned to the multi-column sample submission.",
            "Multi-target regression via independent fold-averaged LightGBM models on fully numeric encoded features, preserving raw target scales for every output column.",
            "A conservative multi-output regression pipeline with CatBoost or XGBoost per target and careful column alignment after rebuilding numeric matrices.",
        ],
        "timeseries_forecasting": [
            "Walk-forward forecasting with strict time sorting, lag/rolling/calendar features, and no shuffle anywhere.",
            "Grouped time-series baseline with past-only lag features, TimeSeriesSplit, and fold averaging over continuous forecasts.",
            "A forecasting baseline built from lag windows, rolling stats, and horizon-safe validation that mirrors the submission horizon.",
        ],
        "vision_segmentation": [
            "U-Net or FPN with a timm encoder, BCE+Dice loss, and threshold tuning on validation masks.",
            "A lightweight U-Net with strong resize/crop augmentation, mixed precision if GPU exists, and patient/source-aware validation.",
            "FPN with an efficient encoder, fold-level averaging, and simple mask post-processing.",
        ],
        "vision_object_detection": [
            "Torchvision FasterRCNN or RetinaNet with validation grouped by source/video and explicit confidence/NMS tuning.",
            "A conservative FasterRCNN baseline with pretrained weights, restrained augmentations, and mAP-oriented validation.",
            "RetinaNet/FasterRCNN with careful box filtering and post-processing that matches the sample format exactly.",
        ],
        "vision_classification": [
            "timm efficientnet/convnext image classifier with stratified or grouped folds, probability outputs, and simple TTA.",
            "torchvision/timm classification baseline with fold averaging, augmentation, and metric-aligned validation.",
            "A pretrained image classifier with conservative augmentations and fold-level prediction averaging.",
        ],
        "audio_classification": [
            "Mel-spectrogram plus CNN classifier with grouped folds when speaker/source leakage is plausible.",
            "Torchaudio spectrogram pipeline with time/frequency masking, fold averaging, and probability outputs.",
            "A spectrogram classification baseline with conservative augmentations and route-aware validation.",
        ],
        "nlp_qa": [
            "HuggingFace extractive QA model with fast tokenization, doc stride, and span-based post-processing.",
            "A QA transformer baseline that predicts start/end positions instead of class labels.",
            "DeBERTa/RoBERTa QA head with context-window handling tuned for the task description.",
        ],
        "nlp_seq2seq": [
            "A sequence-to-sequence transformer baseline with task-specific text normalization and deterministic decoding.",
            "A compact seq2seq baseline with constrained generation and validation on exact task outputs.",
            "A text-to-text baseline that treats the task as generation, not classification.",
        ],
        "nlp_classification": [
            "DeBERTa/RoBERTa text classifier with stratified/group-aware validation and probability outputs from the classifier head.",
            "TF-IDF + LightGBM text baseline with stratified CV as a strong CPU-safe reference.",
            "A transformer text-classification baseline with validation aligned to the competition leakage risks.",
        ],
        "tabular_ranking": [
            "LightGBM ranker or group-wise ranking baseline with validation grouped by query/user/item.",
            "A ranking baseline that preserves per-query structure and uses rank-oriented validation metrics.",
            "A grouped recommendation/ranking baseline with query-aware feature aggregation.",
        ],
    }
    pool = route_pools.get(route)
    if pool is None:
        fallback = {
            "vision": route_pools["vision_classification"],
            "nlp": route_pools["nlp_classification"],
            "audio": route_pools["audio_classification"],
            "timeseries": route_pools["timeseries_forecasting"],
        }
        pool = fallback.get(profile.category, route_pools["tabular_regression"])
    return pool[variant % len(pool)]


def _route_blocks(task_profile: TaskProfile | None) -> str:
    """Compact single-block task route: output, validation, objective, leakage, metric."""
    profile = _profile(task_profile)
    parts: list[str] = []

    # One-line output contract.
    targets = ", ".join(profile.sample_target_cols) if profile.sample_target_cols else "see sample_submission"
    parts.append(f"OUTPUT: {profile.prediction_mode} → {targets}")

    # One-line validation.
    val_mode = profile.validation_mode
    if val_mode == "time_ordered":
        time_cols = ", ".join(profile.likely_time_cols) if profile.likely_time_cols else "detect"
        parts.append(f"VALIDATION: time-ordered split on {time_cols} — NO shuffle, no IID/random CV")
    elif val_mode in ("stratified_group", "grouped"):
        group_cols = ", ".join(profile.likely_group_cols) if profile.likely_group_cols else "detect"
        parts.append(f"VALIDATION: group-aware split on {group_cols} — IID split primitives are invalid here")
    elif val_mode == "stratified":
        parts.append("VALIDATION: stratified CV")
    else:
        parts.append("VALIDATION: IID CV (confirm no time/group leakage first)")

    # Leakage warnings — only if relevant.
    leakage_items: list[str] = []
    if profile.likely_time_cols:
        leakage_items.append(f"time cols: {', '.join(profile.likely_time_cols)}")
    if profile.likely_group_cols:
        leakage_items.append(f"group cols: {', '.join(profile.likely_group_cols)}")
    if leakage_items:
        parts.append(f"LEAKAGE RISK: {'; '.join(leakage_items)}")

    # Metric guidance — one line.
    metric_line = _metric_guidance_block(task_profile)
    if metric_line:
        parts.append(metric_line)

    return "TASK ROUTE:\n" + "\n".join(f"- {p}" for p in parts)


def build_draft_prompt(
    *,
    task_desc: str,
    task_type: str,
    data_files: list[str],
    sample_sub_preview: str,
    kb_card: str,
    data_preview: str,
    task_profile_summary: str = "",
    task_profile: TaskProfile | None = None,
    env_summary: str = "",
    time_remaining: float,
    variant: int,
    total_variants: int,
) -> list[dict[str, str]]:
    from .strategies import render_strategy_menu

    profile = _profile(task_profile)
    files_str = "\n".join(f"  {value}" for value in data_files[:20])
    menu = render_strategy_menu(task_type, objective=profile.objective, metric_name=profile.metric_name)
    variant_hint = _draft_variant_hint(variant, profile)
    tabular_guidance = _tabular_guidance_block(profile)

    user = dedent(f"""
{task_desc.strip()}

Environment: {env_summary.strip()}
Time remaining: {fmt_seconds(time_remaining)}
Data files: {files_str}

{task_profile_summary.strip()}

{_route_blocks(profile)}

{_working_baseline_block()}

{_grounding_block()}

{tabular_guidance}

Data preview:
{data_preview.strip()}

Sample submission:
{sample_sub_preview.strip()}

{kb_card.strip()}

Strategy tags: {menu}

VARIANT {variant}/{total_variants}: {variant_hint}

Write the complete solution.py. End with STRATEGIES: tag1, tag2, ...
""").strip()

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def _classify_error(error_summary: str, cleaned_log: str) -> tuple[str, str]:
    combined = (error_summary + "\n" + cleaned_log).lower()

    if any(keyword in combined for keyword in ("out of memory", "oom", "cuda out of memory", "allocat", "memory error", "killed")):
        return "oom", (
            "OOM FIX TACTICS:\n"
            "- Reduce batch size substantially\n"
            "- Use mixed precision if GPU exists\n"
            "- Reduce image size or max_length\n"
            "- Stream or chunk large datasets instead of materializing everything"
        )
    if any(keyword in combined for keyword in ("shape mismatch", "cannot reshape", "size mismatch", "broadcast", "matmul")):
        return "shape_mismatch", (
            "SHAPE FIX TACTICS:\n"
            "- Print shapes at key points\n"
            "- Ensure train/test feature matrices align after preprocessing\n"
            "- Check target shape and classifier/regressor output shape"
        )
    if any(keyword in combined for keyword in ("filenotfounderror", "no such file", "not a directory", "permission denied")):
        return "data_loading", (
            "DATA LOADING FIX TACTICS:\n"
            "- Discover actual files under ./input/ before hardcoding paths\n"
            "- Check nested directories and file extensions"
        )
    if any(keyword in combined for keyword in ("modulenotfounderror", "importerror", "no module named")):
        return "import_error", (
            "IMPORT FIX TACTICS:\n"
            "- Do not pip install\n"
            "- Replace the missing dependency with a preinstalled alternative"
        )
    if any(
        keyword in combined for keyword in (
            "could not convert string to float",
            "invalid columns:",
            "enable_categorical",
            "dataframe.dtypes for data must be",
            "categorical with a new category",
            "invalid value 'unknown' for dtype 'boolean'",
            "dtype 'boolean'",
            "uniformly strings or numbers",
            "got ['bool', 'str']",
            "got ['str', 'bool']",
        )
    ):
        return "categorical_model_mismatch", (
            "CATEGORICAL/MODEL MISMATCH FIX TACTICS:\n"
            "- CatBoost can consume raw string categoricals via cat_features; numeric-only models cannot\n"
            "- For LightGBM/XGBoost/sklearn models, rebuild X_train/X_valid/X_test from raw frames and encode every object/category column first\n"
            "- Assert that no object/category dtypes remain before fit\n"
            "- Fill missing values before encoding, not after converting to pandas Categorical\n"
            "- Do not use `astype('boolean')` on feature columns that later receive string sentinels such as 'missing' or 'Unknown'\n"
            "- Before OneHotEncoder/LabelEncoder/pd.factorize, cast categorical and bool feature columns to string and fill missing with 'missing' so the encoder sees a single dtype"
        )
    if any(keyword in combined for keyword in ("typeerror", "unsupported operand", "not callable")):
        return "type_error", (
            "TYPE FIX TACTICS:\n"
            "- Check dtype conversions carefully\n"
            "- Avoid astype(int) on string labels and rebuild numeric-only matrices before fitting numeric models"
        )
    if any(keyword in combined for keyword in ("keyerror", "not in index", "column", "not found")):
        return "key_error", (
            "KEY FIX TACTICS:\n"
            "- Print actual column names and reconcile train/test/sample schema\n"
            "- Reindex after one-hot encoding when needed"
        )
    if any(keyword in combined for keyword in ("timeout", "timed out", "time limit")):
        return "timeout", (
            "TIMEOUT FIX TACTICS:\n"
            "- Reduce folds, estimators, epochs, or image/text sizes\n"
            "- Prefer simpler baselines over partial heavy training"
        )
    if any(keyword in combined for keyword in ("cuda", "gpu", "device", "nccl")):
        return "cuda", "CUDA FIX TACTICS:\n- Make device handling explicit and keep model/data on the same device"
    if any(keyword in combined for keyword in ("unicodedecodeerror", "encoding", "codec", "charmap")):
        return "encoding", (
            "ENCODING FIX TACTICS:\n"
            "- Try utf-8 first, then latin-1, and inspect whether the file is actually parquet/binary"
        )
    return "general", ""


def build_debug_prompt(
    *,
    parent: SearchNode,
    error_summary: str,
    cleaned_log: str,
    time_remaining: float,
    data_preview: str = "",
    task_profile_summary: str = "",
    task_profile: TaskProfile | None = None,
    env_summary: str = "",
    prior_attempts: list[SearchNode] | None = None,
) -> list[dict[str, str]]:
    prior_block = ""
    escalation = ""
    if prior_attempts:
        lines = []
        errors = []
        for node in prior_attempts[-3:]:
            err = (node.result.error_summary if node.result else "") or "unknown"
            lines.append(f"- {node.id}: {err}")
            errors.append(err)
        prior_block = "Prior failed attempts:\n" + "\n".join(lines)
        if len(errors) >= 2 and len({err.split(":")[0] for err in errors}) == 1:
            escalation = "\nREPEATED FAILURE. Stop making tiny tweaks. Change the failing component or simplify the approach."

    error_class, error_advice = _classify_error(error_summary, cleaned_log)
    profile = _profile(task_profile)
    route_block = _route_blocks(profile)
    metric_block = _metric_guidance_block(task_profile)
    tabular_guidance = _tabular_guidance_block(profile)

    user = dedent(f"""
Fix this crashed solution. Output the COMPLETE corrected solution.py.

Error: {error_summary}
Error class: {error_class}
Time remaining: {fmt_seconds(time_remaining)}
Environment: {env_summary.strip()}

{error_advice}

{task_profile_summary.strip()}

{route_block}

{metric_block}

{_working_baseline_block()}

{_grounding_block()}

{tabular_guidance}

{_session_reuse_block()}

Data / exploration context:
{data_preview.strip() or '(none)'}

Log tail:
```
{truncate_tail(cleaned_log, 3000)}
```

{prior_block}{escalation}

Previous code:
```python
{parent.code}
```

Fix the root cause. Keep FINAL VAL SCORE / FINAL HOLDOUT SCORE / METRIC DIRECTION prints.
Do NOT add constant-prediction fallbacks. End with STRATEGIES: tag1, tag2, ...
""").strip()

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def build_repair_prompt(
    *,
    code: str,
    error_summary: str,
    cleaned_log: str,
    time_remaining: float,
    repair_attempt: int,
    total_repairs: int,
    data_preview: str = "",
    task_profile_summary: str = "",
    task_profile: TaskProfile | None = None,
    env_summary: str = "",
) -> list[dict[str, str]]:
    error_class, error_advice = _classify_error(error_summary, cleaned_log)
    profile = _profile(task_profile)
    route_block = _route_blocks(profile)
    metric_block = _metric_guidance_block(task_profile)
    tabular_guidance = _tabular_guidance_block(profile)

    user = dedent(f"""
Repair this solution using the real execution failure. Output the COMPLETE corrected solution.py.

Repair attempt: {repair_attempt}/{total_repairs}
Error: {error_summary}
Error class: {error_class}
Time remaining: {fmt_seconds(time_remaining)}
Environment: {env_summary.strip()}

{error_advice}

{task_profile_summary.strip()}

{route_block}

{metric_block}

{_working_baseline_block()}

{_grounding_block()}

{tabular_guidance}

Data / exploration context:
{data_preview.strip() or '(none)'}

Failure log:
```
{truncate_tail(cleaned_log, 3000)}
```

Current code:
```python
{code}
```

Return the simplest corrected version that runs, writes a valid submission.csv, matches sample_submission exactly, and prints the required score lines.
If the current approach is fragile, replace it with a simpler baseline rather than preserving every feature.
End with STRATEGIES: tag1, tag2, ...
""").strip()

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def build_improve_prompt(
    *,
    parent: SearchNode,
    journal_summary: str,
    time_remaining: float,
    fraction_used: float,
    data_preview: str = "",
    task_profile_summary: str = "",
    task_profile: TaskProfile | None = None,
    env_summary: str = "",
    branch_history_rows: list | None = None,
    used_strategies: set[str] | None = None,
    untried_strategies: list[str] | None = None,
    required_strategy: str | None = None,
    task_type: str = "tabular",
) -> list[dict[str, str]]:
    from .strategies import STRATEGY_VOCAB, render_branch_history_table
    parent_score = f"{parent.val_score:.5f}" if parent.val_score is not None else "N/A"
    parent_holdout = f"{parent.holdout_score:.5f}" if parent.holdout_score is not None else "N/A"
    profile = _profile(task_profile)
    if parent.scores.maximize is False:
        objective_goal = "lower"
        direction_hint = "Lower validation score is better for this metric."
    elif parent.scores.maximize is True:
        objective_goal = "higher"
        direction_hint = "Higher validation score is better for this metric."
    else:
        objective_goal = "better"
        direction_hint = "Improve the validation objective in the correct direction."

    history_table = render_branch_history_table(branch_history_rows or [])
    used_str = ", ".join(sorted(used_strategies)) if used_strategies else "(none)"
    untried_str = ", ".join(f"`{tag}`" for tag in (untried_strategies or [])[:10]) or "(all tried)"
    route_block = _route_blocks(profile)
    metric_block = _metric_guidance_block(task_profile)
    tabular_guidance = _tabular_guidance_block(profile)
    suspicion_block = ""
    if parent.suspicion_reasons:
        suspicion_block = "Current runner concerns:\n" + "\n".join(f"- {reason}" for reason in parent.suspicion_reasons[:4])

    if required_strategy:
        req_desc = STRATEGY_VOCAB.get(required_strategy, "")
        req_block = (
            f"REQUIRED STRATEGY: `{required_strategy}` - {req_desc}\n"
            "You MUST apply this and declare it in STRATEGIES."
        )
    else:
        req_block = (
            "REQUIRED STRATEGY: none (the current route menu is exhausted). "
            "Focus on the highest-leverage route-aligned refinement."
        )

    user = dedent(f"""
Improve this solution. Write a NEW complete solution.py that scores {objective_goal}.

Parent {parent.id}: val={parent_score}, holdout={parent_holdout}
Time: {fmt_seconds(time_remaining)} remaining ({fraction_used:.0%} used)
{direction_hint}
Environment: {env_summary.strip()}

{task_profile_summary.strip()}

{route_block}

{metric_block}

{_working_baseline_block()}

{_grounding_block()}

{tabular_guidance}

{_session_reuse_block()}

Data / exploration context:
{data_preview.strip() or '(none)'}

Branch history:
{history_table}

Already tried: {used_str}
Untried: {untried_str}
{req_block}
{suspicion_block}

Other branches: {journal_summary or '(none)'}

Previous code:
```python
{parent.code}
```

REFLECT in 1-2 sentences on why the score is what it is, then PLAN one change, then write the full solution.py.
If the current code looks brittle or overfit, simplifying it is allowed.
End with STRATEGIES: tag1, tag2, ...
""").strip()

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def summarise_other_attempts(nodes: list[SearchNode], skip_id: str | None = None) -> str:
    if not nodes:
        return ""
    lines = []
    for node in nodes:
        if node.id == skip_id:
            continue
        strats = ", ".join(sorted(getattr(node, "strategies", set()) or set()))
        score = f"val={node.val_score:.4f}" if node.val_score is not None else "buggy"
        lines.append(f"{node.id}({node.stage} {score}) [{strats}]")
    return "\n".join(lines[:6])


def code_summary(code: str, max_chars: int = 400) -> str:
    lines = code.splitlines()
    keep = []
    for line in lines:
        stripped = line.strip()
        if any(stripped.startswith(prefix) for prefix in ("import ", "from ", "def ", "class ", "MODEL", "SEED", "TARGET")):
            keep.append(stripped)
        if len("\n".join(keep)) > max_chars:
            break
    return "\n".join(keep)
