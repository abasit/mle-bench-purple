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
- SEED = 42 everywhere. Suppress warnings. No pip install.
- Never use `astype('category')`; convert object columns with `df[c] = df[c].astype(str).fillna('missing')`.
- Never wrap training in try/except with a constant fallback; the runner detects and rejects constant-prediction submissions.
- For bool/string targets: `y.map({'True':1,'False':0})`; never `astype(int)` on strings.
- For multiclass probability submissions, preserve sample_submission column order and renormalize each row to sum to 1 after any blending or clipping.
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


def _output_contract_block(task_profile: TaskProfile | None) -> str:
    profile = _profile(task_profile)
    targets = ", ".join(profile.sample_target_cols) if profile.sample_target_cols else "(inspect sample_submission)"
    mode = profile.prediction_mode
    if mode == "multiclass_probabilities":
        return (
            "OUTPUT CONTRACT:\n"
            f"- Submit probability columns in this exact order: {targets}\n"
            "- Every value must be in [0, 1]\n"
            "- Each row must sum to 1 after post-processing or ensembling\n"
            "- Train a true multiclass model and map class order carefully"
        )
    if mode == "multilabel_probabilities":
        return (
            "OUTPUT CONTRACT:\n"
            f"- Submit one probability column per target: {targets}\n"
            "- Every value must be in [0, 1]\n"
            "- Treat targets independently; do NOT renormalize row sums"
        )
    if mode == "single_probability":
        return (
            "OUTPUT CONTRACT:\n"
            f"- Submit a single probability-like target column: {targets}\n"
            "- Values should stay in [0, 1]\n"
            "- Use probability-producing inference, not hard labels"
        )
    if mode == "single_label":
        return (
            "OUTPUT CONTRACT:\n"
            f"- Submit the expected label column(s): {targets}\n"
            "- Final predictions should be hard labels matching the training label space"
        )
    if mode in {"single_continuous", "multioutput_regression"}:
        return (
            "OUTPUT CONTRACT:\n"
            f"- Submit continuous predictions for: {targets}\n"
            "- Preserve the target scale; do not rank-normalize or quantize outputs"
        )
    if mode == "ranking_scores":
        return (
            "OUTPUT CONTRACT:\n"
            f"- Submit ranking scores for: {targets}\n"
            "- Preserve per-query/per-group ordering and do not collapse to labels"
        )
    if mode == "segmentation_encoding":
        return (
            "OUTPUT CONTRACT:\n"
            "- Follow the sample submission encoding exactly for masks\n"
            "- Tune thresholds and post-processing on validation masks only"
        )
    if mode == "detection_rows":
        return (
            "OUTPUT CONTRACT:\n"
            "- Match the exact detection-row schema and post-processing format of sample_submission\n"
            "- Keep box filtering and confidence thresholds explicit in code"
        )
    return "OUTPUT CONTRACT:\n- Match sample_submission exactly."


def _validation_route_block(task_profile: TaskProfile | None) -> str:
    profile = _profile(task_profile)
    time_cols = ", ".join(profile.likely_time_cols) if profile.likely_time_cols else "(detect from data)"
    group_cols = ", ".join(profile.likely_group_cols) if profile.likely_group_cols else "(none obvious)"
    if profile.validation_mode == "time_ordered":
        return (
            "VALIDATION ROUTE:\n"
            f"- Sort by time before splitting. Likely time columns: {time_cols}\n"
            "- Validation data must be strictly later than training data\n"
            "- No shuffle=True, no random KFold, no leakage from future rows"
        )
    if profile.validation_mode == "stratified_group":
        return (
            "VALIDATION ROUTE:\n"
            f"- Use grouped validation with label balance. Likely group columns: {group_cols}\n"
            "- Prefer StratifiedGroupKFold when feasible; otherwise group by the leakage axis first"
        )
    if profile.validation_mode == "grouped":
        return (
            "VALIDATION ROUTE:\n"
            f"- Use grouped validation. Likely group columns: {group_cols}\n"
            "- Keep all rows from the same group in a single fold"
        )
    if profile.validation_mode == "stratified":
        return (
            "VALIDATION ROUTE:\n"
            "- Use stratified validation because the objective is classification-like\n"
            "- Keep class balance similar across folds"
        )
    return (
        "VALIDATION ROUTE:\n"
        "- Use IID validation only if no time or group leakage axis is present\n"
        "- Prefer CV over a single split unless runtime is prohibitive"
    )


def _objective_route_block(task_profile: TaskProfile | None) -> str:
    profile = _profile(task_profile)
    objective = profile.objective
    if objective == "multiclass_classification":
        return (
            "OBJECTIVE ROUTE:\n"
            "- Use a true multiclass model or a carefully aligned one-vs-rest setup\n"
            "- Keep label encoding and sample_submission class-column order aligned\n"
            "- Do not treat multiclass like multilabel"
        )
    if objective == "multilabel_classification":
        return (
            "OBJECTIVE ROUTE:\n"
            "- Treat each target column as an independent binary task or use a multilabel wrapper\n"
            "- Use sigmoid-style probabilities per target, not a softmax over all targets"
        )
    if objective == "binary_classification":
        return "OBJECTIVE ROUTE:\n- Use a binary classifier and keep label/probability output aligned with the metric."
    if objective in {"regression", "multioutput_regression"}:
        return "OBJECTIVE ROUTE:\n- Use a regression model and preserve numeric target scale throughout."
    if objective == "forecasting":
        return (
            "OBJECTIVE ROUTE:\n"
            "- Build lag, rolling, and calendar features only from past data\n"
            "- Forecasting validation must mirror the deployment horizon"
        )
    if objective == "recommendation":
        return (
            "OBJECTIVE ROUTE:\n"
            "- Treat this as grouped ranking/retrieval rather than plain classification\n"
            "- Build validation around users/queries/items and rank candidates within group"
        )
    if objective == "segmentation":
        return "OBJECTIVE ROUTE:\n- Use a mask model, threshold tuning, and mask-aware validation/post-processing."
    if objective == "object_detection":
        return "OBJECTIVE ROUTE:\n- Use a detection model and explicit post-processing for confidence/NMS."
    if objective in {"qa", "span_extraction"}:
        return "OBJECTIVE ROUTE:\n- Treat this as span prediction, not plain sentence classification."
    if objective == "seq2seq":
        return "OBJECTIVE ROUTE:\n- Use sequence-to-sequence generation or structured text normalization, not classification."
    return "OBJECTIVE ROUTE:\n- Match the modeling family to the inferred objective."


def _leakage_watchlist_block(task_profile: TaskProfile | None) -> str:
    profile = _profile(task_profile)
    lines = ["LEAKAGE WATCHLIST:"]
    if profile.likely_time_cols:
        lines.append(f"- Time leakage risk through: {', '.join(profile.likely_time_cols)}")
    if profile.likely_group_cols:
        lines.append(f"- Group leakage risk through: {', '.join(profile.likely_group_cols)}")
    if "classification" in profile.objective:
        lines.append("- Any target encoding must be out-of-fold only")
    lines.append("- Never use sample_submission values or test-only label proxies during training")
    return "\n".join(lines)


def _stepwise_plan_block(task_profile: TaskProfile | None) -> str:
    profile = _profile(task_profile)
    return (
        "STEPWISE ROUTE:\n"
        f"1. Honor the output contract for `{profile.prediction_mode}`.\n"
        f"2. Build validation for `{profile.validation_mode}`.\n"
        f"3. Choose a `{profile.route_id}` baseline, not a generic `{profile.category}` baseline.\n"
        "4. Apply metric-specific tactics only after the route and validation are correct."
    )


def _draft_variant_hint(variant: int, task_profile: TaskProfile | None) -> str:
    profile = _profile(task_profile)
    route = profile.route_id
    route_pools: dict[str, list[str]] = {
        "tabular_binary_prob": [
            "CatBoostClassifier with predict_proba(), stratified validation, and probability clipping for the final submission.",
            "LightGBM binary classifier with careful categorical handling, stratified CV, and probability outputs from predict_proba().",
            "XGBoost binary classifier with frequency/one-hot encoding split by cardinality and probability outputs only.",
        ],
        "tabular_binary_label": [
            "CatBoostClassifier with stratified CV and threshold tuning on validation before converting probabilities to labels.",
            "LightGBM binary classifier with class balancing and explicit threshold search on validation.",
            "XGBoost binary classifier with a conservative feature pipeline and threshold tuning from validation scores.",
        ],
        "tabular_multiclass_prob": [
            "CatBoostClassifier with loss_function='MultiClass', stratified CV, and submission built from predict_proba() columns aligned to sample_submission order.",
            "LightGBM multiclass model with objective='multiclass', label encoding from train only, and explicit class-column alignment before submission.",
            "XGBoost multiclass softprob model with careful label mapping, stratified CV, and row-wise probability renormalization before writing submission.",
        ],
        "tabular_multilabel": [
            "One-vs-rest LightGBM/CatBoost multilabel setup with one model per target column and independent probability outputs.",
            "Independent binary models per target with shared preprocessing, OOF validation per target, and no row normalization.",
            "Multilabel one-vs-rest baseline with class weights for rare targets and per-target probability calibration if needed.",
        ],
        "tabular_regression": [
            "CatBoostRegressor with robust tabular preprocessing, KFold CV, and continuous predictions averaged across folds.",
            "LightGBM regressor with frequency/datetime features and conservative regularization for stable CV.",
            "XGBoost regressor with mixed one-hot/frequency encoding and continuous fold-averaged predictions.",
        ],
        "tabular_multioutput_regression": [
            "One model per target column with shared preprocessing and continuous outputs aligned to the multi-column sample submission.",
            "Multi-target regression via independent fold-averaged models, preserving raw target scales for every output column.",
            "A conservative multi-output regression pipeline with one regressor per target and careful column alignment.",
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
    return "\n\n".join(
        block
        for block in (
            _stepwise_plan_block(task_profile),
            _output_contract_block(task_profile),
            _validation_route_block(task_profile),
            _objective_route_block(task_profile),
            _leakage_watchlist_block(task_profile),
            _metric_guidance_block(task_profile),
        )
        if block
    )


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

    user = dedent(f"""
{task_desc.strip()}

Environment:
{env_summary.strip()}

Time remaining: {fmt_seconds(time_remaining)}
Data files:
{files_str}

{task_profile_summary.strip()}

{_route_blocks(profile)}

Data preview:
{data_preview.strip()}

Sample submission:
{sample_sub_preview.strip()}

{kb_card.strip()}

Strategy tags (declare in STRATEGIES line): {menu}

VARIANT {variant}/{total_variants}: {variant_hint}

Write the complete solution.py. Adapt the data loading to the actual filenames.
Do not skip the route steps: output contract, validation, leakage controls, then model.
End with STRATEGIES: tag1, tag2, ...
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
    if any(keyword in combined for keyword in ("typeerror", "unsupported operand", "not callable")):
        return "type_error", (
            "TYPE FIX TACTICS:\n"
            "- Check dtype conversions carefully\n"
            "- Avoid astype('category') and avoid astype(int) on string labels"
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

    user = dedent(f"""
Fix this crashed solution. Output the COMPLETE corrected solution.py.

Error: {error_summary}
Error class: {error_class}
Time remaining: {fmt_seconds(time_remaining)}

Environment:
{env_summary.strip()}

{task_profile_summary.strip()}

{_route_blocks(profile)}

{error_advice}

Log tail:
```
{truncate_tail(cleaned_log, 3000)}
```

Data preview:
{data_preview.strip()}

{prior_block}{escalation}

Previous code:
```python
{parent.code}
```

Fix the root cause without violating the task route. Keep FINAL VAL SCORE / FINAL HOLDOUT SCORE / METRIC DIRECTION prints.
If CV is used, print CV FOLD SCORES as well. Do NOT add constant-prediction fallbacks.
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

    profile = _profile(task_profile)
    parent_score = f"{parent.val_score:.5f}" if parent.val_score is not None else "N/A"
    parent_holdout = f"{parent.holdout_score:.5f}" if parent.holdout_score is not None else "N/A"
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
Positive delta below means improvement versus the prior successful node in this branch.

Environment:
{env_summary.strip()}

{task_profile_summary.strip()}

{_route_blocks(profile)}

Branch history:
{history_table}

Already tried: {used_str}
Untried: {untried_str}
{req_block}

{suspicion_block}

Other branches:
{journal_summary or '(none)'}

Data preview:
{data_preview.strip()}

Previous code:
```python
{parent.code}
```

BEFORE writing code, follow this order:
1. REFLECT on why the current solution is underperforming or overfitting for this specific route.
2. PLAN one concrete change that is route-correct and metric-correct.
3. CODE the full improved solution.py.

State reflection and plan in 2-4 sentences, then output the full solution.py, then STRATEGIES: tag1, tag2, ...
Keep FINAL VAL SCORE / FINAL HOLDOUT SCORE / METRIC DIRECTION prints. If CV is used, print CV FOLD SCORES too.
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
