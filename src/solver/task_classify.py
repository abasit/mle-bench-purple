"""Task profiling from competition description + local data layout.

The solver needs more than a single coarse label. We derive a compact
``TaskProfile`` that captures:

    - broad category: tabular | vision | nlp | timeseries | audio
    - objective hint: binary/multiclass/regression/segmentation/etc.
    - prediction mode: labels vs probabilities vs regression values
    - validation mode: iid vs stratified vs grouped vs time ordered
    - likely leakage axes: time columns / group columns
    - metric hint: name + maximize/minimize direction when detectable
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

logger = logging.getLogger("solver")

_VISION_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif", ".webp", ".dcm"}
_AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}
_TABULAR_EXTS = {".csv", ".tsv", ".parquet", ".jsonl"}

_VISION_KEYWORDS = {
    "image",
    "images",
    "photograph",
    "photographs",
    "computer vision",
    "pixel",
    "pixels",
    "dicom",
    "x-ray",
    "mri",
    "ct scan",
    "segmentation",
    "object detection",
    "bounding box",
    "mask",
}
_NLP_KEYWORDS = {
    "tweet",
    "tweets",
    "review",
    "essay",
    "essays",
    "text",
    "sentence",
    "paragraph",
    "question answering",
    "question-answering",
    "sentiment",
    "summarization",
    "translation",
    "token",
    "tokenization",
    "bert",
    "deberta",
    "roberta",
}
_TIMESERIES_KEYWORDS = {
    "time series",
    "time-series",
    "forecast",
    "forecasting",
    "seasonality",
    "lag feature",
    "lag features",
    "rolling window",
    "temporal",
    "timestamp",
    "chronological",
}
_AUDIO_KEYWORDS = {
    "audio",
    "audio recording",
    "speech",
    "speaker",
    "waveform",
    "spectrogram",
    "acoustic",
    "phoneme",
    "birdsong",
}
_GROUP_KEYWORDS = {
    "patient",
    "user",
    "group",
    "session",
    "speaker",
    "video",
    "clip",
    "series",
    "household",
    "customer",
    "store",
    "shop",
    "source",
}

_TEXT_COL_NAME_HINTS = {
    "text",
    "comment_text",
    "comment",
    "review",
    "reviews",
    "sentence",
    "content",
    "question",
    "context",
    "prompt",
    "essay",
    "excerpt",
    "anchor",
    "target",
}
_TIME_COL_NAME_HINTS = {
    "date",
    "time",
    "timestamp",
    "datetime",
    "week",
    "month",
    "year",
    "period",
}
_GROUP_COL_NAME_HINTS = {
    "patient",
    "user",
    "group",
    "session",
    "speaker",
    "author",
    "store",
    "shop",
    "source",
    "household",
    "customer",
    "series",
    "video",
    "clip",
    "fold",
}
_IMAGE_PATH_HINTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".dcm"}
_AUDIO_PATH_HINTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}

_METRIC_HINTS: list[tuple[re.Pattern[str], str, bool]] = [
    (re.compile(r"\b(auc|roc[- ]?auc|auroc)\b", re.I), "auc", True),
    (re.compile(r"\b(accuracy|acc|balanced accuracy)\b", re.I), "accuracy", True),
    (re.compile(r"\b(f1|macro f1|micro f1)\b", re.I), "f1", True),
    (re.compile(r"\b(precision|recall)\b", re.I), "classification_metric", True),
    (re.compile(r"\b(jaccard|iou|dice)\b", re.I), "overlap_metric", True),
    (re.compile(r"\b(map|mAP|mean average precision)\b", re.I), "map", True),
    (re.compile(r"\b(ndcg|mean reciprocal rank|mrr|recall@|precision@)\b", re.I), "ranking_metric", True),
    (re.compile(r"\b(pearson|spearman|correlation)\b", re.I), "correlation", True),
    (re.compile(r"\b(log ?loss|cross[- ]?entropy|binary crossentropy|multiclass logloss)\b", re.I), "logloss", False),
    (re.compile(r"\b(rmse|mse|mae|rmsle|mape|smape)\b", re.I), "error_metric", False),
]


@dataclass(frozen=True)
class TaskProfile:
    category: str = "tabular"
    objective: str = "unknown"
    metric_name: str = ""
    maximize: bool | None = None
    prediction_mode: str = "single_target"
    validation_mode: str = "iid"
    route_id: str = "tabular_unknown"
    confidence: float = 0.0
    reasons: tuple[str, ...] = field(default_factory=tuple)
    auxiliary_modalities: tuple[str, ...] = field(default_factory=tuple)
    likely_group_cols: tuple[str, ...] = field(default_factory=tuple)
    likely_time_cols: tuple[str, ...] = field(default_factory=tuple)
    sample_target_cols: tuple[str, ...] = field(default_factory=tuple)
    primary_target: str = ""
    n_classes: int | None = None


@dataclass
class _TargetStat:
    name: str
    unique: int | None = None
    is_numeric: bool = False
    is_float: bool = False
    is_binaryish: bool = False


@dataclass
class _SchemaSignals:
    has_tabular: bool = False
    text_rich_cols: int = 0
    text_named_cols: int = 0
    image_path_cols: int = 0
    audio_path_cols: int = 0
    sample_target_cols: list[str] = field(default_factory=list)
    target_candidates: list[str] = field(default_factory=list)
    target_stats: list[_TargetStat] = field(default_factory=list)
    target_name: str | None = None
    target_unique: int | None = None
    target_is_float: bool = False
    target_is_numeric: bool = False
    likely_time_cols: list[str] = field(default_factory=list)
    likely_group_cols: list[str] = field(default_factory=list)


def _count_keyword_hits(desc_lower: str, keywords: set[str]) -> int:
    return sum(1 for keyword in keywords if keyword in desc_lower)


def classify_task(description: str, data_dir: Path) -> str:
    """Backward-compatible wrapper returning the broad category only."""
    return profile_task(description, data_dir).category


def profile_task(description: str, data_dir: Path) -> TaskProfile:
    desc_lower = (description or "").lower()
    ext_counts, top_dirs = _scan_files(data_dir)
    schema = _inspect_schema(data_dir)

    n_total = max(sum(ext_counts.values()), 1)
    n_images = sum(ext_counts.get(ext, 0) for ext in _VISION_EXTS)
    n_audio = sum(ext_counts.get(ext, 0) for ext in _AUDIO_EXTS)
    img_frac = n_images / n_total
    aud_frac = n_audio / n_total

    scores = {
        "tabular": 0.0,
        "vision": 0.0,
        "nlp": 0.0,
        "timeseries": 0.0,
        "audio": 0.0,
    }
    reasons: dict[str, list[str]] = {key: [] for key in scores}

    if schema.has_tabular:
        scores["tabular"] += 2.0
        reasons["tabular"].append("tabular train/test files detected")

    if n_images >= 20 and img_frac >= 0.10:
        scores["vision"] += 6.0
        reasons["vision"].append(f"{n_images} image-like files detected")
    elif n_images >= 5:
        scores["vision"] += 2.5
        reasons["vision"].append("some image files detected")

    if schema.image_path_cols:
        scores["vision"] += 3.0
        reasons["vision"].append("tabular metadata references image files")

    if n_audio >= 10 and aud_frac >= 0.08:
        scores["audio"] += 6.0
        reasons["audio"].append(f"{n_audio} audio-like files detected")
    elif n_audio >= 3:
        scores["audio"] += 2.5
        reasons["audio"].append("some audio files detected")

    if schema.audio_path_cols:
        scores["audio"] += 3.0
        reasons["audio"].append("tabular metadata references audio files")

    nlp_hits = _count_keyword_hits(desc_lower, _NLP_KEYWORDS)
    if schema.text_rich_cols:
        scores["nlp"] += 4.0 + min(2.0, schema.text_rich_cols)
        reasons["nlp"].append("long free-text columns detected")
    if schema.text_named_cols:
        scores["nlp"] += 1.5
        reasons["nlp"].append("text-like column names detected")
    if nlp_hits:
        scores["nlp"] += min(4.0, 1.5 * nlp_hits)
        reasons["nlp"].append(f"description has {nlp_hits} NLP cues")

    ts_hits = _count_keyword_hits(desc_lower, _TIMESERIES_KEYWORDS)
    if schema.likely_time_cols:
        scores["timeseries"] += 3.5
        reasons["timeseries"].append("datetime columns detected")
    if ts_hits:
        scores["timeseries"] += min(4.0, 1.5 * ts_hits)
        reasons["timeseries"].append(f"description has {ts_hits} time-series cues")
    if any(name in {"train_images", "test_images", "train_audio", "test_audio"} for name in top_dirs):
        scores["tabular"] -= 0.5

    vision_hits = _count_keyword_hits(desc_lower, _VISION_KEYWORDS)
    if vision_hits:
        scores["vision"] += min(4.0, 1.5 * vision_hits)
        reasons["vision"].append(f"description has {vision_hits} vision cues")

    audio_hits = _count_keyword_hits(desc_lower, _AUDIO_KEYWORDS)
    if audio_hits:
        scores["audio"] += min(4.0, 1.5 * audio_hits)
        reasons["audio"].append(f"description has {audio_hits} audio cues")

    category = max(scores, key=scores.get)
    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best_score = ordered[0][1]
    second_score = ordered[1][1] if len(ordered) > 1 else 0.0

    if best_score <= 0:
        category = "tabular"
    elif category != "tabular" and schema.has_tabular and (best_score - second_score) < 1.0:
        category = "tabular"

    metric_name, maximize = _infer_metric_hint(desc_lower)
    objective = _infer_objective(desc_lower, category, schema)
    prediction_mode = _infer_prediction_mode(objective, metric_name, schema)
    validation_mode = _infer_validation_mode(category, objective, schema, desc_lower)
    route_id = _derive_route_id(category, objective, prediction_mode, metric_name)
    modalities = _auxiliary_modalities(category, schema)
    category_reasons = tuple(reasons.get(category) or ("defaulted to tabular",))
    confidence = 0.0 if best_score <= 0 else min(0.99, 0.5 + max(0.0, best_score - second_score) / 8.0)
    n_classes = schema.target_unique if objective in {"binary_classification", "multiclass_classification"} else None

    return TaskProfile(
        category=category,
        objective=objective,
        metric_name=metric_name,
        maximize=maximize,
        prediction_mode=prediction_mode,
        validation_mode=validation_mode,
        route_id=route_id,
        confidence=confidence,
        reasons=category_reasons,
        auxiliary_modalities=modalities,
        likely_group_cols=tuple(schema.likely_group_cols[:4]),
        likely_time_cols=tuple(schema.likely_time_cols[:4]),
        sample_target_cols=tuple(schema.sample_target_cols[:8]),
        primary_target=schema.target_name or "",
        n_classes=n_classes,
    )


def render_task_profile(profile: TaskProfile) -> str:
    """Compact prompt-ready summary of the inferred task profile."""
    metric = profile.metric_name or "unknown"
    direction = (
        "maximize"
        if profile.maximize is True
        else "minimize"
        if profile.maximize is False
        else "unknown"
    )
    aux = ", ".join(profile.auxiliary_modalities) if profile.auxiliary_modalities else "(none)"
    reasons = "; ".join(profile.reasons[:4]) if profile.reasons else "fallback heuristics"
    groups = ", ".join(profile.likely_group_cols) if profile.likely_group_cols else "(none)"
    times = ", ".join(profile.likely_time_cols) if profile.likely_time_cols else "(none)"
    sample_targets = ", ".join(profile.sample_target_cols) if profile.sample_target_cols else "(unknown)"
    target = profile.primary_target or "(unknown)"
    n_classes = str(profile.n_classes) if profile.n_classes is not None else "unknown"
    return (
        "Task profile:\n"
        f"- broad_category: {profile.category}\n"
        f"- objective_hint: {profile.objective}\n"
        f"- metric_hint: {metric} ({direction})\n"
        f"- prediction_mode: {profile.prediction_mode}\n"
        f"- validation_mode: {profile.validation_mode}\n"
        f"- route_id: {profile.route_id}\n"
        f"- primary_target: {target}\n"
        f"- sample_target_cols: {sample_targets}\n"
        f"- likely_time_cols: {times}\n"
        f"- likely_group_cols: {groups}\n"
        f"- n_classes: {n_classes}\n"
        f"- auxiliary_modalities: {aux}\n"
        f"- confidence: {profile.confidence:.2f}\n"
        f"- evidence: {reasons}"
    )


def _scan_files(data_dir: Path) -> tuple[dict[str, int], set[str]]:
    ext_counts: dict[str, int] = {}
    top_dirs: set[str] = set()
    if not data_dir.exists():
        return ext_counts, top_dirs

    total = 0
    for path in data_dir.rglob("*"):
        if path.is_dir():
            try:
                rel = path.relative_to(data_dir).parts
                if rel:
                    top_dirs.add(rel[0].lower())
            except Exception:
                pass
            continue
        ext = path.suffix.lower()
        ext_counts[ext] = ext_counts.get(ext, 0) + 1
        total += 1
        if total >= 10000:
            break
    return ext_counts, top_dirs


def _inspect_schema(data_dir: Path) -> _SchemaSignals:
    signals = _SchemaSignals()
    if not data_dir.exists():
        return signals

    try:
        import pandas as pd
    except Exception:
        return signals

    files = _select_tabular_files(data_dir)
    if not files:
        return signals
    signals.has_tabular = True

    sample_df = None
    train_df = None
    test_df = None

    for path in files[:6]:
        name = path.name.lower()
        try:
            df = _read_tabular(path, pd, nrows=400)
        except Exception:
            continue

        _update_schema_cues(df, signals)

        if sample_df is None and "sample" in name and "submission" in name:
            sample_df = df
        elif train_df is None and "train" in name and "label" not in name:
            train_df = df
        elif test_df is None and "test" in name:
            test_df = df

    sample_cols = list(sample_df.columns) if sample_df is not None else []
    signals.sample_target_cols = [col for col in sample_cols if not _looks_like_id(col)]

    if train_df is None:
        return signals

    train_cols = list(train_df.columns)
    test_cols = set(test_df.columns) if test_df is not None else set()

    target_candidates = [
        col
        for col in signals.sample_target_cols
        if col in train_cols and col not in test_cols and not _looks_like_id(col)
    ]
    if not target_candidates:
        train_only = [
            col for col in train_cols
            if col not in test_cols and not _looks_like_id(col)
        ]
        ranked = sorted(train_only, key=_score_target_name, reverse=True)
        if signals.sample_target_cols and len(signals.sample_target_cols) > 1 and len(ranked) == 1:
            target_candidates = ranked[:1]
        elif not signals.sample_target_cols and ranked:
            target_candidates = ranked[:1]
        elif not target_candidates and ranked:
            limit = 1 if len(signals.sample_target_cols) <= 1 else min(len(ranked), len(signals.sample_target_cols))
            target_candidates = ranked[:limit]

    signals.target_candidates = target_candidates
    signals.target_stats = _summarize_targets(train_df, target_candidates, pd)
    if signals.target_stats:
        primary = signals.target_stats[0]
        signals.target_name = primary.name
        signals.target_unique = primary.unique
        signals.target_is_numeric = primary.is_numeric
        signals.target_is_float = primary.is_float
    return signals


def _select_tabular_files(data_dir: Path) -> list[Path]:
    found: list[Path] = []
    for path in data_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in _TABULAR_EXTS:
            found.append(path)

    def rank(path: Path) -> tuple[int, int]:
        name = path.name.lower()
        if "train" in name and "label" not in name:
            return (0, len(name))
        if "test" in name:
            return (1, len(name))
        if "sample" in name and "submission" in name:
            return (2, len(name))
        return (3, len(name))

    return sorted(found, key=rank)


def _read_tabular(path: Path, pd, nrows: int | None = None):
    ext = path.suffix.lower()
    if ext == ".parquet":
        df = pd.read_parquet(path)
        return df.head(nrows) if nrows is not None else df
    if ext == ".tsv":
        return pd.read_csv(path, sep="\t", nrows=nrows)
    if ext == ".jsonl":
        return pd.read_json(path, lines=True, nrows=nrows)
    return pd.read_csv(path, nrows=nrows)


def _update_schema_cues(df, signals: _SchemaSignals) -> None:
    try:
        import pandas as pd
    except Exception:
        return

    for col in df.columns:
        name = str(col).lower()
        series = df[col]
        values = series.dropna().astype(str).head(80).tolist()
        if not values:
            continue

        if any(hint in name for hint in _TEXT_COL_NAME_HINTS):
            signals.text_named_cols += 1

        avg_len = sum(len(value) for value in values) / max(len(values), 1)
        avg_words = sum(len(value.split()) for value in values) / max(len(values), 1)
        if avg_len >= 40 or avg_words >= 6:
            signals.text_rich_cols += 1

        if any(hint in name for hint in _GROUP_COL_NAME_HINTS):
            _append_unique(signals.likely_group_cols, str(col))

        if any(hint in name for hint in _TIME_COL_NAME_HINTS):
            _append_unique(signals.likely_time_cols, str(col))
            continue

        if _looks_like_path_series(values, _IMAGE_PATH_HINTS):
            signals.image_path_cols += 1
        if _looks_like_path_series(values, _AUDIO_PATH_HINTS):
            signals.audio_path_cols += 1

        if not pd.api.types.is_numeric_dtype(series) and any(any(ch in value for ch in "-/:") for value in values[:10]):
            try:
                parsed = pd.to_datetime(series.dropna().head(40), errors="coerce")
                if len(parsed) >= 5 and float(parsed.notna().mean()) >= 0.8:
                    _append_unique(signals.likely_time_cols, str(col))
            except Exception:
                continue


def _append_unique(items: list[str], value: str) -> None:
    if value not in items:
        items.append(value)


def _looks_like_path_series(values: Iterable[str], suffixes: set[str]) -> bool:
    vals = [value.strip().lower() for value in values if isinstance(value, str)]
    if len(vals) < 3:
        return False
    hits = sum(any(value.endswith(suffix) for suffix in suffixes) for value in vals[:20])
    return hits >= max(2, min(5, len(vals[:20]) // 2))


def _summarize_targets(train_df, target_candidates: list[str], pd) -> list[_TargetStat]:
    out: list[_TargetStat] = []
    for col in target_candidates[:8]:
        try:
            series = train_df[col]
        except Exception:
            continue
        try:
            unique = int(series.nunique(dropna=True))
        except Exception:
            unique = None
        is_numeric = bool(pd.api.types.is_numeric_dtype(series))
        is_float = bool(pd.api.types.is_float_dtype(series))
        is_binaryish = False
        try:
            values = series.dropna()
            if not values.empty:
                if is_numeric:
                    rounded = values.astype(float)
                    is_binaryish = bool(set(rounded.unique().tolist()) <= {0, 1, 0.0, 1.0})
                elif unique is not None:
                    is_binaryish = unique <= 2
        except Exception:
            pass
        out.append(
            _TargetStat(
                name=col,
                unique=unique,
                is_numeric=is_numeric,
                is_float=is_float,
                is_binaryish=is_binaryish,
            )
        )
    return out


def _score_target_name(name: str) -> int:
    lower = name.lower()
    score = 0
    if lower in {"target", "label", "labels", "class", "classes", "y"}:
        score += 10
    if "target" in lower:
        score += 6
    if "label" in lower or "class" in lower or "category" in lower:
        score += 5
    if "score" in lower or "rating" in lower or "sentiment" in lower:
        score += 3
    if _looks_like_id(name):
        score -= 8
    return score


def _infer_objective(desc_lower: str, category: str, signals: _SchemaSignals) -> str:
    if "recommend" in desc_lower or "retrieval" in desc_lower or "ranking" in desc_lower:
        return "recommendation"
    if "question answering" in desc_lower or "start_position" in desc_lower or "end_position" in desc_lower:
        return "qa"
    if "selected text" in desc_lower or "extract the part" in desc_lower:
        return "span_extraction"
    if "segmentation" in desc_lower or "mask" in desc_lower or "encodedpixels" in desc_lower:
        return "segmentation"
    if "object detection" in desc_lower or "bounding box" in desc_lower or "bbox" in desc_lower:
        return "object_detection"
    if category == "timeseries":
        return "forecasting"
    if "translation" in desc_lower or "normalization" in desc_lower or "sequence to sequence" in desc_lower:
        return "seq2seq"

    sample_targets = signals.sample_target_cols
    target_stats = signals.target_stats
    primary = target_stats[0] if target_stats else None

    if len(sample_targets) > 1:
        if len(target_stats) == 1 and primary is not None and primary.unique is not None and primary.unique > 2:
            if primary.unique <= max(len(sample_targets) + 3, len(sample_targets) * 2):
                return "multiclass_classification"
        if target_stats and all(stat.is_binaryish for stat in target_stats):
            return "multilabel_classification"
        if target_stats and all(stat.is_numeric for stat in target_stats):
            if any(stat.is_float for stat in target_stats) or any((stat.unique or 0) > 20 for stat in target_stats):
                return "multioutput_regression"

    if primary is not None:
        if primary.is_numeric and (primary.is_float or (primary.unique or 0) > 20):
            return "regression"
        if primary.unique is not None:
            if primary.unique <= 2:
                return "binary_classification"
            if primary.unique <= 50:
                return "multiclass_classification"

    if category in {"vision", "audio", "nlp"}:
        return "classification"
    return "unknown"


def _infer_metric_hint(desc_lower: str) -> tuple[str, bool | None]:
    for pattern, name, maximize in _METRIC_HINTS:
        if pattern.search(desc_lower):
            return name, maximize
    return "", None


def _infer_prediction_mode(objective: str, metric_name: str, signals: _SchemaSignals) -> str:
    sample_target_count = len(signals.sample_target_cols)
    if objective == "segmentation":
        return "segmentation_encoding"
    if objective == "object_detection":
        return "detection_rows"
    if objective in {"qa", "span_extraction"}:
        return "span_text"
    if objective == "seq2seq":
        return "sequence_text"
    if objective == "recommendation" or metric_name in {"ranking_metric", "map"}:
        return "ranking_scores"
    if objective in {"regression", "forecasting"}:
        return "single_continuous"
    if objective == "multioutput_regression":
        return "multioutput_regression"
    if objective == "multilabel_classification":
        return "multilabel_probabilities"
    if objective == "multiclass_classification":
        return "multiclass_probabilities" if sample_target_count > 1 else "single_label"
    if objective == "binary_classification":
        return "single_probability" if metric_name in {"logloss", "auc"} else "single_label"
    if objective == "classification":
        if sample_target_count > 1:
            return "multiclass_probabilities"
        return "single_probability" if metric_name in {"logloss", "auc"} else "single_label"
    return "single_target"


def _infer_validation_mode(
    category: str,
    objective: str,
    signals: _SchemaSignals,
    desc_lower: str,
) -> str:
    has_time_axis = category == "timeseries" or bool(signals.likely_time_cols)
    has_group_axis = bool(signals.likely_group_cols) or any(keyword in desc_lower for keyword in _GROUP_KEYWORDS)
    is_classification = "classification" in objective or objective == "classification"

    if has_time_axis:
        return "time_ordered"
    if has_group_axis and is_classification:
        return "stratified_group"
    if has_group_axis:
        return "grouped"
    if is_classification:
        return "stratified"
    return "iid"


def _derive_route_id(category: str, objective: str, prediction_mode: str, metric_name: str) -> str:
    if objective == "recommendation" or metric_name in {"ranking_metric", "map"}:
        return f"{category}_ranking"
    if objective in {"segmentation", "object_detection", "qa", "span_extraction", "seq2seq"}:
        return f"{category}_{objective}"
    if objective == "multiclass_classification" and prediction_mode == "multiclass_probabilities":
        return f"{category}_multiclass_prob"
    if objective == "multilabel_classification":
        return f"{category}_multilabel"
    if objective == "binary_classification" and prediction_mode == "single_probability":
        return f"{category}_binary_prob"
    if objective == "binary_classification":
        return f"{category}_binary_label"
    if objective in {"regression", "forecasting", "multioutput_regression"}:
        return f"{category}_{objective}"
    if objective == "classification":
        return f"{category}_classification"
    return f"{category}_{objective}"


def _auxiliary_modalities(category: str, signals: _SchemaSignals) -> tuple[str, ...]:
    aux: list[str] = []
    if category != "vision" and signals.image_path_cols:
        aux.append("vision")
    if category != "audio" and signals.audio_path_cols:
        aux.append("audio")
    if category != "nlp" and signals.text_rich_cols:
        aux.append("nlp")
    return tuple(aux)


def _looks_like_id(name: str) -> bool:
    lower = name.lower()
    return (
        lower in {"id", "index"}
        or lower.endswith("_id")
        or lower.endswith("id")
        or lower in {"row_id", "image_id", "sample_id"}
    )


def load_kb_card(task_type: str, kb_dir: Path) -> str:
    """Load the markdown KB card for the given task type, if available."""
    if not kb_dir.exists():
        return ""
    candidate = kb_dir / f"{task_type}.md"
    if candidate.exists():
        try:
            return candidate.read_text(encoding="utf-8")
        except Exception:
            return ""
    fallback = kb_dir / "tabular.md"
    if fallback.exists():
        try:
            return fallback.read_text(encoding="utf-8")
        except Exception:
            return ""
    return ""
