"""Blend the top-K submissions into a single CSV."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("solver")


def blend_submissions(
    submission_paths: list[Path],
    weights: list[float] | None = None,
    *,
    objective_hint: str = "",
    metric_hint: str = "",
    prediction_mode: str = "",
    holdout_scores: list[float | None] | None = None,
    maximize: bool = True,
) -> bytes | None:
    """Read and blend a list of submission CSVs. Returns blended bytes or None."""
    dfs: list[pd.DataFrame] = []
    used_weights: list[float] = []
    ref_cols: list[str] | None = None

    if weights is None and holdout_scores is not None:
        weights = _weights_from_holdout(holdout_scores, maximize)
    if weights is None:
        weights = [1.0] * len(submission_paths)
    if len(weights) != len(submission_paths):
        weights = [1.0] * len(submission_paths)

    for path, weight in zip(submission_paths, weights):
        if not path or not path.exists() or weight <= 0:
            continue
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            logger.warning(f"[blend] failed to read {path}: {exc}")
            continue
        if ref_cols is None:
            ref_cols = list(df.columns)
        elif list(df.columns) != ref_cols:
            logger.info(f"[blend] skipping {path.name}: column mismatch")
            continue
        dfs.append(df)
        used_weights.append(float(weight))

    if len(dfs) < 2 or ref_cols is None:
        logger.info(f"[blend] fewer than 2 compatible submissions ({len(dfs)})")
        return None

    total_w = sum(used_weights)
    norm_w = [weight / total_w for weight in used_weights] if total_w > 0 else [1.0 / len(dfs)] * len(dfs)

    ref_df = dfs[0]
    bool_cols = [col for col in ref_cols if _is_binary_col(col, ref_df, dfs)]
    id_cols = [col for col in ref_cols if col not in bool_cols and _is_id_col(col, ref_df, dfs)]
    label_cols = [
        col
        for col in ref_cols
        if col not in bool_cols and col not in id_cols and _is_label_col(col, ref_df, dfs, objective_hint, prediction_mode)
    ]
    numeric_cols = [col for col in ref_cols if col not in bool_cols and col not in id_cols and col not in label_cols]

    logger.info(
        f"[blend] blending {len(dfs)} submissions with weights={[round(w, 3) for w in norm_w]} "
        f"prediction_mode={prediction_mode or '(unknown)'} columns={ref_cols}"
    )

    blended = dfs[0][id_cols].copy()

    for col in bool_cols:
        votes = sum(weight * df[col].astype(float) for weight, df in zip(norm_w, dfs))
        majority = votes >= 0.5
        blended[col] = majority if pd.api.types.is_bool_dtype(ref_df[col]) else majority.astype(int)

    for col in label_cols:
        blended[col] = _weighted_mode([df[col] for df in dfs], norm_w)

    if numeric_cols:
        numeric_values = {
            col: _blend_numeric_column(
                col,
                dfs,
                norm_w,
                objective_hint=objective_hint,
                metric_hint=metric_hint,
                prediction_mode=prediction_mode,
            )
            for col in numeric_cols
        }
        if prediction_mode == "multiclass_probabilities":
            matrix = np.column_stack([numeric_values[col] for col in numeric_cols])
            matrix = np.clip(matrix, 1e-15, 1.0)
            row_sums = matrix.sum(axis=1, keepdims=True)
            row_sums[row_sums <= 0] = 1.0
            matrix = matrix / row_sums
            for idx, col in enumerate(numeric_cols):
                blended[col] = matrix[:, idx]
        else:
            for col, values in numeric_values.items():
                blended[col] = values

    blended = blended[ref_cols]
    return blended.to_csv(index=False).encode()


def _weights_from_holdout(
    holdout_scores: list[float | None],
    maximize: bool,
) -> list[float]:
    valid = [score for score in holdout_scores if score is not None]
    if len(valid) < 2:
        return [1.0] * len(holdout_scores)

    baseline = min(valid) if maximize else max(valid)
    weights: list[float] = []
    for score in holdout_scores:
        if score is None:
            weights.append(0.0)
            continue
        delta = (score - baseline) if maximize else (baseline - score)
        weights.append(max(delta, 1e-6))
    return weights


def _blend_numeric_column(
    col: str,
    dfs: list[pd.DataFrame],
    weights: list[float],
    *,
    objective_hint: str,
    metric_hint: str,
    prediction_mode: str,
) -> np.ndarray:
    values = [df[col].values.astype(float) for df in dfs]
    metric = (metric_hint or "").lower()
    objective = (objective_hint or "").lower()

    if objective in {"regression", "forecasting", "multioutput_regression"}:
        return _arithmetic_blend(values, weights)
    if prediction_mode in {"multiclass_probabilities", "multilabel_probabilities"}:
        return np.clip(_arithmetic_blend(values, weights), 0.0, 1.0)
    if prediction_mode == "single_probability":
        if metric in {"auc", "ranking_metric", "correlation", "map"}:
            return _rank_blend(values, weights)
        return np.clip(_arithmetic_blend(values, weights), 0.0, 1.0)
    if metric in {"ranking_metric", "correlation", "map"}:
        return _rank_blend(values, weights)
    return _arithmetic_blend(values, weights)


def _arithmetic_blend(values: list[np.ndarray], weights: list[float]) -> np.ndarray:
    total_w = sum(weights)
    if total_w <= 0:
        return values[0]
    return sum(weight * value for weight, value in zip(weights, values)) / total_w


def _rank_blend(values: list[np.ndarray], weights: list[float]) -> np.ndarray:
    total_w = sum(weights)
    if total_w <= 0:
        return values[0]
    ranked = [pd.Series(value).rank(pct=True).values for value in values]
    return sum(weight * value for weight, value in zip(weights, ranked)) / total_w


def _is_binary_col(col: str, ref_df: pd.DataFrame, dfs: list[pd.DataFrame]) -> bool:
    if pd.api.types.is_bool_dtype(ref_df[col]):
        return True
    if not pd.api.types.is_numeric_dtype(ref_df[col]):
        return False
    unique_vals: set[object] = set()
    for df in dfs:
        unique_vals.update(df[col].dropna().unique().tolist())
    return bool(unique_vals) and unique_vals <= {0, 1, 0.0, 1.0, True, False}


_ID_NAME_HINTS = ("id", "uuid", "key", "guid", "index", "row_id", "image_id", "sample_id")


def _is_id_col(col: str, ref_df: pd.DataFrame, dfs: list[pd.DataFrame]) -> bool:
    lower = col.lower()
    name_hint = lower in _ID_NAME_HINTS or lower.endswith("_id") or lower.endswith("id")
    if not _column_equal_across_frames(col, ref_df, dfs):
        return False
    return col == ref_df.columns[0] or name_hint


def _is_label_col(
    col: str,
    ref_df: pd.DataFrame,
    dfs: list[pd.DataFrame],
    objective_hint: str,
    prediction_mode: str,
) -> bool:
    objective = (objective_hint or "").lower()
    if prediction_mode in {"multiclass_probabilities", "multilabel_probabilities", "single_probability"}:
        return False
    if not pd.api.types.is_numeric_dtype(ref_df[col]):
        return True
    if "classification" not in objective and prediction_mode != "single_label":
        return False
    try:
        values = pd.concat([df[col] for df in dfs], ignore_index=True).dropna()
        if values.empty:
            return False
        arr = values.astype(float).to_numpy()
        if not np.allclose(arr, np.round(arr), atol=1e-8):
            return False
        return int(values.nunique(dropna=True)) <= 100
    except Exception:
        return False


def _column_equal_across_frames(col: str, ref_df: pd.DataFrame, dfs: list[pd.DataFrame]) -> bool:
    try:
        ref_vals = ref_df[col].reset_index(drop=True)
        for df in dfs[1:]:
            if len(df) != len(ref_vals):
                return False
            if not ref_vals.equals(df[col].reset_index(drop=True)):
                return False
        return True
    except Exception:
        return False


def _weighted_mode(series_list: list[pd.Series], weights: list[float]) -> list[object]:
    if not series_list:
        return []
    rows = len(series_list[0])
    out: list[object] = []
    for row_idx in range(rows):
        scores: dict[object, list[object]] = {}
        for order, (weight, series) in enumerate(zip(weights, series_list)):
            raw = series.iat[row_idx]
            key = "__nan__" if pd.isna(raw) else raw
            entry = scores.get(key)
            if entry is None:
                scores[key] = [float(weight), order, raw]
            else:
                entry[0] = float(entry[0]) + float(weight)
        winner = max(scores.values(), key=lambda item: (float(item[0]), -int(item[1])))
        out.append(winner[2])
    return out
