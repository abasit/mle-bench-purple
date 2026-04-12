"""Blend top-K submission CSVs into a single submission."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("purple_next")


def blend_submissions(
    submission_paths: list[Path],
    holdout_scores: list[float | None] | None = None,
    maximize: bool = True,
) -> bytes | None:
    """Weighted-average blend of submission CSVs. Returns blended CSV bytes or None."""
    weights = _weights_from_holdout(holdout_scores or [], maximize)

    dfs: list[pd.DataFrame] = []
    used_weights: list[float] = []
    ref_cols: list[str] | None = None

    for path, w in zip(submission_paths, weights):
        if not path or not path.exists() or w <= 0:
            continue
        try:
            df = pd.read_csv(path)
        except Exception as e:
            logger.warning(f"[blend] failed to read {path}: {e}")
            continue
        if ref_cols is None:
            ref_cols = list(df.columns)
        elif list(df.columns) != ref_cols:
            logger.info(f"[blend] skipping {path.name}: column mismatch")
            continue
        dfs.append(df)
        used_weights.append(w)

    if len(dfs) < 2 or ref_cols is None:
        return None

    total = sum(used_weights)
    norm = [w / total for w in used_weights] if total > 0 else [1.0 / len(dfs)] * len(dfs)

    logger.info(f"[blend] blending {len(dfs)} submissions, weights={[round(w, 3) for w in norm]}")

    ref = dfs[0]
    blended = pd.DataFrame()

    for col in ref_cols:
        if _is_id_col(col, ref, dfs):
            blended[col] = ref[col]
        elif _is_binary_col(col, dfs):
            votes = sum(w * df[col].astype(float) for w, df in zip(norm, dfs))
            blended[col] = (votes >= 0.5).astype(ref[col].dtype)
        elif pd.api.types.is_numeric_dtype(ref[col]):
            blended[col] = sum(w * df[col].astype(float) for w, df in zip(norm, dfs))
        else:
            # Non-numeric, non-ID: weighted mode
            blended[col] = _weighted_mode([df[col] for df in dfs], norm)

    return blended[ref_cols].to_csv(index=False).encode()


def _weights_from_holdout(scores: list[float | None], maximize: bool) -> list[float]:
    valid = [s for s in scores if s is not None]
    if len(valid) < 2:
        return [1.0] * len(scores)
    baseline = min(valid) if maximize else max(valid)
    return [
        max((s - baseline) if maximize else (baseline - s), 1e-6) if s is not None else 0.0
        for s in scores
    ]


def _is_id_col(col: str, ref: pd.DataFrame, dfs: list[pd.DataFrame]) -> bool:
    low = col.lower()
    if not (low == "id" or low.endswith("_id") or low.endswith("id") or col == ref.columns[0]):
        return False
    # ID columns should be identical across all submissions
    ref_vals = ref[col].reset_index(drop=True)
    return all(ref_vals.equals(df[col].reset_index(drop=True)) for df in dfs[1:])


def _is_binary_col(col: str, dfs: list[pd.DataFrame]) -> bool:
    if pd.api.types.is_bool_dtype(dfs[0][col]):
        return True
    if not pd.api.types.is_numeric_dtype(dfs[0][col]):
        return False
    unique = set()
    for df in dfs:
        unique.update(df[col].dropna().unique().tolist())
    return bool(unique) and unique <= {0, 1, 0.0, 1.0, True, False}


def _weighted_mode(series_list: list[pd.Series], weights: list[float]) -> list:
    rows = len(series_list[0])
    out = []
    for i in range(rows):
        scores: dict = {}
        for w, s in zip(weights, series_list):
            val = s.iat[i]
            key = "__nan__" if pd.isna(val) else val
            if key in scores:
                scores[key][0] += w
            else:
                scores[key] = [w, val]
        winner = max(scores.values(), key=lambda x: x[0])
        out.append(winner[1])
    return out
