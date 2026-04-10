"""Submission post-processor: deterministic column patching.

After all the LLM work is done, we open ``sample_submission.csv`` and force
the candidate ``submission.csv`` to match its shape — backfilling any
missing columns from ``test.csv`` (preferred) or the sample.

This salvages a meaningful fraction of runs where the LLM forgot a single
metadata column or emitted columns in the wrong order.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger("solver")


def patch_submission_columns(submission_csv: bytes, data_dir: Path) -> bytes:
    """Return a column-aligned submission CSV.

    If anything goes wrong, return the input bytes unchanged — we'd rather
    submit a slightly imperfect file than nothing at all.
    """
    if not submission_csv:
        return submission_csv

    sample_path = _find_sample_submission(data_dir)
    if sample_path is None:
        logger.info("[postprocess] no sample_submission file found, skipping")
        return submission_csv

    try:
        sample_df = pd.read_csv(sample_path)
    except Exception as e:
        logger.warning(f"[postprocess] could not read sample submission: {e}")
        return submission_csv

    try:
        from io import BytesIO
        sub_df = pd.read_csv(BytesIO(submission_csv))
    except Exception as e:
        logger.warning(f"[postprocess] could not parse candidate submission: {e}")
        return submission_csv

    sample_cols = list(sample_df.columns)
    sub_cols = list(sub_df.columns)
    if sample_cols == sub_cols and len(sub_df) == len(sample_df):
        return submission_csv

    # Backfill missing columns.
    missing = [c for c in sample_cols if c not in sub_cols]
    extra = [c for c in sub_cols if c not in sample_cols]
    if missing or extra:
        logger.info(
            f"[postprocess] patching columns: missing={missing}, extra={extra}"
        )

    if missing:
        test_df = _read_test(data_dir)
        for col in missing:
            if test_df is not None and col in test_df.columns and len(test_df) == len(sub_df):
                sub_df[col] = test_df[col].values
            elif col in sample_df.columns and len(sample_df) == len(sub_df):
                sub_df[col] = sample_df[col].values
            else:
                # Last resort: a constant placeholder.
                if pd.api.types.is_numeric_dtype(sample_df[col]):
                    sub_df[col] = 0
                else:
                    sub_df[col] = ""

    # Drop extras and reorder.
    try:
        sub_df = sub_df[sample_cols]
    except KeyError:
        # If we still can't align, give up gracefully.
        logger.warning("[postprocess] could not align to sample columns")
        return submission_csv

    if len(sub_df) != len(sample_df):
        logger.warning(
            f"[postprocess] row count mismatch: sub={len(sub_df)}, sample={len(sample_df)} — keeping sub rows"
        )

    return sub_df.to_csv(index=False).encode()


def _find_sample_submission(data_dir: Path) -> Path | None:
    if not data_dir.exists():
        return None
    candidates: list[Path] = []
    for p in data_dir.rglob("*.csv"):
        name = p.name.lower()
        if "sample" in name and "submission" in name:
            candidates.append(p)
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: (len(p.parts), len(p.name)))[0]


def _read_test(data_dir: Path) -> pd.DataFrame | None:
    """Try several common names for the test split."""
    for name in ("test.csv", "test.parquet", "test_features.csv", "test_X.csv"):
        for p in data_dir.rglob(name):
            try:
                if name.endswith(".parquet"):
                    return pd.read_parquet(p)
                return pd.read_csv(p)
            except Exception:
                continue
    return None
