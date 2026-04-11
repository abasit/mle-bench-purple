"""Fake-success detector.

Two cheap checks that catch the common "script crashed and printed fallback
scores" pattern:

1. Every non-id prediction column is strictly constant (nunique <= 1).
2. The submission is byte-identical to sample_submission.csv.

Near-constant predictions are NOT flagged — on imbalanced classification
they are legitimate. Gap-threshold audits live in the reviewer agent, not here.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger("purple_next")


def detect_fake_success(submission_path: Path, sample_submission_path: Path | None) -> str | None:
    if submission_path is None or not submission_path.exists():
        return "no submission path"

    try:
        import pandas as pd
        df = pd.read_csv(submission_path)
    except Exception as e:
        logger.debug(f"[fake-success] submission read failed: {e}")
        return None

    if df.empty or len(df) < 2:
        return None

    pred_cols = [
        c for c in df.columns
        if not (c.lower().endswith("id") or c.lower() in {"id", "index"})
    ]
    if pred_cols and all(df[c].nunique(dropna=False) <= 1 for c in pred_cols):
        return f"submission has constant prediction column(s) {pred_cols[:3]}"

    if sample_submission_path is not None and sample_submission_path.exists():
        try:
            if submission_path.read_bytes() == sample_submission_path.read_bytes():
                return f"submission is byte-identical to {sample_submission_path.name}"
        except Exception as e:
            logger.debug(f"[fake-success] byte compare failed: {e}")

    return None
