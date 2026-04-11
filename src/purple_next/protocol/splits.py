"""Runner-owned dev/holdout split and CV fold assignment.

The runner writes ``_splits.csv`` and ``_protocol.json`` into the data dir
before any solver code runs. Every generated script must read those files
and follow them. That moves the validation protocol out of LLM negotiation
and into ground truth.

``_splits.csv`` columns:
    row_index : integer row index into the labeled training file
    split     : "dev" or "holdout"
    fold      : fold id (0..n_folds-1 for dev rows, -1 for holdout rows)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .contract import TaskContract

logger = logging.getLogger("purple_next")

SPLIT_CSV = "_splits.csv"
PROTOCOL_JSON = "_protocol.json"


@dataclass
class SplitArtifact:
    split_csv: Path
    protocol_json: Path
    n_rows: int
    n_dev: int
    n_holdout: int
    n_folds: int


def prepare_splits(
    data_dir: Path,
    contract: TaskContract,
    *,
    train_filename: str = "train.csv",
) -> SplitArtifact | None:
    """Write the split protocol for a labeled training file.

    Returns None when the training file can't be located or has no target
    column — the runner will fall back to asking the LLM to produce its own
    splits, though the audits will no longer apply.
    """
    train_path = data_dir / train_filename
    if not train_path.exists():
        alt = _guess_train_file(data_dir)
        if alt is None:
            logger.warning(f"[splits] no training file found under {data_dir}")
            return None
        train_path = alt

    try:
        df = pd.read_csv(train_path)
    except Exception as e:
        logger.warning(f"[splits] failed to read {train_path}: {e}")
        return None

    target = contract.target_col if contract.target_col and contract.target_col in df.columns else None
    rng = np.random.default_rng(contract.seed)

    n = len(df)
    if n == 0:
        return None

    indices = np.arange(n)
    rng.shuffle(indices)

    holdout_size = max(1, int(round(n * contract.holdout_fraction)))
    holdout_idx = np.sort(indices[:holdout_size])
    dev_idx = indices[holdout_size:]

    if target is not None and contract.maximize and _is_classification_target(df[target]):
        fold_assign = _stratified_folds(df[target].iloc[dev_idx].to_numpy(), contract.n_folds, rng)
    else:
        fold_assign = np.arange(len(dev_idx), dtype=np.int32) % contract.n_folds
        rng.shuffle(fold_assign)

    split_col = np.full(n, "dev", dtype=object)
    split_col[holdout_idx] = "holdout"
    fold_col = np.full(n, -1, dtype=np.int32)
    fold_col[dev_idx] = fold_assign

    out = pd.DataFrame(
        {"row_index": np.arange(n, dtype=np.int32), "split": split_col, "fold": fold_col}
    )
    split_csv = data_dir / SPLIT_CSV
    out.to_csv(split_csv, index=False)

    protocol = contract.to_dict()
    protocol["train_filename"] = train_path.name
    protocol["split_csv"] = SPLIT_CSV
    protocol_json = data_dir / PROTOCOL_JSON
    protocol_json.write_text(json.dumps(protocol, indent=2), encoding="utf-8")

    logger.info(
        f"[splits] wrote {SPLIT_CSV}: n={n} dev={len(dev_idx)} holdout={holdout_size} "
        f"folds={contract.n_folds} target={target!r}"
    )

    return SplitArtifact(
        split_csv=split_csv,
        protocol_json=protocol_json,
        n_rows=n,
        n_dev=int(len(dev_idx)),
        n_holdout=int(holdout_size),
        n_folds=contract.n_folds,
    )


def _guess_train_file(data_dir: Path) -> Path | None:
    for name in ("train.csv", "training.csv", "train.parquet"):
        p = data_dir / name
        if p.exists():
            return p
    for p in sorted(data_dir.glob("*.csv")):
        if "sample" in p.name.lower() or "test" in p.name.lower():
            continue
        return p
    return None


def _is_classification_target(series: pd.Series) -> bool:
    if series.dtype == object or str(series.dtype).startswith("category"):
        return True
    unique = series.dropna().unique()
    return len(unique) <= max(20, int(round(len(series) ** 0.25)))


def _stratified_folds(labels: np.ndarray, n_folds: int, rng: np.random.Generator) -> np.ndarray:
    folds = np.full(len(labels), -1, dtype=np.int32)
    for value in np.unique(labels):
        positions = np.where(labels == value)[0]
        rng.shuffle(positions)
        for k, pos in enumerate(positions):
            folds[pos] = k % n_folds
    return folds
