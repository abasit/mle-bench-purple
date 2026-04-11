"""Task contract: what the runner knows about the competition before any code runs.

The LLM does not need to parse the metric direction from stdout — the runner
reads the description and sample_submission, infers direction and target
column, and writes it to ``_protocol.json`` alongside the split file.
"""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path

logger = logging.getLogger("purple_next")


@dataclass
class TaskContract:
    metric: str = "unknown"          # "auc", "rmse", "logloss", ...
    maximize: bool = True
    target_col: str = ""
    id_col: str = "id"
    category: str = "tabular"        # tabular | nlp | vision | timeseries | audio
    n_folds: int = 5
    holdout_fraction: float = 0.20
    seed: int = 42

    def to_dict(self) -> dict:
        return asdict(self)


# Keyword hints scanned from description.md for category + metric direction.
_MAXIMIZE_METRICS = {"auc", "roc_auc", "accuracy", "f1", "map", "ndcg", "recall", "precision", "correlation"}
_MINIMIZE_METRICS = {"rmse", "rmsle", "mse", "mae", "logloss", "log_loss", "error"}


def infer_contract(
    data_dir: Path,
    *,
    n_folds: int,
    holdout_fraction: float,
    seed: int,
) -> TaskContract:
    """Look at description.md and sample_submission.csv to build a task contract.

    Falls back to reasonable defaults when signal is missing.
    """
    contract = TaskContract(n_folds=n_folds, holdout_fraction=holdout_fraction, seed=seed)

    desc_text = ""
    for name in ("description.md", "description.txt", "README.md"):
        path = data_dir / name
        if path.exists():
            try:
                desc_text = path.read_text(encoding="utf-8", errors="replace")
                break
            except Exception:
                pass

    if desc_text:
        lowered = desc_text.lower()
        for metric in _MAXIMIZE_METRICS:
            if re.search(rf"\b{re.escape(metric)}\b", lowered):
                contract.metric = metric
                contract.maximize = True
                break
        else:
            for metric in _MINIMIZE_METRICS:
                if re.search(rf"\b{re.escape(metric)}\b", lowered):
                    contract.metric = metric
                    contract.maximize = False
                    break

        if re.search(r"\b(image|images|pixel|vision)\b", lowered):
            contract.category = "vision"
        elif re.search(r"\b(text|sentence|tokens|nlp)\b", lowered):
            contract.category = "nlp"
        elif re.search(r"\b(time[- ]series|temporal|date|forecast)\b", lowered):
            contract.category = "timeseries"
        elif re.search(r"\b(audio|spectrogram|waveform)\b", lowered):
            contract.category = "audio"

    sample = data_dir / "sample_submission.csv"
    if sample.exists():
        try:
            first_line = sample.read_text(encoding="utf-8", errors="replace").splitlines()[0]
            cols = [c.strip() for c in first_line.split(",") if c.strip()]
            if cols:
                lowered = [c.lower() for c in cols]
                id_candidates = [c for c, low in zip(cols, lowered) if low == "id" or low.endswith("_id")]
                if id_candidates:
                    contract.id_col = id_candidates[0]
                target_candidates = [c for c in cols if c not in id_candidates]
                if target_candidates:
                    contract.target_col = target_candidates[0]
        except Exception as e:
            logger.debug(f"[contract] sample_submission parse failed: {e}")

    return contract
