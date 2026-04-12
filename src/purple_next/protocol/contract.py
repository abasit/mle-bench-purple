"""Task contract: what the runner knows about the competition before any code runs.

Uses a single LLM call to read the competition description and sample_submission
header, then returns a structured contract with metric, direction, target column,
ID column, and category.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from textwrap import dedent

logger = logging.getLogger("purple_next")


@dataclass
class TaskContract:
    metric: str = "unknown"
    maximize: bool = True
    target_col: str = ""
    id_col: str = "id"
    category: str = "tabular"
    n_folds: int = 5
    holdout_fraction: float = 0.20
    seed: int = 42

    def to_dict(self) -> dict:
        return asdict(self)


_CONTRACT_SYS = dedent("""
    Your task is the following: Given a Kaggle competition description and
    submission format, identify the evaluation setup. Return ONLY a JSON object:
    {
      "metric": "<e.g. accuracy, rmse, auc, f1, logloss, quadratic_weighted_kappa>",
      "maximize": <true if higher is better, false if lower is better>,
      "target_col": "<prediction column(s) in sample_submission, not the ID column>",
      "id_col": "<ID column in sample_submission>",
      "category": "<tabular|vision|nlp|timeseries|audio>"
    }
    Return the JSON and nothing else.
""").strip()


def infer_contract(
    data_dir: Path,
    *,
    llm,
    n_folds: int,
    holdout_fraction: float,
    seed: int,
) -> TaskContract:
    """Use a single LLM call to infer the task contract from competition files."""
    contract = TaskContract(n_folds=n_folds, holdout_fraction=holdout_fraction, seed=seed)

    desc = _read_file(data_dir, "description.md", "description.txt", "README.md")
    sample_header = _read_first_line(data_dir / "sample_submission.csv")
    train_header = _read_first_line(data_dir / "train.csv") or _read_first_line(data_dir / "training.csv")

    if not desc and not sample_header:
        logger.warning("[contract] no description or sample_submission found")
        return contract

    user_msg = f"Competition description:\n{desc}\n\n"
    if sample_header:
        user_msg += f"sample_submission.csv columns: {sample_header}\n\n"
    if train_header:
        user_msg += f"train.csv columns: {train_header}\n\n"
    user_msg += "Return the JSON object described in the system message."

    response = llm.chat(
        [{"role": "system", "content": _CONTRACT_SYS}, {"role": "user", "content": user_msg}],
        label="contract_inference",
    )
    parsed = _parse_json(response)
    if parsed is None:
        logger.warning("[contract] failed to parse LLM response as JSON")
        return contract

    contract.metric = str(parsed.get("metric", "unknown")).strip().lower()
    contract.maximize = bool(parsed.get("maximize", True))
    contract.target_col = str(parsed.get("target_col", "")).strip()
    contract.id_col = str(parsed.get("id_col", "id")).strip()
    category = str(parsed.get("category", "tabular")).strip()
    if category in {"tabular", "vision", "nlp", "timeseries", "audio"}:
        contract.category = category

    logger.info(
        f"[contract] inferred: metric={contract.metric} maximize={contract.maximize} "
        f"target={contract.target_col} id={contract.id_col} category={contract.category}"
    )
    return contract


def _parse_json(text: str) -> dict | None:
    text = text.strip()
    # Strip markdown fences if present
    if text.startswith("```"):
        text = re.sub(r"^```\w*\s*\n?", "", text)
        text = re.sub(r"\n?\s*```\s*$", "", text)
        text = text.strip()
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def _read_file(data_dir: Path, *names: str) -> str:
    for name in names:
        path = data_dir / name
        if path.exists():
            return path.read_text(encoding="utf-8", errors="replace")
    return ""


def _read_first_line(path: Path) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return lines[0] if lines else ""
