"""Improve prompt + rotating hint feedback picker."""

from __future__ import annotations

import random
from textwrap import dedent
from typing import TYPE_CHECKING

from .system import SYSTEM_PROMPT

if TYPE_CHECKING:  # pragma: no cover
    from ..tree.journal import Journal


IMPROVE_HINTS: tuple[tuple[str, str], ...] = (
    ("feature_engineering", "feature engineering — interactions, aggregates, domain-specific transforms"),
    ("model_swap",          "model swap — try a different family (or a stronger config of the same one)"),
    ("hyperparameters",     "hyperparameters — learning rate, depth, regularization, early stopping"),
    ("preprocessing",       "preprocessing — missing values, encoding, scaling, dtype cleanup"),
    ("validation",          "validation — inspect CV fold structure; check for leakage through groups/time"),
    ("post_processing",     "post-processing — threshold tuning, calibration, rank/clip, label mapping"),
    ("data_cleaning",       "data cleaning — outliers, duplicates, suspicious rows"),
    ("in_script_blend",     "in-script blend — average 2–3 diverse models of your current approach"),
)


def hint_label(index: int) -> str:
    return IMPROVE_HINTS[index % len(IMPROVE_HINTS)][0]


def hint_text(index: int) -> str:
    return IMPROVE_HINTS[index % len(IMPROVE_HINTS)][1]


def pick_hint(journal: "Journal", *, rng: random.Random | None = None) -> int:
    """Epsilon-greedy over smoothed win rates.

    Phase 1 (cold start): cycle every hint at least once.
    Phase 2: 30% random, 70% argmax of (wins + 1) / (attempts + 2) across hints,
    with random tie-breaking.
    """
    rng = rng or random
    n = len(IMPROVE_HINTS)
    stats: list[list[int]] = [[0, 0] for _ in range(n)]

    for node in journal:
        if node.stage != "improve" or node.improve_hint_index is None:
            continue
        idx = node.improve_hint_index
        if not (0 <= idx < n):
            continue
        parent = journal.parent_of(node)
        if parent is None or parent.cv_score is None or node.cv_score is None:
            continue
        stats[idx][1] += 1
        maximize = parent.maximize if parent.maximize is not None else True
        won = node.cv_score > parent.cv_score if maximize else node.cv_score < parent.cv_score
        if won:
            stats[idx][0] += 1

    for i, row in enumerate(stats):
        if row[1] == 0:
            return i

    if rng.random() < 0.3:
        return rng.randrange(n)
    scores = [(w + 1) / (a + 2) for w, a in stats]
    best = max(scores)
    winners = [i for i, s in enumerate(scores) if s == best]
    return rng.choice(winners)


def build_improve_prompt(
    *,
    parent_code: str,
    parent_cv: float | None,
    parent_holdout: float | None,
    parent_stdout_tail: str,
    direction: str,
    hint_index: int,
    contract_summary: str,
    data_preview: str = "",
    time_remaining_s: float,
    fraction_used: float,
) -> list[dict[str, str]]:
    cv_s = f"{parent_cv:.5f}" if parent_cv is not None else "N/A"
    ho_s = f"{parent_holdout:.5f}" if parent_holdout is not None else "N/A"
    stdout_block = ""
    if parent_stdout_tail.strip():
        stdout_block = "Parent run output (tail):\n```\n" + parent_stdout_tail.strip() + "\n```"
    data_block = ""
    if data_preview.strip():
        data_block = f"Data preview:\n{data_preview.strip()}"
    user = dedent(
        f"""
        Improve this working solution. Return a complete new solution.py.

        Parent scores: cv={cv_s}, holdout={ho_s}; {direction}
        Time remaining: {time_remaining_s:.0f}s ({fraction_used:.0%} used)
        Focus this iteration on: {hint_text(hint_index)}

        Protocol (runner-owned, still in force):
        {contract_summary}

        {data_block}

        {stdout_block}

        Current code:
        ```python
        {parent_code}
        ```

        Make one targeted change. Keep loading ./input/_splits.csv and following it.
        Print OUTCOME_JSON at the end.
        """
    ).strip()
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]
