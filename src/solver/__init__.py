"""Solver — tree-search ML engineering agent for MLE-bench competitions.

A clean, parallel, debug-aware scaffold that:
    - generates several diverse drafts in parallel,
    - debugs and improves them under a wall-clock budget,
    - blends the top-K validated submissions into a final answer.

Public entry points:
    ``run_competition(work_dir) -> bytes | None``
    ``run_competition_candidates(work_dir) -> list[bytes]``
"""

from .progress import LoggingProgress, ProgressCallback
from .runner import run_competition, run_competition_candidates

__all__ = [
    "run_competition",
    "run_competition_candidates",
    "ProgressCallback",
    "LoggingProgress",
]
