"""purple_next — the distilled purple agent.

Runner-owned validation protocol + LLM-parsed scores + LLM leakage review,
wrapped in a UCB-driven tree search with per-run disposition diversity.

Entry point: ``purple_next.runner.run_competition(work_dir)``
"""

from .config import Config
from .runner import run_competition

__all__ = ["Config", "run_competition"]
