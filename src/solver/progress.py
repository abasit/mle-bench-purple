"""Progress callback protocol for real-time step-by-step streaming.

The runner calls these at key milestones. The agent layer can wire them
to A2A TaskUpdater for live status updates.

Usage:
    callback = ProgressCallback()
    run_competition(work_dir, progress=callback)
"""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

logger = logging.getLogger("solver")


@runtime_checkable
class ProgressCallback(Protocol):
    """Protocol for receiving progress updates from the solver."""

    def on_phase(self, phase: str, message: str) -> None:
        """Called when a major phase starts or ends.

        phase: "explore", "drafts", "search", "finalize"
        """
        ...

    def on_step(self, step: int, total: int, message: str) -> None:
        """Called after each search step completes.

        step: current step number (1-based)
        total: max_steps from config
        message: short description (e.g. "improve d001 → i004 val=0.847")
        """
        ...

    def on_best(self, node_id: str, val_score: float | None, message: str) -> None:
        """Called when a new best score is found."""
        ...


class LoggingProgress:
    """Default implementation that just logs to the solver logger."""

    def on_phase(self, phase: str, message: str) -> None:
        logger.info(f"[progress] phase={phase}: {message}")

    def on_step(self, step: int, total: int, message: str) -> None:
        logger.info(f"[progress] step {step}/{total}: {message}")

    def on_best(self, node_id: str, val_score: float | None, message: str) -> None:
        logger.info(f"[progress] new best: {node_id} val={val_score}: {message}")
