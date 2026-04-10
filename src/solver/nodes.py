"""Search-tree nodes and the journal.

A node represents one full ``solution.py`` along with its execution result.
Each node has a parent (None for drafts), so the tree shape is implicit.

The Journal owns all nodes and provides:
    - id assignment
    - lookup by id
    - top-K selection by val score (with optional honesty filter)
    - debug-target identification
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from .interpreter import ExecResult
from .parsing import ParsedScores

logger = logging.getLogger("solver")


@dataclass
class SearchNode:
    """One full solution attempt."""

    id: str
    stage: str  # "draft" | "improve" | "debug"
    code: str
    plan: str = ""
    parent_id: str | None = None
    # The original draft this node descends from. Drafts are self-rooted
    # (branch_root_id == id). Debug/improve children inherit from their parent.
    branch_root_id: str | None = None
    created_at: float = field(default_factory=time.time)

    # Execution outcome — populated after the interpreter runs.
    result: ExecResult | None = None
    scores: ParsedScores = field(default_factory=ParsedScores)

    # Bookkeeping flags.
    is_buggy: bool = False
    is_suspicious: bool = False  # large val/holdout gap
    suspicion_reasons: list[str] = field(default_factory=list)
    debug_attempts: int = 0      # how many times we've tried to fix this lineage

    # Cached "lessons" string used by improve/debug operators of children.
    summary: str = ""

    # Strategy tags from the controlled vocabulary in ``solver.strategies``.
    # Populated by ``collect_strategies(response, code)`` after each LLM call.
    strategies: set[str] = field(default_factory=set)

    # ── derived ────────────────────────────────────────────────────────────

    @property
    def val_score(self) -> float | None:
        return self.scores.val_score

    @property
    def holdout_score(self) -> float | None:
        return self.scores.holdout_score

    @property
    def has_submission(self) -> bool:
        return self.result is not None and self.result.has_submission

    @property
    def is_valid(self) -> bool:
        """Valid = ran successfully + produced submission + has a parsed val score."""
        return (
            self.result is not None
            and self.result.is_success
            and self.has_submission
            and self.val_score is not None
            and not self.is_buggy
        )

    def submission_path(self) -> Path | None:
        if self.result is None:
            return None
        return self.result.submission_path

    def short(self) -> str:
        score = f"{self.val_score:.4f}" if self.val_score is not None else "N/A"
        gap = (
            f" gap={abs(self.val_score - self.holdout_score):.3f}"
            if self.val_score is not None and self.holdout_score is not None
            else ""
        )
        flag = ""
        if self.is_buggy:
            flag = " [BUGGY]"
        elif self.is_suspicious:
            flag = " [SUSPECT]"
        return f"{self.id}({self.stage} val={score}{gap}{flag})"


class Journal:
    """All nodes generated during a run."""

    def __init__(self):
        self._nodes: list[SearchNode] = []
        self._by_id: dict[str, SearchNode] = {}
        self._lock = threading.RLock()
        self._counter = 0

    def __len__(self) -> int:
        return len(self._nodes)

    def __iter__(self) -> Iterable[SearchNode]:
        return iter(list(self._nodes))

    # ── construction ──────────────────────────────────────────────────────

    def next_id(self, stage: str) -> str:
        with self._lock:
            self._counter += 1
            return f"{stage[:1]}{self._counter:03d}"

    def add(self, node: SearchNode) -> None:
        with self._lock:
            self._nodes.append(node)
            self._by_id[node.id] = node
            logger.info(f"[journal] +{node.short()} (total={len(self._nodes)})")

    def get(self, node_id: str) -> SearchNode | None:
        return self._by_id.get(node_id)

    def parent_of(self, node: SearchNode) -> SearchNode | None:
        if node.parent_id is None:
            return None
        return self._by_id.get(node.parent_id)

    # ── selection ─────────────────────────────────────────────────────────

    def all_valid(self) -> list[SearchNode]:
        return [n for n in self._nodes if n.is_valid]

    def all_buggy(self) -> list[SearchNode]:
        """Buggy nodes that haven't exhausted their debug budget."""
        return [n for n in self._nodes if n.is_buggy]

    def best(self, *, prefer_honest: bool = True) -> SearchNode | None:
        candidates = self.all_valid()
        if not candidates:
            return None
        if prefer_honest:
            honest = [n for n in candidates if not n.is_suspicious]
            if honest:
                candidates = honest
        return self._best(candidates)

    def top_k_by_val(self, k: int, *, prefer_honest: bool = True) -> list[SearchNode]:
        candidates = self.all_valid()
        if not candidates:
            return []
        if prefer_honest:
            honest = [n for n in candidates if not n.is_suspicious]
            if honest:
                candidates = honest
        # Determine maximize direction from any node that has it; default True.
        maximize = self._infer_maximize(candidates)
        ordered = sorted(
            candidates,
            key=lambda n: (n.val_score if n.val_score is not None else float("-inf")),
            reverse=maximize,
        )
        return ordered[:k]

    def _best(self, candidates: list[SearchNode]) -> SearchNode | None:
        if not candidates:
            return None
        maximize = self._infer_maximize(candidates)
        return max(
            candidates,
            key=lambda n: (
                (n.val_score if n.val_score is not None else float("-inf"))
                if maximize
                else -(n.val_score if n.val_score is not None else float("inf"))
            ),
        )

    def _infer_maximize(self, candidates: list[SearchNode]) -> bool:
        for n in candidates:
            if n.scores.maximize is not None:
                return n.scores.maximize
        return True  # default

    @property
    def maximize(self) -> bool:
        return self._infer_maximize(self.all_valid())

    # ── stats ─────────────────────────────────────────────────────────────

    def stats(self) -> dict[str, int]:
        return {
            "total": len(self._nodes),
            "valid": sum(1 for n in self._nodes if n.is_valid),
            "buggy": sum(1 for n in self._nodes if n.is_buggy),
            "suspicious": sum(1 for n in self._nodes if n.is_suspicious),
            "drafts": sum(1 for n in self._nodes if n.stage == "draft"),
            "improves": sum(1 for n in self._nodes if n.stage == "improve"),
            "debugs": sum(1 for n in self._nodes if n.stage == "debug"),
        }
