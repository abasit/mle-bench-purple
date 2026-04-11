"""Search tree node — one candidate solution attempt."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from ..exec.interpreter import ExecResult


@dataclass
class SearchNode:
    id: str
    stage: str                          # "draft" | "improve" | "debug"
    code: str
    parent_id: str | None = None
    branch_root_id: str | None = None
    created_at: float = field(default_factory=time.time)

    result: ExecResult | None = None

    # Scores assigned by the parser agent after execution.
    cv_score: float | None = None
    holdout_score: float | None = None
    maximize: bool | None = None
    notes: str = ""

    # Flags set during post-execution inspection.
    is_buggy: bool = False
    is_suspicious: bool = False
    suspicion_reasons: list[str] = field(default_factory=list)
    review_verdict: str = ""            # "clean" | "suspicious" | "leaky" | ""
    review_confidence: str = ""         # "low" | "medium" | "high" | ""
    review_reasons: list[str] = field(default_factory=list)
    debug_attempts: int = 0

    # For improve nodes: which rotating hint drove the iteration.
    improve_hint_index: int | None = None

    @property
    def submission_path(self) -> Path | None:
        return self.result.submission_path if self.result else None

    @property
    def is_valid(self) -> bool:
        return (
            self.result is not None
            and self.result.is_success
            and self.result.has_submission
            and self.cv_score is not None
            and self.holdout_score is not None
            and not self.is_buggy
        )

    def short(self) -> str:
        cv = f"{self.cv_score:.4f}" if self.cv_score is not None else "N/A"
        ho = f"{self.holdout_score:.4f}" if self.holdout_score is not None else "N/A"
        flag = ""
        if self.is_buggy:
            flag = " [BUGGY]"
        elif self.is_suspicious or self.review_verdict in {"suspicious", "leaky"}:
            conf = f"/{self.review_confidence}" if self.review_confidence else ""
            flag = f" [{self.review_verdict or 'SUSPECT'}{conf}]"
        return f"{self.id}({self.stage} cv={cv} hold={ho}{flag})"
