"""Shared ranking helpers for reviewed candidates."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .node import SearchNode


def review_penalty(verdict: str | None, confidence: str | None) -> float:
    """Return a soft demotion penalty from reviewer output.

    Penalty is deliberately confidence-aware:
    - low-confidence "suspicious" does not demote the candidate
    - medium/high suspicion applies a small demotion
    - "leaky" applies stronger demotion proportional to confidence
    """
    v = (verdict or "").strip().lower()
    c = (confidence or "").strip().lower()

    if v == "leaky":
        return {"high": 3.0, "medium": 2.0, "low": 1.0}.get(c, 2.0)
    if v == "suspicious":
        return {"high": 1.0, "medium": 0.25, "low": 0.0}.get(c, 0.25)
    return 0.0


def hard_leakage_flag(verdict: str | None, confidence: str | None) -> int:
    """Return 1 when leakage signal is explicit enough for hard demotion."""
    v = (verdict or "").strip().lower()
    c = (confidence or "").strip().lower()
    if v == "leaky" and c in {"medium", "high"}:
        return 1
    return 0


def adjusted_review_penalty(
    node: "SearchNode",
    peers: Sequence["SearchNode"],
    *,
    maximize: bool,
) -> float:
    """Base review penalty with score-margin guard for mild suspicion.

    If a ``suspicious`` candidate clearly leads peers on both holdout and CV,
    treat it as low-confidence suspicion (penalty zero). This prevents mild
    reviewer uncertainty from overriding strong validation signal.
    """
    base = review_penalty(node.review_verdict, node.review_confidence)
    verdict = (node.review_verdict or "").strip().lower()
    confidence = (node.review_confidence or "").strip().lower()

    if base <= 0.0:
        return 0.0
    if verdict != "suspicious" or confidence not in {"medium", "high"}:
        return base
    if _has_strong_dual_margin(node, peers, maximize=maximize):
        return 0.0
    return base


def _score(value: float | None, *, maximize: bool) -> float | None:
    if value is None:
        return None
    return value if maximize else -value


def _has_strong_dual_margin(
    node: "SearchNode",
    peers: Sequence["SearchNode"],
    *,
    maximize: bool,
) -> bool:
    own_hold = _score(node.holdout_score, maximize=maximize)
    own_cv = _score(node.cv_score, maximize=maximize)
    if own_hold is None or own_cv is None:
        return False

    other_hold = [
        s
        for other in peers
        if other is not node
        for s in [_score(other.holdout_score, maximize=maximize)]
        if s is not None
    ]
    other_cv = [
        s
        for other in peers
        if other is not node
        for s in [_score(other.cv_score, maximize=maximize)]
        if s is not None
    ]
    if not other_hold or not other_cv:
        return False

    hold_adv = own_hold - max(other_hold)
    cv_adv = own_cv - max(other_cv)
    if hold_adv <= 0.0 or cv_adv <= 0.0:
        return False

    hold_vals = [
        s
        for cand in peers
        for s in [_score(cand.holdout_score, maximize=maximize)]
        if s is not None
    ]
    cv_vals = [
        s
        for cand in peers
        for s in [_score(cand.cv_score, maximize=maximize)]
        if s is not None
    ]
    if len(hold_vals) < 2 or len(cv_vals) < 2:
        return False

    hold_spread = max(hold_vals) - min(hold_vals)
    cv_spread = max(cv_vals) - min(cv_vals)
    hold_threshold = max(1e-12, hold_spread * 0.25)
    cv_threshold = max(1e-12, cv_spread * 0.25)

    return hold_adv >= hold_threshold and cv_adv >= cv_threshold
