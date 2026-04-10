"""Parse VAL / HOLDOUT scores and metric direction from solution stdout.

Every generated solution is required (by prompt) to print:

    FINAL VAL SCORE: <number>
    FINAL HOLDOUT SCORE: <number>
    METRIC DIRECTION: maximize | minimize

This module is the *only* place that knows how to read those lines back out.
We're permissive about whitespace, scientific notation, optional leading
``-``, and the case of the key. Several alternative phrasings are accepted
so small drift in the LLM's output formatting does not break parsing.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger("solver")


_NUMBER_RE = r"(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"


def _make_pattern(*key_phrases: str) -> re.Pattern[str]:
    options = "|".join(re.escape(p) for p in key_phrases)
    return re.compile(
        rf"(?:{options})\s*[:=]?\s*{_NUMBER_RE}",
        re.IGNORECASE,
    )


_VAL_RE = _make_pattern(
    "FINAL VAL SCORE",
    "Final Validation Score",
    "Final Val Score",
    "VAL SCORE",
)
_HOLDOUT_RE = _make_pattern(
    "FINAL HOLDOUT SCORE",
    "Final Holdout Score",
    "HOLDOUT SCORE",
)
_METRIC_DIR_RE = re.compile(
    r"METRIC\s*DIRECTION\s*[:=]?\s*(maximize|minimize|max|min|higher|lower)",
    re.IGNORECASE,
)
_FOLD_SCORES_RE = re.compile(
    r"(?:CV\s*FOLD\s*SCORES?|FOLD\s*SCORES?)\s*[:=]?\s*(.+)",
    re.IGNORECASE,
)


@dataclass
class ParsedScores:
    val_score: float | None = None
    holdout_score: float | None = None
    maximize: bool | None = None
    fold_scores: tuple[float, ...] = ()

    def is_valid(self) -> bool:
        return self.val_score is not None and not (
            self.val_score != self.val_score  # NaN check
        )


def parse_scores(output: str) -> ParsedScores:
    """Extract VAL / HOLDOUT scores and metric direction from solution output."""
    if not output:
        return ParsedScores()

    val = _last_match(_VAL_RE, output)
    hold = _last_match(_HOLDOUT_RE, output)

    direction = None
    m = _METRIC_DIR_RE.search(output)
    if m:
        word = m.group(1).lower()
        if word in {"maximize", "max", "higher"}:
            direction = True
        elif word in {"minimize", "min", "lower"}:
            direction = False

    fold_scores = _last_fold_scores(output)
    return ParsedScores(
        val_score=val,
        holdout_score=hold,
        maximize=direction,
        fold_scores=fold_scores,
    )


def _last_match(pattern: re.Pattern[str], text: str) -> float | None:
    matches = pattern.findall(text)
    if not matches:
        return None
    try:
        return float(matches[-1])
    except (ValueError, TypeError):
        return None


def _last_fold_scores(text: str) -> tuple[float, ...]:
    matches = _FOLD_SCORES_RE.findall(text or "")
    if not matches:
        return ()
    raw = matches[-1]
    numbers = re.findall(_NUMBER_RE, raw)
    out: list[float] = []
    for value in numbers:
        try:
            out.append(float(value))
        except (TypeError, ValueError):
            continue
    return tuple(out)


def is_better(score: float | None, current_best: float | None, maximize: bool) -> bool:
    if score is None:
        return False
    if current_best is None:
        return True
    if maximize:
        return score > current_best
    return score < current_best


def holdout_gap_abs(parsed: ParsedScores) -> float | None:
    """Absolute val/holdout gap. Useful for accuracy/AUC-style metrics."""
    if parsed.val_score is None or parsed.holdout_score is None:
        return None
    return abs(parsed.val_score - parsed.holdout_score)


def holdout_gap_relative(parsed: ParsedScores) -> float | None:
    """Relative val/holdout gap.

    Defined as ``|val - holdout| / max(|val|, |holdout|, 0.01)``. This makes
    the threshold meaningful across metrics with very different scales:
    log-loss might be ~0.5, RMSE might be ~5.0, AUC is ~0.85. The relative
    gap is ~0.20 only when val and holdout disagree by 20% of the larger
    magnitude — a sensible "this might be overfitting" threshold.
    """
    if parsed.val_score is None or parsed.holdout_score is None:
        return None
    denom = max(abs(parsed.val_score), abs(parsed.holdout_score), 0.01)
    return abs(parsed.val_score - parsed.holdout_score) / denom


# Backwards-compat alias.
holdout_gap = holdout_gap_abs
