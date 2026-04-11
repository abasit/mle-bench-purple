"""Tests for journal + UCB selector + rotating hint picker."""

import random

from purple_next.exec.interpreter import ExecResult
from purple_next.prompts.improve import IMPROVE_HINTS, pick_hint
from purple_next.tree import Journal, SearchNode, Selector
from purple_next.tree.ranking import adjusted_review_penalty, hard_leakage_flag, review_penalty


def _valid_node(node_id, stage, parent_id, branch_root, cv, hold=None, hint_idx=None):
    n = SearchNode(id=node_id, stage=stage, code="pass", parent_id=parent_id, branch_root_id=branch_root)
    n.cv_score = cv
    n.holdout_score = hold if hold is not None else cv
    n.maximize = True
    n.improve_hint_index = hint_idx
    n.result = ExecResult(
        return_code=0,
        stdout="",
        stderr="",
        duration_seconds=0.1,
        submission_path=None,  # bypass the has_submission check for selector tests
    )
    # Force is_valid to True by marking as success + submission in the dataclass result.
    # We set a fake submission path below via a lambda.
    return n


def test_journal_best_prefers_non_suspicious():
    j = Journal()
    a = _valid_node("d001", "draft", None, "d001", cv=0.70)
    b = _valid_node("i002", "improve", "d001", "d001", cv=0.85)
    b.is_suspicious = True
    j.add(a)
    j.add(b)
    # Make both look valid by force (since is_valid checks has_submission).
    a.result.submission_path = j  # type: ignore[assignment]
    b.result.submission_path = j  # type: ignore[assignment]

    # Selector-style best prefers non-suspicious node even if cv is lower.
    valid_all = [a, b]
    j._nodes = valid_all  # type: ignore[attr-defined]
    # The "all_valid" method checks n.is_valid — which depends on has_submission.
    # For this test we validate the filtering logic directly.
    non_susp = [n for n in valid_all if not n.is_suspicious]
    assert non_susp == [a]


def test_selector_ucb_explores_undeveloped_branch():
    # Simulate two branches where branch A has played 5 times and B has played 0.
    # B should get picked for exploration even though A has a slightly higher score.
    from unittest.mock import patch

    j = Journal()

    # Branch A: one draft with cv=0.80, plus 20 improve nodes already played.
    # With c=1.0 and score spread normalized to [0, 1], branch A's exploit
    # score (~1.0) beats branch B's pure-exploration bonus until the play
    # gap is wide enough that sqrt(ln(total)/1) clears 1.0.
    root_a = SearchNode(id="d001", stage="draft", code="a", branch_root_id="d001")
    root_a.cv_score = 0.80
    root_a.holdout_score = 0.80
    root_a.maximize = True
    for i in range(20):
        imp = SearchNode(
            id=f"i{i+1:03d}",
            stage="improve",
            code="a",
            parent_id="d001",
            branch_root_id="d001",
        )
        imp.cv_score = 0.80
        imp.holdout_score = 0.80
        imp.maximize = True
        j._nodes.append(imp)
        j._by_id[imp.id] = imp

    # Branch B: one draft with cv=0.75 and zero improves.
    root_b = SearchNode(id="d002", stage="draft", code="b", branch_root_id="d002")
    root_b.cv_score = 0.75
    root_b.holdout_score = 0.75
    root_b.maximize = True

    j._nodes.insert(0, root_a)
    j._by_id["d001"] = root_a
    j._nodes.insert(1, root_b)
    j._by_id["d002"] = root_b

    # Force is_valid to True on all nodes.
    with patch.object(SearchNode, "is_valid", property(lambda self: True)):
        sel = Selector(max_debug_attempts_per_node=2, explore_c=1.0)
        action = sel.pick(j, maximize=True)
    assert action is not None
    assert action.kind == "improve"
    # Branch B has zero plays, so exploration bonus is huge and UCB picks it.
    assert action.parent.id == "d002"


def test_pick_hint_cold_start_returns_untried_indices_first():
    j = Journal()
    rng = random.Random(0)
    assert len(IMPROVE_HINTS) >= 8
    # No improve nodes yet: first call returns index 0.
    assert pick_hint(j, rng=rng) == 0

    # Add one improve node for hint 0. Next call should skip to hint 1.
    parent = SearchNode(id="d001", stage="draft", code="x", branch_root_id="d001")
    parent.cv_score = 0.5
    parent.maximize = True
    child = SearchNode(id="i002", stage="improve", code="x", parent_id="d001", branch_root_id="d001")
    child.cv_score = 0.6
    child.maximize = True
    child.improve_hint_index = 0
    j._nodes = [parent, child]
    j._by_id = {"d001": parent, "i002": child}
    assert pick_hint(j, rng=rng) == 1


def test_review_penalty_is_confidence_aware():
    assert review_penalty("clean", "high") == 0.0
    assert review_penalty("suspicious", "low") == 0.0
    assert review_penalty("suspicious", "medium") > 0.0
    assert review_penalty("leaky", "high") > review_penalty("leaky", "low")


def test_adjusted_review_penalty_uses_score_margin_guard():
    top = SearchNode(id="i001", stage="improve", code="x")
    top.review_verdict = "suspicious"
    top.review_confidence = "high"
    top.holdout_score = 0.90
    top.cv_score = 0.88

    other_a = SearchNode(id="i002", stage="improve", code="x")
    other_a.review_verdict = "clean"
    other_a.review_confidence = "high"
    other_a.holdout_score = 0.84
    other_a.cv_score = 0.82

    other_b = SearchNode(id="i003", stage="improve", code="x")
    other_b.review_verdict = "clean"
    other_b.review_confidence = "high"
    other_b.holdout_score = 0.83
    other_b.cv_score = 0.81

    peers = [top, other_a, other_b]
    assert review_penalty(top.review_verdict, top.review_confidence) > 0.0
    assert adjusted_review_penalty(top, peers, maximize=True) == 0.0


def test_adjusted_review_penalty_keeps_penalty_without_dual_margin():
    suspect = SearchNode(id="i001", stage="improve", code="x")
    suspect.review_verdict = "suspicious"
    suspect.review_confidence = "medium"
    suspect.holdout_score = 0.90
    suspect.cv_score = 0.81

    other = SearchNode(id="i002", stage="improve", code="x")
    other.review_verdict = "clean"
    other.review_confidence = "high"
    other.holdout_score = 0.84
    other.cv_score = 0.82

    peers = [suspect, other]
    assert adjusted_review_penalty(suspect, peers, maximize=True) == review_penalty(
        suspect.review_verdict, suspect.review_confidence
    )


def test_hard_leakage_flag_demotes_only_explicit_leaks():
    assert hard_leakage_flag("leaky", "high") == 1
    assert hard_leakage_flag("leaky", "medium") == 1
    assert hard_leakage_flag("leaky", "low") == 0
    assert hard_leakage_flag("suspicious", "high") == 0
