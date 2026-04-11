"""Tests for the leakage reviewer agent."""

from purple_next.agents.reviewer import review_candidate


class StubLLM:
    def __init__(self, response: str):
        self._response = response

    def chat(self, messages, *, temperature=None, max_tokens=None, label="chat"):
        return self._response


def test_reviewer_parses_clean_verdict():
    llm = StubLLM('{"verdict": "clean", "confidence": "medium", "reasons": [], "summary": "OK"}')
    verdict = review_candidate(
        llm=llm,
        code="import pandas as pd",
        task_desc="binary classification",
        contract_summary="metric: auc (maximize)",
        cv_score=0.82,
        holdout_score=0.80,
        label="review-test",
    )
    assert verdict.verdict == "clean"
    assert verdict.confidence == "medium"
    assert verdict.reasons == []


def test_reviewer_parses_leaky_verdict_with_reasons():
    response = (
        '{"verdict": "leaky", "confidence": "high", '
        '"reasons": ["fits StandardScaler on holdout rows", "shuffled CV on timeseries"], '
        '"summary": "scaler trained on holdout"}'
    )
    llm = StubLLM(response)
    verdict = review_candidate(
        llm=llm,
        code="scaler.fit(all_rows)",
        task_desc="binary classification",
        contract_summary="metric: auc (maximize)",
        cv_score=0.99,
        holdout_score=0.98,
        label="review-test",
    )
    assert verdict.verdict == "leaky"
    assert verdict.confidence == "high"
    assert "fits StandardScaler on holdout rows" in verdict.reasons


def test_reviewer_handles_malformed_response_as_clean_low_confidence():
    llm = StubLLM("??? not json")
    verdict = review_candidate(
        llm=llm,
        code="",
        task_desc="",
        contract_summary="",
        cv_score=None,
        holdout_score=None,
        label="review-test",
    )
    assert verdict.verdict == "clean"
    assert verdict.confidence == "low"
