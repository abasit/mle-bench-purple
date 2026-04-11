"""Tests for the LLM-based parser agent.

The direct happy path reads an OUTCOME_JSON line without calling the LLM.
The fallback path calls a stub LLMClient that returns canned JSON.
"""

from purple_next.agents.parser import parse_outcome


class StubLLM:
    def __init__(self, response: str):
        self._response = response
        self.calls = 0

    def chat(self, messages, *, temperature=None, max_tokens=None, label="chat"):
        self.calls += 1
        return self._response


def test_parse_outcome_direct_from_outcome_json_line():
    llm = StubLLM("")
    stdout = (
        "training...\n"
        "done\n"
        'OUTCOME_JSON: {"cv_score": 0.812, "holdout_score": 0.795, "notes": "ok"}\n'
    )
    parsed = parse_outcome(
        llm=llm,
        code="print('hi')",
        stdout=stdout,
        stderr="",
        maximize=True,
    )
    assert parsed.cv_score == 0.812
    assert parsed.holdout_score == 0.795
    assert parsed.bug is False
    assert parsed.source == "direct"
    assert llm.calls == 0


def test_parse_outcome_fallback_llm_when_no_direct_line():
    llm = StubLLM('{"cv_score": 0.76, "holdout_score": 0.74, "bug": false, "notes": "solid"}')
    parsed = parse_outcome(
        llm=llm,
        code="print('hi')",
        stdout="CV: 0.76\nHoldout: 0.74\n",
        stderr="",
        maximize=True,
    )
    assert parsed.cv_score == 0.76
    assert parsed.holdout_score == 0.74
    assert parsed.bug is False
    assert parsed.source == "llm"
    assert llm.calls == 1


def test_parse_outcome_marks_bug_when_llm_returns_nonjson():
    llm = StubLLM("not json at all")
    parsed = parse_outcome(
        llm=llm,
        code="print('hi')",
        stdout="something",
        stderr="",
        maximize=True,
    )
    assert parsed.bug is True
    assert parsed.source == "missing"
