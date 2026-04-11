"""LLM-based leakage reviewer.

Runs ONCE per top-K candidate (not per node) with the full code + task
description + runner protocol + scores. Returns a structured verdict so
the final ranker can demote suspected leaky candidates without false
positives on easy tasks.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from textwrap import dedent

from ..llm import LLMClient

logger = logging.getLogger("purple_next")


@dataclass
class ReviewVerdict:
    verdict: str = "clean"             # "clean" | "suspicious" | "leaky"
    confidence: str = "low"            # "low" | "medium" | "high"
    reasons: list[str] = field(default_factory=list)
    summary: str = ""


_SYS = dedent(
    """
    You are a senior ML engineer reviewing a Kaggle solution for data leakage
    and honesty problems. You will see the code, the task description, the
    runner's protocol (train/dev/holdout split), and the reported scores.

    Categories to look for:
    1. Train/validation contamination — fitting transformers or scalers on the
       full labeled dataset (including dev+holdout rows) before splitting.
    2. Holdout touched during training — features or model fit seeing rows
       where _splits.csv says split=="holdout".
    3. Incorrect CV structure — shuffled CV on time-ordered data, breaking
       group structure, using test labels.
    4. Target leakage — features that encode the label directly or post-event
       information that wouldn't be available at prediction time.
    5. Fake-success patterns — try/except that silently writes a constant
       submission and prints a placeholder score.

    Only flag what you can point to in the code. If the task is legitimately
    easy and the scores are high without any suspicious code, return "clean".

    Return ONLY a JSON object:
    - "verdict": "clean" | "suspicious" | "leaky"
    - "confidence": "low" | "medium" | "high"
    - "reasons": list of short strings (may be empty)
    - "summary": one sentence
    """
).strip()


def review_candidate(
    *,
    llm: LLMClient,
    code: str,
    task_desc: str,
    contract_summary: str,
    cv_score: float | None,
    holdout_score: float | None,
    label: str,
) -> ReviewVerdict:
    cv_s = f"{cv_score:.5f}" if cv_score is not None else "N/A"
    ho_s = f"{holdout_score:.5f}" if holdout_score is not None else "N/A"
    user = dedent(
        f"""
        Task description:
        {task_desc[:4000]}

        Runner protocol:
        {contract_summary}

        Reported scores: cv={cv_s}, holdout={ho_s}

        Code:
        ```python
        {code[:8000]}
        ```

        Return the JSON verdict described in the system message and nothing else.
        """
    ).strip()
    try:
        response = llm.chat(
            [{"role": "system", "content": _SYS}, {"role": "user", "content": user}],
            label=label,
        )
    except Exception as e:
        logger.warning(f"[reviewer] failed: {e}")
        return ReviewVerdict(verdict="clean", confidence="low", summary=f"reviewer failed: {e}")

    text = response.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:].lstrip()
    try:
        payload = json.loads(text)
    except Exception:
        return ReviewVerdict(verdict="clean", confidence="low", summary="reviewer returned non-json")
    if not isinstance(payload, dict):
        return ReviewVerdict(verdict="clean", confidence="low", summary="reviewer returned non-object")

    verdict = str(payload.get("verdict", "clean")).lower()
    if verdict not in {"clean", "suspicious", "leaky"}:
        verdict = "clean"
    confidence = str(payload.get("confidence", "low")).lower()
    if confidence not in {"low", "medium", "high"}:
        confidence = "low"
    reasons_raw = payload.get("reasons", []) or []
    reasons = [str(r)[:200] for r in reasons_raw if r][:6]
    summary = str(payload.get("summary", ""))[:240]
    return ReviewVerdict(verdict=verdict, confidence=confidence, reasons=reasons, summary=summary)
