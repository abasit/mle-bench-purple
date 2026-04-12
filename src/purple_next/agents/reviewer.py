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
    You are a senior ML engineer reviewing a Kaggle solution for data leakage.

    LEAKY (flag these):
    - StandardScaler/MinMaxScaler/PCA fit on full data (dev+holdout) before splitting
    - Target encoding or group-mean features computed on data that includes holdout rows
    - Model trained on holdout rows (split=="holdout" rows used in fit())
    - Test labels used anywhere during training
    - SMOTE/upsampling applied before train/val split

    SAFE (do NOT flag these):
    - LabelEncoder / pd.factorize / OrdinalEncoder fit on combined train+test (just maps strings to ints, no statistical leakage)
    - fillna with a constant or median (negligible information, standard practice)
    - One-hot encoding on combined train+test
    - Using _splits.csv to separate dev/holdout and only training on dev rows

    Only flag what you can point to in the code with a specific line or pattern.
    If the code properly loads _splits.csv and only trains on dev rows, it is likely clean.

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
    temperature: float | None = None,
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
            temperature=temperature,
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
