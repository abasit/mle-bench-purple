"""LLM-based score parser.

Reads the solution's stdout + code and returns a structured outcome:
``cv_score``, ``holdout_score``, ``bug`` (did the script actually train a
model?), ``notes``. A one-line JSON tag ``OUTCOME_JSON: {...}`` in the
script's stdout is the happy path; when missing or malformed we fall back
to asking the LLM to extract.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from textwrap import dedent

from ..llm import LLMClient

logger = logging.getLogger("purple_next")


_OUTCOME_LINE_RE = re.compile(r"OUTCOME_JSON\s*:\s*(\{.*\})", re.DOTALL)


@dataclass
class ParsedOutcome:
    cv_score: float | None = None
    holdout_score: float | None = None
    bug: bool = False
    notes: str = ""
    source: str = "direct"             # "direct" (json line) | "llm" | "missing"


def parse_outcome(
    *,
    llm: LLMClient,
    code: str,
    stdout: str,
    stderr: str,
    maximize: bool,
) -> ParsedOutcome:
    # Happy path: OUTCOME_JSON: {"cv_score": ..., "holdout_score": ..., "notes": "..."}
    direct = _parse_direct(stdout)
    if direct is not None:
        return direct

    # Fallback: ask the LLM to read the output.
    return _llm_fallback(llm=llm, code=code, stdout=stdout, stderr=stderr, maximize=maximize)


def _parse_direct(stdout: str) -> ParsedOutcome | None:
    if not stdout:
        return None
    matches = _OUTCOME_LINE_RE.findall(stdout)
    if not matches:
        return None
    raw = matches[-1].strip()
    try:
        payload = json.loads(raw)
    except Exception:
        # Try trimming at the first closing brace to handle trailing noise.
        end = raw.find("}")
        if end == -1:
            return None
        try:
            payload = json.loads(raw[: end + 1])
        except Exception:
            return None
    if not isinstance(payload, dict):
        return None
    return ParsedOutcome(
        cv_score=_to_float(payload.get("cv_score")),
        holdout_score=_to_float(payload.get("holdout_score")),
        bug=bool(payload.get("bug", False)),
        notes=str(payload.get("notes", ""))[:240],
        source="direct",
    )


def _to_float(value) -> float | None:
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if f != f:  # NaN
        return None
    return f


_SYS = dedent(
    """
    You are a Kaggle grandmaster acting as a parser. Given the code and stdout
    of a solution script, extract the validation metrics and whether the script
    actually trained a model or hid a crash behind a fallback.

    Return ONLY a JSON object with these fields:
    - "cv_score": float or null (cross-validation OOF score on dev folds)
    - "holdout_score": float or null (score on the runner-provided holdout slice)
    - "bug": boolean (true if the script clearly did not train/evaluate a real model)
    - "notes": string (one short sentence)

    Report scores in the metric's native direction — do not sign-flip.
    """
).strip()


def _llm_fallback(
    *,
    llm: LLMClient,
    code: str,
    stdout: str,
    stderr: str,
    maximize: bool,
) -> ParsedOutcome:
    tail_out = stdout[-3500:] if stdout else ""
    tail_err = stderr[-1500:] if stderr else ""
    direction = "maximize" if maximize else "minimize"
    user = dedent(
        f"""
        Metric direction: {direction}

        Code:
        ```python
        {code[-4000:]}
        ```

        Stdout (tail):
        ```
        {tail_out or '(empty)'}
        ```

        Stderr (tail):
        ```
        {tail_err or '(empty)'}
        ```

        Return the JSON object described in the system message and nothing else.
        """
    ).strip()

    try:
        response = llm.chat(
            [{"role": "system", "content": _SYS}, {"role": "user", "content": user}],
            label="parser",
        )
    except Exception as e:
        logger.warning(f"[parser] llm fallback failed: {e}")
        return ParsedOutcome(bug=True, notes=f"parser failed: {e}", source="missing")

    text = response.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:].lstrip()
    try:
        payload = json.loads(text)
    except Exception:
        return ParsedOutcome(bug=True, notes="parser returned non-json", source="missing")
    if not isinstance(payload, dict):
        return ParsedOutcome(bug=True, notes="parser returned non-object", source="missing")
    return ParsedOutcome(
        cv_score=_to_float(payload.get("cv_score")),
        holdout_score=_to_float(payload.get("holdout_score")),
        bug=bool(payload.get("bug", False)),
        notes=str(payload.get("notes", ""))[:240],
        source="llm",
    )
