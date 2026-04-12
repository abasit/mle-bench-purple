"""Debug prompt + cheap error classifier."""

from __future__ import annotations

from textwrap import dedent

from .system import SYSTEM_PROMPT


def classify_error(summary: str, log_tail: str) -> tuple[str, str]:
    """Return (error_class, fix_focus) — a one-line hint for the debug LLM call."""
    combined = (summary + "\n" + log_tail).lower()
    if any(k in combined for k in ("out of memory", "oom", "cuda out of memory", "memory error", "killed")):
        return "oom", "Reduce memory: smaller batches/folds/models, stream data, lower input sizes."
    if any(k in combined for k in ("shape mismatch", "cannot reshape", "size mismatch", "broadcast")):
        return "shape_mismatch", "Print and align train/valid/test matrix and prediction shapes."
    if any(k in combined for k in ("filenotfounderror", "no such file", "not a directory")):
        return "data_loading", "Discover actual files under ./input before hard-coding paths."
    if any(k in combined for k in ("modulenotfounderror", "importerror", "no module named")):
        return "import_error", "No pip install — replace the missing dependency with an installed one."
    if any(k in combined for k in ("could not convert string to float", "dtype", "cannot convert")):
        return "dtype_error", "Cast cat/object columns to string and encode; assert no object dtypes before fit."
    if any(k in combined for k in ("timeout", "timed out", "time limit")):
        return "timeout", "Simplify: fewer folds/estimators/epochs and smaller inputs."
    if any(k in combined for k in ("cuda", "gpu", "device", "nccl")):
        return "cuda", "Make device placement explicit or switch to a CPU-safe route."
    if any(k in combined for k in ("unicodedecodeerror", "encoding", "codec", "charmap")):
        return "encoding", "Try utf-8 then latin-1; verify file type before reading as text."
    return "general", "Fix the root cause shown in the log tail."


def build_debug_prompt(
    *,
    parent_code: str,
    error_summary: str,
    log_tail: str,
    contract_summary: str,
    data_preview: str = "",
    time_remaining_s: float,
) -> list[dict[str, str]]:
    error_class, fix_focus = classify_error(error_summary, log_tail)
    data_block = f"Data preview:\n{data_preview.strip()}" if data_preview.strip() else ""
    user = dedent(
        f"""
        Fix this crashed solution. Return the complete corrected solution.py.

        Error class: {error_class}
        Error: {error_summary or '(unknown)'}
        Fix focus: {fix_focus}
        Time remaining: {time_remaining_s:.0f}s

        Protocol (runner-owned, still in force):
        {contract_summary}

        {data_block}

        Log tail:
        ```
        {log_tail[-3000:] if log_tail else '(empty)'}
        ```

        Current code:
        ```python
        {parent_code}
        ```

        Fix the root cause. If the approach is brittle, simplify it.
        Keep loading ./input/_splits.csv and print OUTCOME_JSON at the end.
        """
    ).strip()
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]
