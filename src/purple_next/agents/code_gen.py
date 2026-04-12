"""Code generation agents — draft, improve, debug.

Each agent is a thin wrapper around a prompt builder + LLM chat call +
code extraction. No business logic beyond routing.
"""

from __future__ import annotations

import logging

from ..llm import LLMClient
from ..prompts import build_debug_prompt, build_draft_prompt, build_improve_prompt

logger = logging.getLogger("purple_next")


def generate_draft_code(
    *,
    llm: LLMClient,
    task_desc: str,
    data_files: list[str],
    data_preview: str,
    contract_summary: str,
    env_summary: str,
    time_remaining_s: float,
    disposition: str,
    variant: int,
    temperature: float | None,
    label: str,
) -> str:
    messages = build_draft_prompt(
        task_desc=task_desc,
        data_files=data_files,
        data_preview=data_preview,
        contract_summary=contract_summary,
        env_summary=env_summary,
        time_remaining_s=time_remaining_s,
        disposition=disposition,
        variant=variant,
    )
    response = llm.chat(messages, temperature=temperature, label=label)
    return llm.extract_python_code(response)


def generate_improve_code(
    *,
    llm: LLMClient,
    parent_code: str,
    parent_cv: float | None,
    parent_holdout: float | None,
    parent_stdout_tail: str,
    direction: str,
    hint_index: int,
    contract_summary: str,
    data_preview: str = "",
    time_remaining_s: float,
    fraction_used: float,
    temperature: float | None,
    label: str,
) -> str:
    messages = build_improve_prompt(
        parent_code=parent_code,
        parent_cv=parent_cv,
        parent_holdout=parent_holdout,
        parent_stdout_tail=parent_stdout_tail,
        direction=direction,
        hint_index=hint_index,
        contract_summary=contract_summary,
        data_preview=data_preview,
        time_remaining_s=time_remaining_s,
        fraction_used=fraction_used,
    )
    response = llm.chat(messages, temperature=temperature, label=label)
    code = llm.extract_python_code(response)
    return code or parent_code


def generate_debug_code(
    *,
    llm: LLMClient,
    parent_code: str,
    error_summary: str,
    log_tail: str,
    contract_summary: str,
    data_preview: str = "",
    time_remaining_s: float,
    temperature: float | None,
    label: str,
) -> str:
    messages = build_debug_prompt(
        parent_code=parent_code,
        error_summary=error_summary,
        log_tail=log_tail,
        contract_summary=contract_summary,
        data_preview=data_preview,
        time_remaining_s=time_remaining_s,
    )
    response = llm.chat(messages, temperature=temperature, label=label)
    code = llm.extract_python_code(response)
    return code or parent_code
