"""Draft prompt — per-run disposition + task description + data preview."""

from __future__ import annotations

from textwrap import dedent

from .system import SYSTEM_PROMPT


DISPOSITIONS: tuple[str, ...] = (
    "SPEED FIRST — Ship a valid, honest submission fast. Simple preprocessing, "
    "default hyperparameters, one CV run. Optimize later.",

    "DATA FIRST — Spend real effort understanding the data before modeling: "
    "schema, missing rates, class balance, target distribution, group and time "
    "structure. Let what you learn drive feature choices and model family.",

    "GO BIG — Push model capacity. For images/text/audio/sequences, use the "
    "right deep learning stack (torchvision, timm, transformers, torchaudio). "
    "For tabular, spend compute on a well-tuned GBDT with thoughtful features.",
)


def disposition_for_run(run_index: int, custom: list[str] | None = None) -> str:
    pool = custom if custom else list(DISPOSITIONS)
    if not pool:
        return ""
    return pool[run_index % len(pool)]


def build_draft_prompt(
    *,
    task_desc: str,
    data_files: list[str],
    data_preview: str,
    contract_summary: str,
    env_summary: str,
    time_remaining_s: float,
    disposition: str,
    variant: int,
) -> list[dict[str, str]]:
    files_str = "\n".join(f"- {f}" for f in data_files[:25])
    variant_label = f"Draft variant {variant + 1}"
    user = dedent(
        f"""
        RUN DISPOSITION:
        {disposition}

        {variant_label}. Aim for a different angle than the sibling drafts in this run.

        TASK:
        {task_desc.strip()}

        Protocol (runner-owned):
        {contract_summary}

        Environment: {env_summary or '(unknown)'}
        Time remaining: {time_remaining_s:.0f}s

        Data files:
        {files_str}

        Data preview:
        {data_preview.strip() or '(none)'}

        Reminder: load ./input/_splits.csv and follow it. Print OUTCOME_JSON at the end.
        """
    ).strip()
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]
