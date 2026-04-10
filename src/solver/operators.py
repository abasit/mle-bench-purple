"""Operators that turn a parent node + context into a new SearchNode.

Three operators only:
    - draft   : create a new branch from scratch (one of N parallel variants)
    - debug   : fix a buggy parent (preserves approach)
    - improve : propose a better version of a working parent (one focused change)

Each operator does ONE LLM call. They do not execute the resulting code —
that is the runner's job. They are intentionally pure.
"""

from __future__ import annotations

import logging

from .config import SolverConfig
from .llm import LLMClient
from .nodes import Journal, SearchNode
from .prompts import (
    build_debug_prompt,
    build_draft_prompt,
    build_improve_prompt,
    code_summary,
    summarise_other_attempts,
)
from .task_classify import TaskProfile
from .strategies import (
    BranchHistoryRow,
    build_branch_history,
    collect_strategies,
    cumulative_strategies,
    pick_required_strategy,
    untried_strategies,
)
from .utils import TimeBudget

logger = logging.getLogger("solver")


def make_draft(
    *,
    journal: Journal,
    llm: LLMClient,
    cfg: SolverConfig,
    task_desc: str,
    task_type: str,
    data_files: list[str],
    sample_sub_preview: str,
    kb_card: str,
    data_preview: str,
    task_profile_summary: str = "",
    task_profile: TaskProfile | None = None,
    env_summary: str = "",
    budget: TimeBudget,
    variant: int,
    total_variants: int,
) -> SearchNode:
    """Generate a fresh draft node (no execution yet)."""
    messages = build_draft_prompt(
        task_desc=task_desc,
        task_type=task_type,
        data_files=data_files,
        sample_sub_preview=sample_sub_preview,
        kb_card=kb_card,
        data_preview=data_preview,
        task_profile_summary=task_profile_summary,
        task_profile=task_profile,
        env_summary=env_summary,
        time_remaining=budget.remaining(),
        variant=variant,
        total_variants=total_variants,
    )
    response = llm.chat(
        messages,
        temperature=cfg.llm.temperature,
        label=f"draft_v{variant}",
    )
    code = llm.extract_python_code(response)
    plan = llm.extract_first_paragraph(response)
    if not code:
        logger.warning(f"[draft v{variant}] LLM returned no code block")

    node_id = journal.next_id("draft")
    node = SearchNode(
        id=node_id,
        stage="draft",
        code=code or "",
        plan=plan,
        parent_id=None,
    )
    # Drafts are self-rooted; downstream improves/debugs inherit this id.
    node.branch_root_id = node_id
    node.strategies = collect_strategies(response, node.code)
    node.summary = code_summary(node.code) if node.code else "(empty draft)"
    if node.strategies:
        logger.info(f"[draft v{variant}] {node_id} strategies: {sorted(node.strategies)}")
    return node


def make_debug(
    *,
    journal: Journal,
    llm: LLMClient,
    cfg: SolverConfig,
    parent: SearchNode,
    budget: TimeBudget,
    data_preview: str = "",
    task_profile_summary: str = "",
    task_profile: TaskProfile | None = None,
    env_summary: str = "",
    prior_attempts: list[SearchNode] | None = None,
) -> SearchNode:
    """Generate a debug-fix node from a buggy parent."""
    if parent.result is None:
        cleaned_log = "(no execution log available)"
        error_summary = "(no error summary)"
    else:
        cleaned_log = parent.result.cleaned_log(max_chars=6000)
        error_summary = parent.result.error_summary or "(no error summary parsed)"

    messages = build_debug_prompt(
        parent=parent,
        error_summary=error_summary,
        cleaned_log=cleaned_log,
        time_remaining=budget.remaining(),
        data_preview=data_preview,
        task_profile_summary=task_profile_summary,
        task_profile=task_profile,
        env_summary=env_summary,
        prior_attempts=prior_attempts or [],
    )
    response = llm.chat(
        messages,
        temperature=cfg.llm.temperature * 0.6,
        label=f"debug<-{parent.id}",
    )
    code = llm.extract_python_code(response)
    plan = llm.extract_first_paragraph(response)
    if not code:
        logger.warning(f"[debug from {parent.id}] LLM returned no code block")
        # Fall back to the parent code unchanged so the run can continue.
        code = parent.code

    node_id = journal.next_id("debug")
    node = SearchNode(
        id=node_id,
        stage="debug",
        code=code,
        plan=plan,
        parent_id=parent.id,
        # Per-node debug_attempts is incremented in Selector.pick(), NOT here.
        # The child starts at 0 so it can also be debugged if it fails.
        debug_attempts=0,
    )
    # Inherit branch root from parent so the strategy log stays attached to the lineage.
    node.branch_root_id = parent.branch_root_id or parent.id
    node.strategies = collect_strategies(response, node.code)
    node.summary = f"Debug of {parent.id}: {error_summary[:120]}"
    return node


def make_improve(
    *,
    journal: Journal,
    llm: LLMClient,
    cfg: SolverConfig,
    parent: SearchNode,
    budget: TimeBudget,
    data_preview: str = "",
    task_profile_summary: str = "",
    task_profile: TaskProfile | None = None,
    env_summary: str = "",
    task_type: str = "tabular",
) -> SearchNode:
    """Generate an improve node from a working parent.

    Constructs a per-branch strategy history table and a *required* strategy
    that the LLM must apply this iteration, threading them through the
    improve prompt builder. The required strategy is picked deterministically
    by ``solver.strategies.pick_required_strategy`` based on the current
    fraction of the time budget used and what's already been tried in this
    branch.
    """
    other_summary = summarise_other_attempts(
        journal.top_k_by_val(k=cfg.search.improve_top_k * 2),
        skip_id=parent.id,
    )

    # Branch history & exploration state.
    branch_root = parent.branch_root_id or parent.id
    history_rows = build_branch_history(journal, branch_root)
    used_strategies = cumulative_strategies(history_rows) | (parent.strategies or set())
    objective = task_profile.objective if task_profile is not None else ""
    metric_name = task_profile.metric_name if task_profile is not None else ""
    untried_list = untried_strategies(
        used_strategies,
        task_type=task_type,
        objective=objective,
        metric_name=metric_name,
    )
    required = pick_required_strategy(
        fraction_used=budget.fraction_used(),
        used=used_strategies,
        task_type=task_type,
        objective=objective,
        metric_name=metric_name,
    )

    messages = build_improve_prompt(
        parent=parent,
        journal_summary=other_summary,
        time_remaining=budget.remaining(),
        fraction_used=budget.fraction_used(),
        data_preview=data_preview,
        task_profile_summary=task_profile_summary,
        task_profile=task_profile,
        env_summary=env_summary,
        branch_history_rows=history_rows,
        used_strategies=used_strategies,
        untried_strategies=untried_list,
        required_strategy=required,
        task_type=task_type,
    )
    response = llm.chat(
        messages,
        temperature=cfg.llm.temperature,
        label=f"improve<-{parent.id}",
    )
    code = llm.extract_python_code(response)
    plan = llm.extract_first_paragraph(response)
    if not code:
        logger.warning(f"[improve from {parent.id}] LLM returned no code block")
        code = parent.code  # safe fallback

    node_id = journal.next_id("improve")
    node = SearchNode(
        id=node_id,
        stage="improve",
        code=code,
        plan=plan,
        parent_id=parent.id,
    )
    node.branch_root_id = parent.branch_root_id or parent.id
    node.strategies = collect_strategies(response, node.code)
    node.summary = code_summary(node.code) if node.code else "(empty improve)"
    if node.strategies:
        logger.info(
            f"[improve from {parent.id}] {node_id} strategies: {sorted(node.strategies)} "
            f"(required was {required!r})"
        )
    return node
