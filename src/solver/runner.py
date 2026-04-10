"""Main solve loop.

Three phases:

    1. **Drafts** — generate N diverse drafts, execute them in parallel.
    2. **Search** — pipelined debug + improve over the validated nodes,
       running ``max_parallel`` worker threads.
    3. **Finalize** — pick top-K single-model candidates by val score,
       post-process columns, return bytes.

Cross-cutting features:

    - Fail-fast startup validation (config, API key, data dir, sample submission).
    - Environment introspection (Python / OS / CPU / RAM / GPU) so the LLM
      knows what hardware it has.
    - Honest validation: every node prints VAL + HOLDOUT scores; relative gap
      check flags suspicious nodes.
    - Journal snapshot on disk after every step for debuggability.
    - Deterministic baseline fallback when every search step fails.
"""

from __future__ import annotations

import json
import logging
import os
import random
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from .config import SolverConfig
from .data_preview import build_data_preview
from .explore import run_exploration
from .interpreter import Interpreter
from .llm import LLMClient
from .nodes import Journal, SearchNode
from .operators import make_debug, make_draft, make_improve
from .parsing import holdout_gap_relative, parse_scores
from .postprocess import patch_submission_columns
from .progress import LoggingProgress, ProgressCallback
from .prompts import build_repair_prompt
from .selection import Selector
from .strategies import collect_strategies
from .task_classify import TaskProfile, load_kb_card, profile_task, render_task_profile
from .utils import (
    TimeBudget,
    detect_environment,
    ensure_dir,
    env_summary_for_prompt,
    find_sample_submission,
    fmt_seconds,
    list_data_files,
)

logger = logging.getLogger("solver")


_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "solver.yaml"
_KB_DIR = Path(__file__).resolve().parent / "kb"


# ── public entry point ────────────────────────────────────────────────────


def run_competition(
    work_dir: Path,
    progress: ProgressCallback | None = None,
) -> bytes | None:
    """Run the solver pipeline and return the primary submission candidate."""
    candidates = run_competition_candidates(work_dir, progress=progress)
    return candidates[0] if candidates else None


def run_competition_candidates(
    work_dir: Path,
    progress: ProgressCallback | None = None,
) -> list[bytes]:
    """Run the solver pipeline and return ranked candidate submissions."""
    work_dir = Path(work_dir)
    data_dir = work_dir / "home" / "data"
    workspace_dir = ensure_dir(work_dir / "solver_workspace")

    cfg = _load_config()
    cfg.workspace_dir = workspace_dir
    cfg.data_dir = data_dir

    _seed_everything(cfg.seed)
    _configure_logging(cfg)
    prog = progress or LoggingProgress()

    logger.info(f"[runner] starting solver: {cfg}")
    logger.info(f"[runner] data_dir={data_dir}")
    logger.info(f"[runner] workspace_dir={workspace_dir}")

    # ── Environment + startup validation ─────────────────────────────────
    env = detect_environment()
    logger.info(f"[runner] env={env}")
    env_summary = env_summary_for_prompt(env)

    cfg_errors = cfg.validate()
    setup_errors = _validate_setup(data_dir)
    fatal = cfg_errors + setup_errors
    if fatal:
        for e in fatal:
            logger.error(f"[runner] startup validation: {e}")
        # If we have a fatal config issue we can't even call the LLM. Best we
        # can do is return the sample submission directly so the grader still
        # gets *something*. The caller (agent.py) handles the None path.
        if any("api_key" in e or "model" in e for e in cfg_errors):
            logger.error("[runner] aborting before phase 1: cannot call LLM")
            fallback = _emergency_fallback(data_dir)
            return [fallback] if fallback is not None else []

    # ── Inputs ───────────────────────────────────────────────────────────
    description = _read_description(data_dir)
    task_profile = profile_task(description, data_dir)
    task_type = task_profile.category
    task_profile_summary = render_task_profile(task_profile)
    data_files = list_data_files(data_dir)
    sample_sub_preview = _sample_submission_preview(data_dir)
    kb_card = load_kb_card(task_type, _KB_DIR) if cfg.use_kb else ""
    data_preview = build_data_preview(data_dir)

    logger.info(
        f"[runner] task_type={task_type}; "
        f"objective={task_profile.objective}; "
        f"metric_hint={task_profile.metric_name or 'unknown'}; "
        f"description_chars={len(description)}; "
        f"data_files_visible={len(data_files)}; "
        f"first_files={data_files[:8]}; "
        f"kb_card={'yes' if kb_card else 'no'}; "
        f"data_preview_chars={len(data_preview)}"
    )

    # ── Dependencies ────────────────────────────────────────────────────
    llm = LLMClient(cfg.llm)
    interpreter = Interpreter(
        workspace_dir=workspace_dir,
        data_dir=data_dir,
        timeout=cfg.exec.timeout,
    )
    journal = Journal()
    selector = Selector(
        cfg,
        task_type=task_type,
        objective=task_profile.objective,
        metric_name=task_profile.metric_name,
    )
    budget = TimeBudget(total_seconds=cfg.time_limit, grace_seconds=cfg.search.grace_seconds)

    # Lock for selector + journal mutations across the worker pool.
    state_lock = threading.RLock()

    # ── Phase 0 — interactive data exploration ───────────────────────────
    # Run lightweight Python snippets against the actual data so the LLM
    # sees real file listings, column stats, and distributions — not just
    # the static heuristic preview.
    prog.on_phase("explore", "Running interactive data exploration")
    try:
        exploration_report = run_exploration(interpreter, timeout=60.0)
        if exploration_report:
            # Cap to avoid prompt bloat — the static preview already covers basics.
            max_explore = max(0, 4000 - len(data_preview))
            if max_explore > 500:
                trimmed = exploration_report[:max_explore]
                data_preview = data_preview + "\n\n" + trimmed
                logger.info(f"[runner] exploration report appended ({len(trimmed)} chars)")
            else:
                logger.info("[runner] skipping exploration injection — data_preview already large")
    except Exception as e:
        logger.warning(f"[runner] exploration failed (non-fatal): {e}")

    # ── Phase 1 — drafts in parallel ─────────────────────────────────────
    prog.on_phase("drafts", f"Generating {cfg.search.num_drafts} diverse drafts")
    _phase_drafts(
        cfg=cfg,
        llm=llm,
        interpreter=interpreter,
        journal=journal,
        budget=budget,
        task_desc=description,
        task_type=task_type,
        data_files=data_files,
        sample_sub_preview=sample_sub_preview,
        kb_card=kb_card,
        data_preview=data_preview,
        task_profile_summary=task_profile_summary,
        env_summary=env_summary,
        task_profile=task_profile,
    )
    _persist_journal(journal, workspace_dir)
    # Report draft results.
    stats = journal.stats()
    prog.on_phase("drafts", f"Drafts done: {stats['valid']} valid, {stats['buggy']} buggy")
    best = journal.best()
    if best and best.val_score is not None:
        prog.on_best(best.id, best.val_score, f"Best after drafts: {best.short()}")

    # ── Phase 2 — pipelined debug + improve ──────────────────────────────
    prog.on_phase("search", f"Starting search (max {cfg.search.max_steps} steps)")
    _phase_search_parallel(
        cfg=cfg,
        llm=llm,
        interpreter=interpreter,
        journal=journal,
        selector=selector,
        budget=budget,
        data_preview=data_preview,
        task_profile_summary=task_profile_summary,
        env_summary=env_summary,
        state_lock=state_lock,
        workspace_dir=workspace_dir,
        task_type=task_type,
        task_profile=task_profile,
        progress=prog,
    )
    _persist_journal(journal, workspace_dir)

    # ── Phase 3 — pick best candidates + post-process ────────────────────
    prog.on_phase("finalize", "Selecting top single-model candidates")
    return _phase_finalize_candidates(
        cfg=cfg,
        journal=journal,
        data_dir=data_dir,
        task_profile=task_profile,
    )


# ── phase 1: drafts ───────────────────────────────────────────────────────


def _phase_drafts(
    *,
    cfg: SolverConfig,
    llm: LLMClient,
    interpreter: Interpreter,
    journal: Journal,
    budget: TimeBudget,
    task_desc: str,
    task_type: str,
    data_files: list[str],
    sample_sub_preview: str,
    kb_card: str,
    data_preview: str,
    task_profile_summary: str,
    env_summary: str,
    task_profile: TaskProfile | None = None,
) -> None:
    n = cfg.search.num_drafts
    logger.info(f"[phase1] generating {n} drafts in parallel")

    # Generate all draft codes in parallel via threads (LLM calls overlap).
    pending: list[SearchNode] = []
    pending_lock = threading.Lock()

    def _gen_one(idx: int) -> None:
        if not budget.can_spawn_step():
            return
        try:
            node = make_draft(
                journal=journal,
                llm=llm,
                cfg=cfg,
                task_desc=task_desc,
                task_type=task_type,
                data_files=data_files,
                sample_sub_preview=sample_sub_preview,
                kb_card=kb_card,
                data_preview=data_preview,
                task_profile_summary=task_profile_summary,
                task_profile=task_profile,
                env_summary=env_summary,
                budget=budget,
                variant=idx,
                total_variants=n,
            )
            with pending_lock:
                pending.append(node)
            logger.info(f"[phase1] generated draft {idx+1}/{n}: {node.id}")
        except Exception as e:
            logger.exception(f"[phase1] draft {idx} generation failed: {e}")

    with ThreadPoolExecutor(max_workers=min(n, max(1, cfg.search.max_parallel))) as ex:
        list(ex.map(_gen_one, range(n)))

    if not pending:
        logger.error("[phase1] zero drafts generated")
        return

    # Execute drafts in parallel.
    _execute_in_parallel(
        cfg=cfg,
        interpreter=interpreter,
        llm=llm,
        journal=journal,
        nodes=pending,
        budget=budget,
        data_preview=data_preview,
        task_profile_summary=task_profile_summary,
        env_summary=env_summary,
        task_profile=task_profile,
    )


def _execute_in_parallel(
    *,
    cfg: SolverConfig,
    interpreter: Interpreter,
    llm: LLMClient,
    journal: Journal,
    nodes: list[SearchNode],
    budget: TimeBudget,
    data_preview: str = "",
    task_profile_summary: str = "",
    env_summary: str = "",
    task_profile: TaskProfile | None = None,
) -> None:
    if not nodes:
        return
    max_workers = min(cfg.search.max_parallel, len(nodes))
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(
                _execute_node,
                interpreter,
                journal,
                n,
                cfg=cfg,
                llm=llm,
                budget=budget,
                data_preview=data_preview,
                task_profile_summary=task_profile_summary,
                env_summary=env_summary,
                task_profile=task_profile,
            ): n
            for n in nodes
        }
        for fut in futures:
            try:
                fut.result()
            except Exception as e:
                logger.exception(f"[exec] parallel execution raised: {e}")


# ── phase 2: pipelined debug + improve loop ───────────────────────────────


def _phase_search_parallel(
    *,
    cfg: SolverConfig,
    llm: LLMClient,
    interpreter: Interpreter,
    journal: Journal,
    selector: Selector,
    budget: TimeBudget,
    data_preview: str,
    task_profile_summary: str,
    env_summary: str,
    state_lock: threading.RLock,
    workspace_dir: Path,
    task_type: str = "tabular",
    task_profile: TaskProfile | None = None,
    progress: ProgressCallback | None = None,
) -> None:
    """Run debug/improve operators in a pipelined worker pool.

    Up to ``max_parallel`` LLM-generation + execution pipelines run
    concurrently. Selection is serialised by ``state_lock`` so two workers
    never claim the same parent in the same instant.
    """
    max_steps = cfg.search.max_steps
    max_parallel = max(1, cfg.search.max_parallel)
    logger.info(
        f"[phase2] starting pipelined search: max_steps={max_steps} "
        f"max_parallel={max_parallel} remaining={fmt_seconds(budget.remaining())}"
    )

    in_flight: set[Future] = set()
    in_flight_parents: set[str] = set()  # parent_ids currently being acted on

    def _spawn_one(executor: ThreadPoolExecutor) -> bool:
        """Try to spawn one new operator step. Return True if spawned.

        Only call this when something has just completed (or at startup) —
        the selector has side effects on debug_attempts so we can't speculatively
        call it during idle waits.
        """
        with state_lock:
            # Count in-flight steps as already-claimed slots so we don't overshoot.
            if len(journal) + len(in_flight) >= max_steps:
                return False
            if not budget.can_spawn_step():
                return False
            # Selector skips parents currently in flight. No rollback dance.
            action = selector.pick(journal, excluded_ids=in_flight_parents)
            if action is None:
                return False
            in_flight_parents.add(action.parent.id)
            kind = action.kind
            parent = action.parent
            logger.info(
                f"[phase2] spawning {kind.upper()} of {parent.id} "
                f"(in_flight={len(in_flight)+1}/{max_parallel})"
            )

        fut = executor.submit(
            _step_worker,
            cfg=cfg,
            llm=llm,
            interpreter=interpreter,
            journal=journal,
            parent=parent,
            kind=kind,
            budget=budget,
            data_preview=data_preview,
            task_profile_summary=task_profile_summary,
            env_summary=env_summary,
            in_flight_parents=in_flight_parents,
            state_lock=state_lock,
            task_type=task_type,
            task_profile=task_profile,
        )
        in_flight.add(fut)
        return True

    with ThreadPoolExecutor(max_workers=max_parallel) as ex:
        # Initial fill.
        for _ in range(max_parallel):
            if not _spawn_one(ex):
                break

        # Drain + refill. We only act on real completions; idle waits are silent.
        while in_flight:
            done, _pending = wait(in_flight, return_when=FIRST_COMPLETED, timeout=30.0)
            if not done:
                # Idle timeout — nothing to do, no logs, no refills, no spam.
                continue
            for fut in done:
                in_flight.discard(fut)
                try:
                    fut.result()
                except Exception as e:
                    logger.exception(f"[phase2] worker raised: {e}")
            # Persist + log progress on each completion.
            with state_lock:
                _persist_journal(journal, workspace_dir)
                stats = journal.stats()
                best = journal.best()
                best_str = (
                    f"{best.val_score:.5f}" if best and best.val_score is not None else "N/A"
                )
                logger.info(
                    f"[phase2] stats={stats} best_val={best_str} "
                    f"in_flight={len(in_flight)} remaining={fmt_seconds(budget.remaining())}"
                )
                # Fire progress callbacks.
                if progress is not None:
                    step_num = stats["total"]
                    progress.on_step(
                        step_num, max_steps,
                        f"valid={stats['valid']} buggy={stats['buggy']} best={best_str} "
                        f"remaining={fmt_seconds(budget.remaining())}",
                    )
                    if best and best.val_score is not None:
                        progress.on_best(best.id, best.val_score, best.short())
            # Refill — keep workers fed.
            while len(in_flight) < max_parallel:
                if not _spawn_one(ex):
                    break

    logger.info(
        f"[phase2] loop complete. final stats={journal.stats()} "
        f"elapsed={fmt_seconds(budget.elapsed())}"
    )


def _step_worker(
    *,
    cfg: SolverConfig,
    llm: LLMClient,
    interpreter: Interpreter,
    journal: Journal,
    parent: SearchNode,
    kind: str,
    budget: TimeBudget,
    data_preview: str,
    task_profile_summary: str,
    env_summary: str,
    in_flight_parents: set[str],
    state_lock: threading.RLock,
    task_type: str = "tabular",
    task_profile: TaskProfile | None = None,
) -> None:
    """Run one debug/improve step from start (LLM call) to finish (journal append)."""
    try:
        if kind == "debug":
            with state_lock:
                prior = _collect_prior_debug_attempts(journal, parent)
            child = make_debug(
                journal=journal,
                llm=llm,
                cfg=cfg,
                parent=parent,
                budget=budget,
                data_preview=data_preview,
                task_profile_summary=task_profile_summary,
                task_profile=task_profile,
                env_summary=env_summary,
                prior_attempts=prior,
            )
        else:
            child = make_improve(
                journal=journal,
                llm=llm,
                cfg=cfg,
                parent=parent,
                budget=budget,
                data_preview=data_preview,
                task_profile_summary=task_profile_summary,
                task_profile=task_profile,
                env_summary=env_summary,
                task_type=task_type,
            )

        try:
            _execute_node(
                interpreter,
                journal,
                child,
                cfg=cfg,
                llm=llm,
                budget=budget,
                data_preview=data_preview,
                task_profile_summary=task_profile_summary,
                env_summary=env_summary,
                task_profile=task_profile,
            )
        except Exception as e:
            logger.exception(f"[phase2] execution raised on {child.id}: {e}")
            child.is_buggy = True
            with state_lock:
                journal.add(child)
    finally:
        with state_lock:
            in_flight_parents.discard(parent.id)


# ── phase 3: finalize ─────────────────────────────────────────────────────


def _phase_finalize(
    *,
    cfg: SolverConfig,
    journal: Journal,
    data_dir: Path,
) -> bytes | None:
    candidates = _phase_finalize_candidates(cfg=cfg, journal=journal, data_dir=data_dir)
    return candidates[0] if candidates else None


def _phase_finalize_candidates(
    *,
    cfg: SolverConfig,
    journal: Journal,
    data_dir: Path,
    task_profile: TaskProfile | None = None,
) -> list[bytes]:
    valid = journal.all_valid()
    logger.info(f"[phase3] valid nodes: {len(valid)}; total: {len(journal)}")

    if not valid:
        fallback = _last_resort_submission(journal, data_dir)
        return [fallback] if fallback is not None else []

    # Build the candidate list.
    top_k_nodes = journal.top_k_by_val(k=cfg.search.final_candidate_top_k)

    logger.info(
        f"[phase3] top-{len(top_k_nodes)} single-model candidates: "
        + ", ".join(
            f"{n.id}={n.val_score:.5f}{'(suspicious)' if n.is_suspicious else ''}"
            if n.val_score is not None
            else f"{n.id}=N/A"
            for n in top_k_nodes
        )
    )

    candidates: list[bytes] = []
    for best in top_k_nodes:
        logger.info(
            f"[phase3] adding single-model candidate: {best.id} (val={best.val_score})"
        )
        try:
            chosen = best.submission_path().read_bytes()  # type: ignore[union-attr]
        except Exception as e:
            logger.exception(f"[phase3] failed to read candidate {best.id}: {e}")
            continue
        try:
            chosen = patch_submission_columns(chosen, data_dir)
        except Exception as e:
            logger.exception(f"[phase3] column patching raised for {best.id}: {e}")
        candidates.append(chosen)

    deduped: list[bytes] = []
    seen: set[bytes] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        deduped.append(candidate)
    if deduped:
        return deduped

    fallback = _last_resort_submission(journal, data_dir)
    return [fallback] if fallback is not None else []


def _last_resort_submission(journal: Journal, data_dir: Path) -> bytes | None:
    """When everything failed, salvage anything we can."""
    # 1. Any node that produced a submission file at all (even if no parsed score).
    any_sub = [n for n in journal if n.has_submission]
    if any_sub:
        logger.warning(
            "[phase3] no valid scored nodes — falling back to any produced submission"
        )
        for n in any_sub:
            try:
                raw = n.submission_path().read_bytes()  # type: ignore[union-attr]
            except Exception:
                continue
            try:
                return patch_submission_columns(raw, data_dir)
            except Exception:
                return raw
    # 2. Sample submission as a last-ditch (trivially valid format).
    return _emergency_fallback(data_dir)


def _emergency_fallback(data_dir: Path) -> bytes | None:
    """Return ``sample_submission.csv`` bytes if available — never None unless the
    sample is also missing."""
    sample = find_sample_submission(data_dir)
    if sample is None:
        logger.error("[phase3] no sample submission found, returning None")
        return None
    try:
        logger.warning(
            f"[phase3] EMERGENCY FALLBACK: returning sample_submission.csv from {sample.name}"
        )
        return sample.read_bytes()
    except Exception as e:
        logger.exception(f"[phase3] failed to read sample submission: {e}")
        return None


# ── single node execution ─────────────────────────────────────────────────


def _execute_node(
    interpreter: Interpreter,
    journal: Journal,
    node: SearchNode,
    *,
    cfg: SolverConfig | None = None,
    llm: LLMClient | None = None,
    budget: TimeBudget | None = None,
    data_preview: str = "",
    task_profile_summary: str = "",
    env_summary: str = "",
    task_profile: TaskProfile | None = None,
) -> None:
    """Execute a node, parse scores, decide validity, append to journal."""
    if not node.code:
        logger.warning(f"[exec] node {node.id} has empty code, marking buggy")
        node.is_buggy = True
        journal.add(node)
        return

    parent_state_path: Path | None = None
    session_parent_node_id: str | None = None
    if node.parent_id is not None:
        parent = journal.parent_of(node)
        if (
            parent is not None
            and parent.result is not None
            and parent.result.is_success
            and not parent.is_buggy
            and parent.result.session_state_path is not None
            and parent.result.session_state_path.exists()
        ):
            parent_state_path = parent.result.session_state_path
            session_parent_node_id = parent.id

    result = _run_node_with_repairs(
        interpreter=interpreter,
        node=node,
        parent_state_path=parent_state_path,
        session_parent_node_id=session_parent_node_id,
        cfg=cfg,
        llm=llm,
        budget=budget,
        data_preview=data_preview,
        task_profile_summary=task_profile_summary,
        env_summary=env_summary,
        task_profile=task_profile,
    )

    node.result = result

    if not result.is_success:
        node.is_buggy = True
        logger.warning(
            f"[exec] node {node.id} BUGGY: rc={result.return_code} "
            f"timed_out={result.timed_out} error_summary={result.error_summary or '(none)'}"
        )
        tail = result.cleaned_log(max_chars=1500)
        if tail:
            logger.warning(f"[exec] node {node.id} log tail:\n{tail}")
        journal.add(node)
        return

    if not result.has_submission:
        logger.warning(
            f"[exec] node {node.id} ran clean but produced no submission.csv — buggy. "
            f"log tail:\n{result.cleaned_log(max_chars=1500)}"
        )
        node.is_buggy = True
        journal.add(node)
        return

    _patch_submission_in_place(node, interpreter)

    parsed = parse_scores(result.stdout + "\n" + result.stderr)
    if parsed.maximize is None and task_profile is not None and task_profile.maximize is not None:
        parsed.maximize = task_profile.maximize
    node.scores = parsed
    if (
        task_profile is not None
        and task_profile.maximize is not None
        and parsed.maximize is not None
        and parsed.maximize != task_profile.maximize
    ):
        _mark_suspicious(
            node,
            (
                f"metric direction {parsed.maximize} disagrees with task-profile hint "
                f"{task_profile.maximize}"
            ),
        )

    if parsed.val_score is None:
        logger.warning(
            f"[exec] node {node.id} produced submission but no parseable VAL score — buggy. "
            f"log tail:\n{result.cleaned_log(max_chars=1500)}"
        )
        node.is_buggy = True
        journal.add(node)
        return

    # Validate the submission shape against sample_submission, if available.
    if not _submission_shape_ok(node, interpreter):
        _mark_suspicious(node, "submission shape does not match sample_submission")

    # ── Detect FAKE successes ────────────────────────────────────────────
    # The most common cause of "valid score but useless submission" is that
    # the LLM wrapped its training in a try/except that, on failure, writes a
    # constant prediction (e.g. all True for binary classification) and prints
    # a placeholder val=0.5 / holdout=0.5. We aggressively reject these so
    # they don't pollute the top-K candidate list.
    fake_reason = _detect_fake_success(node, parsed)
    if fake_reason is not None:
        logger.warning(
            f"[exec] node {node.id} FAKE success rejected: {fake_reason}. "
            f"Marking as buggy so it is not selected as a final candidate."
        )
        node.is_buggy = True
        # Stuff the reason into the result.error_summary so the debug operator
        # sees it on the next attempt.
        if node.result is not None and not node.result.error_summary:
            node.result.error_summary = f"FakeSuccess: {fake_reason}"
        journal.add(node)
        return

    for reason in _audit_score_outputs(node, parsed, cfg):
        _mark_suspicious(node, reason)
    for reason in _audit_submission_predictions(node, interpreter, task_profile):
        _mark_suspicious(node, reason)
    for reason in _audit_static_code_route(node, task_profile):
        _mark_suspicious(node, reason)
    for reason in _audit_branch_overfit(journal, node, cfg):
        _mark_suspicious(node, reason)

    # Honesty check — only when BOTH scores are present and non-zero.
    # If holdout was not printed (None or 0), don't penalise the node.
    if (
        parsed.val_score is not None
        and parsed.holdout_score is not None
        and abs(parsed.holdout_score) > 1e-6
    ):
        rel_gap = holdout_gap_relative(parsed)
        if rel_gap is not None:
            threshold = (
                cfg.search.holdout_gap_rel_threshold
                if cfg is not None
                else 0.20
            )
            if rel_gap > threshold:
                _mark_suspicious(
                    node,
                    f"large relative val/holdout gap ({rel_gap:.3f} > {threshold:.2f})",
                )
    elif parsed.holdout_score is None or abs(parsed.holdout_score or 0) < 1e-6:
        logger.info(
            f"[exec] node {node.id} holdout score missing or zero — skipping gap check"
        )

    journal.add(node)


def _run_node_with_repairs(
    *,
    interpreter: Interpreter,
    node: SearchNode,
    parent_state_path: Path | None,
    session_parent_node_id: str | None,
    cfg: SolverConfig | None,
    llm: LLMClient | None,
    budget: TimeBudget | None,
    data_preview: str,
    task_profile_summary: str,
    env_summary: str,
    task_profile: TaskProfile | None,
):
    if cfg is None or llm is None:
        return _run_node_once(
            interpreter=interpreter,
            node=node,
            parent_state_path=parent_state_path,
            session_parent_node_id=session_parent_node_id,
        )

    max_repairs = max(0, cfg.search.inline_repair_attempts)
    for attempt in range(max_repairs + 1):
        result = _run_node_once(
            interpreter=interpreter,
            node=node,
            parent_state_path=parent_state_path,
            session_parent_node_id=session_parent_node_id,
        )
        repair_reason = _repair_reason_for_result(
            node=node,
            result=result,
            interpreter=interpreter,
            task_profile=task_profile,
        )
        if repair_reason is None or attempt >= max_repairs:
            return result
        if budget is not None and not budget.can_spawn_step():
            logger.info(f"[exec] skipping inline repair for {node.id}: time budget too low")
            return result

        messages = build_repair_prompt(
            code=node.code,
            error_summary=repair_reason,
            cleaned_log=result.cleaned_log(max_chars=6000),
            time_remaining=budget.remaining() if budget is not None else 0.0,
            repair_attempt=attempt + 1,
            total_repairs=max_repairs,
            data_preview=data_preview,
            task_profile_summary=task_profile_summary,
            task_profile=task_profile,
            env_summary=env_summary,
        )
        response = llm.chat(
            messages,
            temperature=max(0.1, cfg.llm.temperature * 0.5),
            label=f"repair<-{node.id}/attempt{attempt+1}",
        )
        repaired_code = llm.extract_python_code(response)
        if not repaired_code:
            logger.warning(f"[exec] inline repair returned no code for {node.id}")
            return result
        if repaired_code.strip() == node.code.strip():
            logger.info(f"[exec] inline repair for {node.id} returned unchanged code")
            return result

        node.code = repaired_code
        node.plan = llm.extract_first_paragraph(response) or node.plan
        node.strategies = set(node.strategies or set()) | collect_strategies(response, node.code)
        node.summary = f"Inline repair of {node.id}: {repair_reason[:120]}"
        logger.info(
            f"[exec] inline repair {attempt+1}/{max_repairs} prepared for {node.id}: "
            f"{repair_reason}"
        )

    return _run_node_once(
        interpreter=interpreter,
        node=node,
        parent_state_path=parent_state_path,
        session_parent_node_id=session_parent_node_id,
    )


def _run_node_once(
    *,
    interpreter: Interpreter,
    node: SearchNode,
    parent_state_path: Path | None,
    session_parent_node_id: str | None,
):
    try:
        return interpreter.run(
            node.code,
            node.id,
            parent_state_path=parent_state_path,
            session_parent_node_id=session_parent_node_id,
        )
    except Exception as e:
        logger.exception(f"[exec] interpreter raised on {node.id}: {e}")
        from .interpreter import ExecResult

        return ExecResult(
            return_code=-1,
            stdout="",
            stderr=f"InterpreterError: {type(e).__name__}: {e}",
            duration_seconds=0.0,
            timed_out=False,
            submission_path=None,
            session_state_path=None,
            error_summary=f"InterpreterError: {type(e).__name__}: {e}",
        )


def _repair_reason_for_result(
    *,
    node: SearchNode,
    result,
    interpreter: Interpreter,
    task_profile: TaskProfile | None,
) -> str | None:
    if not result.is_success:
        return result.error_summary or "ExecutionFailed: solution.py did not run successfully"
    if not result.has_submission:
        return "MissingSubmission: script ran but did not create submission.csv"

    node.result = result
    _patch_submission_in_place(node, interpreter)
    if not _submission_shape_ok(node, interpreter):
        return (
            "SubmissionSchemaMismatch: submission.csv still does not match "
            "sample_submission.csv after patching"
        )

    parsed = parse_scores(result.stdout + "\n" + result.stderr)
    if parsed.maximize is None and task_profile is not None and task_profile.maximize is not None:
        parsed.maximize = task_profile.maximize
    if parsed.val_score is None:
        return "MissingValScore: submission was produced but FINAL VAL SCORE was missing or unparsable"
    return None


def _patch_submission_in_place(node: SearchNode, interpreter: Interpreter) -> None:
    sub_path = node.submission_path()
    if sub_path is None or not sub_path.exists():
        return
    try:
        original = sub_path.read_bytes()
        patched = patch_submission_columns(original, interpreter.data_dir)
        if patched != original:
            sub_path.write_bytes(patched)
            logger.info(f"[exec] node {node.id} submission patched against sample_submission")
    except Exception as e:
        logger.debug(f"[exec] submission patching skipped for {node.id}: {e}")


def _detect_fake_success(node: SearchNode, parsed) -> str | None:
    """Heuristic check for 'the script crashed and printed a fallback score'.

    Returns a human-readable reason string if the node looks fake, or None
    if it looks like a real solution.
    """
    sub_path = node.submission_path()
    if sub_path is None:
        return "no submission path"

    val = parsed.val_score
    hold = parsed.holdout_score

    # Signal 1: predictions in the submission are all the same value.
    try:
        import pandas as pd
        df = pd.read_csv(sub_path)
        if df.empty or len(df) < 2:
            return None  # not enough rows to judge
        # Heuristic: any numeric/bool column whose unique values == 1 is constant.
        constant_cols: list[str] = []
        for col in df.columns:
            try:
                nu = df[col].nunique(dropna=False)
                if nu <= 1:
                    constant_cols.append(col)
            except Exception:
                continue
        if constant_cols:
            non_id_constants = [
                c for c in constant_cols
                if not (c.lower().endswith("id") or c.lower() in {"id", "index"})
            ]
            if non_id_constants:
                return (
                    f"submission has constant prediction column(s) "
                    f"{non_id_constants[:3]} (all rows identical)"
                )

        # Signal 1b: a numeric prediction column where ≥98% of rows are the
        # same value is also a "near-constant" predictor.
        for col in df.columns:
            if col.lower() in {"id", "index"} or col.lower().endswith("id"):
                continue
            try:
                vc = df[col].value_counts(dropna=False, normalize=True)
                if not vc.empty and float(vc.iloc[0]) >= 0.98:
                    return (
                        f"submission column '{col}' is {float(vc.iloc[0]):.0%} "
                        f"the single value {vc.index[0]!r}"
                    )
            except Exception:
                continue
    except Exception as e:
        logger.debug(f"[exec] fake-success constant check failed: {e}")

    # Signal 2: val score == holdout score == known placeholder value.
    if val is not None and hold is not None:
        gap = abs(val - hold)
        if gap < 1e-6:
            for placeholder in (0.5, 0.0, 1.0, 0.25, 0.33, 0.333, 0.3333):
                if abs(val - placeholder) < 1e-3:
                    return (
                        f"val == holdout == {placeholder} (looks like a placeholder "
                        f"score from an except branch)"
                    )

    # Signal 3: the submission is byte-identical to the sample_submission file.
    # That can only happen if training silently failed and the script wrote
    # the sample back unchanged.
    try:
        from .utils import find_sample_submission
        node_input_dir = node.result.submission_path.parent / "input"  # type: ignore[union-attr]
        sample_path = find_sample_submission(node_input_dir)
        if sample_path is not None and sample_path.exists():
            if sub_path.read_bytes() == sample_path.read_bytes():
                return f"submission is byte-identical to sample_submission ({sample_path.name})"
    except Exception as e:
        logger.debug(f"[exec] fake-success sample-equality check failed: {e}")

    return None


def _submission_shape_ok(node: SearchNode, interpreter: Interpreter) -> bool:
    """Return True if the node's submission has compatible shape with the sample."""
    sub_path = node.submission_path()
    if sub_path is None:
        return False
    sample = find_sample_submission(interpreter.data_dir)
    if sample is None:
        return True  # nothing to compare against
    try:
        import pandas as pd
        sub = pd.read_csv(sub_path, nrows=1)
        sample_df = pd.read_csv(sample, nrows=1)
        # Same columns?
        if list(sub.columns) != list(sample_df.columns):
            logger.info(
                f"[exec] node {node.id} columns differ from sample: "
                f"sub={list(sub.columns)} sample={list(sample_df.columns)}"
            )
            return False
        # Same row count? (Cheap-ish via line count.)
        sub_rows = _quick_row_count(sub_path)
        sample_rows = _quick_row_count(sample)
        if sub_rows >= 0 and sample_rows >= 0 and sub_rows != sample_rows:
            logger.info(
                f"[exec] node {node.id} row count {sub_rows} != sample {sample_rows}"
            )
            return False
    except Exception as e:
        logger.debug(f"[exec] shape check failed: {e}")
        return True  # don't penalise nodes for our own check failure
    return True


def _quick_row_count(path: Path) -> int:
    try:
        with path.open("rb") as fh:
            return sum(1 for _ in fh) - 1
    except Exception:
        return -1


def _mark_suspicious(node: SearchNode, reason: str) -> None:
    node.is_suspicious = True
    if reason not in node.suspicion_reasons:
        node.suspicion_reasons.append(reason)
        logger.warning(f"[exec] node {node.id} suspicious: {reason}")


def _audit_score_outputs(
    node: SearchNode,
    parsed,
    cfg: SolverConfig | None,
) -> list[str]:
    reasons: list[str] = []
    identical_tol = cfg.search.identical_score_abs_tol if cfg is not None else 1e-6
    perfect_tol = cfg.search.perfect_score_abs_tol if cfg is not None else 1e-6
    variance_threshold = cfg.search.fold_variance_rel_threshold if cfg is not None else 0.20

    if parsed.maximize is not None:
        for label, score in (("validation", parsed.val_score), ("holdout", parsed.holdout_score)):
            if score is None:
                continue
            if parsed.maximize and score >= 1.0 - perfect_tol:
                reasons.append(f"{label} score is nearly perfect ({score:.6f}) and needs extra scrutiny")
            if not parsed.maximize and score <= perfect_tol:
                reasons.append(f"{label} score is nearly perfect ({score:.6f}) and needs extra scrutiny")

    if parsed.val_score is not None and parsed.holdout_score is not None:
        if abs(parsed.val_score - parsed.holdout_score) <= identical_tol:
            reasons.append("validation and holdout scores are numerically identical; validation may have been reused")

    if np is not None and len(parsed.fold_scores) >= 3:
        fold_scores = np.asarray(parsed.fold_scores, dtype=float)
        std = float(np.std(fold_scores))
        mean = float(np.mean(fold_scores))
        rel_std = std / max(abs(mean), 0.01)
        if rel_std > variance_threshold:
            reasons.append(
                f"CV fold scores are unstable (std={std:.4f}, rel_std={rel_std:.2f} > {variance_threshold:.2f})"
            )
    return reasons


def _audit_submission_predictions(
    node: SearchNode,
    interpreter: Interpreter,
    task_profile: TaskProfile | None,
) -> list[str]:
    if task_profile is None:
        return []
    sub_path = node.submission_path()
    if sub_path is None:
        return []
    sample = find_sample_submission(interpreter.data_dir)
    if sample is None:
        return []
    try:
        import pandas as pd

        sub_df = pd.read_csv(sub_path)
        sample_df = pd.read_csv(sample, nrows=1)
    except Exception:
        return []

    target_cols = list(task_profile.sample_target_cols) or [
        col for col in sample_df.columns
        if col.lower() not in {"id", "index"} and not col.lower().endswith("id")
    ]
    if not target_cols or any(col not in sub_df.columns for col in target_cols):
        return []

    reasons: list[str] = []
    mode = task_profile.prediction_mode
    try:
        target_values = sub_df[target_cols].astype(float).to_numpy() if np is not None else None
    except Exception:
        target_values = None

    if np is not None and target_values is not None and mode == "multiclass_probabilities" and target_values.size:
        if np.any(target_values < -1e-6) or np.any(target_values > 1.0 + 1e-6):
            reasons.append("multiclass probability submission contains values outside [0, 1]")
        row_sums = target_values.sum(axis=1)
        if row_sums.size and (
            float(np.mean(np.abs(row_sums - 1.0))) > 1e-3 or float(np.max(np.abs(row_sums - 1.0))) > 0.05
        ):
            reasons.append("multiclass probability rows do not sum to 1")

    if np is not None and target_values is not None and (
        mode == "multilabel_probabilities"
        or (mode == "single_probability" and task_profile.metric_name == "logloss")
    ) and target_values.size:
        if np.any(target_values < -1e-6) or np.any(target_values > 1.0 + 1e-6):
            reasons.append("probability submission contains values outside [0, 1]")

    return reasons


def _audit_static_code_route(
    node: SearchNode,
    task_profile: TaskProfile | None,
) -> list[str]:
    if task_profile is None or not node.code:
        return []

    code = node.code
    code_lower = code.lower()
    reasons: list[str] = []

    if task_profile.validation_mode == "time_ordered":
        if "timeseriessplit" not in code_lower and any(
            token in code for token in ("KFold(", "StratifiedKFold(", "train_test_split(")
        ):
            reasons.append(
                "time-ordered task uses IID/random split primitives instead of TimeSeriesSplit or a manual chronological split"
            )
        if "shuffle=true" in code_lower or "shuffle = true" in code_lower:
            reasons.append("time-ordered task enables shuffle=True")
        if task_profile.likely_time_cols and "sort_values" not in code_lower and "sort_index" not in code_lower:
            reasons.append("time-ordered task does not appear to sort by time before splitting")

    if task_profile.validation_mode in {"grouped", "stratified_group"}:
        has_group_split = any(
            token in code for token in ("GroupKFold", "StratifiedGroupKFold", "GroupShuffleSplit")
        )
        uses_iid_split = any(
            token in code for token in ("KFold(", "StratifiedKFold(", "train_test_split(")
        )
        if uses_iid_split and not has_group_split:
            reasons.append("group-leakage task uses IID split primitives instead of a group-aware splitter")

    if task_profile.prediction_mode == "multiclass_probabilities":
        if "predict_proba" not in code_lower and "softmax" not in code_lower:
            reasons.append("multiclass probability route does not clearly produce class probabilities")

    if task_profile.prediction_mode == "multilabel_probabilities":
        if "softmax" in code_lower and "sigmoid" not in code_lower:
            reasons.append("multilabel route appears to use softmax instead of independent sigmoid probabilities")

    return reasons


def _audit_branch_overfit(
    journal: Journal,
    node: SearchNode,
    cfg: SolverConfig | None,
) -> list[str]:
    if node.val_score is None or node.holdout_score is None:
        return []
    branch_root = getattr(node, "branch_root_id", None) or node.id
    prior_nodes = [
        prior for prior in journal
        if (getattr(prior, "branch_root_id", None) or prior.id) == branch_root
        and not prior.is_buggy
        and prior.val_score is not None
        and prior.holdout_score is not None
    ]
    if not prior_nodes:
        return []
    previous = sorted(prior_nodes, key=lambda item: item.created_at)[-1]
    maximize = (
        node.scores.maximize
        if node.scores.maximize is not None
        else previous.scores.maximize
        if previous.scores.maximize is not None
        else True
    )
    eps = cfg.search.metric_improve_eps if cfg is not None else 1e-4
    val_delta = (node.val_score - previous.val_score) if maximize else (previous.val_score - node.val_score)
    holdout_delta = (
        (node.holdout_score - previous.holdout_score)
        if maximize
        else (previous.holdout_score - node.holdout_score)
    )
    if val_delta > eps and holdout_delta <= eps:
        status = "degraded" if holdout_delta < -eps else "stayed flat"
        return [
            f"validation improved by {val_delta:.5f} but holdout {status} versus branch node {previous.id}"
        ]
    return []


# ── helpers ───────────────────────────────────────────────────────────────


def _collect_prior_debug_attempts(journal: Journal, parent: SearchNode) -> list[SearchNode]:
    """All previous debug-stage children of ``parent`` (failed attempts).

    Used by the debug operator to avoid re-proposing the same wrong fix.
    """
    out: list[SearchNode] = []
    for n in journal:
        if n.parent_id == parent.id and n.stage == "debug":
            out.append(n)
    return out


def _persist_journal(journal: Journal, workspace_dir: Path) -> None:
    """Write a json snapshot of the journal for offline debugging."""
    try:
        snapshot_path = workspace_dir / "journal.json"
        rows = []
        for n in journal:
            rows.append(
                {
                    "id": n.id,
                    "stage": n.stage,
                    "parent_id": n.parent_id,
                    "branch_root_id": getattr(n, "branch_root_id", None),
                    "val_score": n.val_score,
                    "holdout_score": n.holdout_score,
                    "fold_scores": list(getattr(n.scores, "fold_scores", ()) or ()),
                    "is_buggy": n.is_buggy,
                    "is_suspicious": n.is_suspicious,
                    "suspicion_reasons": list(n.suspicion_reasons),
                    "debug_attempts": n.debug_attempts,
                    "created_at": n.created_at,
                    "summary": n.summary,
                    "strategies": sorted(getattr(n, "strategies", set()) or []),
                    "error": (n.result.error_summary if n.result else "") or "",
                    "submission": (
                        str(n.result.submission_path)
                        if n.result and n.result.submission_path
                        else None
                    ),
                    "session_state": (
                        str(n.result.session_state_path)
                        if n.result and n.result.session_state_path
                        else None
                    ),
                }
            )
        snapshot_path.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
    except Exception as e:
        logger.debug(f"[runner] journal persist failed: {e}")


def _validate_setup(data_dir: Path) -> list[str]:
    errors: list[str] = []
    if not data_dir.exists():
        errors.append(f"data_dir does not exist: {data_dir}")
        return errors
    if not any(data_dir.iterdir()):
        errors.append(f"data_dir is empty: {data_dir}")
    sample = find_sample_submission(data_dir)
    if sample is None:
        # Not necessarily fatal — some competitions emit other naming.
        logger.warning(f"[runner] no sample_submission file found under {data_dir}")
    return errors


def _load_config() -> SolverConfig:
    cfg = SolverConfig.from_yaml(_CONFIG_PATH)
    cfg.resolve_env()
    return cfg


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    if np is not None:
        try:
            np.random.seed(seed)
        except Exception:
            pass
    os.environ.setdefault("PYTHONHASHSEED", str(seed))


def _configure_logging(cfg: SolverConfig) -> None:
    level = getattr(logging, cfg.log_level.upper(), logging.INFO)
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] %(message)s")
        )
        logger.addHandler(handler)
    logger.propagate = True


def _read_description(data_dir: Path) -> str:
    desc = data_dir / "description.md"
    if desc.exists():
        try:
            return desc.read_text(encoding="utf-8", errors="replace")
        except Exception:
            pass
    return "Solve the machine learning competition in ./input/."


def _sample_submission_preview(data_dir: Path, max_chars: int = 800) -> str:
    sample = find_sample_submission(data_dir)
    if sample is None:
        return ""
    try:
        text = sample.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""
    head = "\n".join(text.splitlines()[:6])
    if len(head) > max_chars:
        head = head[:max_chars] + "..."
    return f"# {sample.name}\n{head}"
