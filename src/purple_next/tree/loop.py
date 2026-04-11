"""One isolated tree-search run — drafts → pipelined improve/debug → finalize.

The loop wires together agents, interpreter, selector, and journal. It
does not know about panel/pass@K — that's the layer above.
"""

from __future__ import annotations

import json
import logging
import random
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path

from ..agents import (
    ParsedOutcome,
    ReviewVerdict,
    generate_debug_code,
    generate_draft_code,
    generate_improve_code,
    parse_outcome,
    review_candidate,
)
from ..config import Config
from ..exec import Interpreter, detect_fake_success
from ..llm import LLMClient
from ..prompts import pick_hint
from .journal import Journal
from .node import SearchNode
from .ranking import adjusted_review_penalty, hard_leakage_flag
from .selector import Selector

logger = logging.getLogger("purple_next")


@dataclass
class RunContext:
    task_desc: str
    data_files: list[str]
    data_preview: str
    env_summary: str
    contract_summary: str
    maximize: bool
    direction_label: str
    sample_submission_path: Path | None
    disposition: str
    variant_temperatures: list[float] = field(default_factory=list)


@dataclass
class RunReport:
    journal: Journal
    candidates: list[SearchNode]
    best: SearchNode | None
    elapsed_s: float


class TreeLoop:
    def __init__(
        self,
        *,
        cfg: Config,
        llm: LLMClient,
        interpreter: Interpreter,
        workspace_dir: Path,
        run_context: RunContext,
    ):
        self.cfg = cfg
        self.llm = llm
        self.interpreter = interpreter
        self.workspace_dir = workspace_dir
        self.ctx = run_context
        self.journal = Journal()
        self.selector = Selector(
            max_debug_attempts_per_node=cfg.search.max_debug_attempts_per_node,
            explore_c=cfg.search.ucb_explore_c,
        )
        self._lock = threading.RLock()
        self._in_flight_parents: set[str] = set()
        self._started_at = time.time()
        self._rng = random.Random(cfg.seed)

    # ── public entry ─────────────────────────────────────────────────────

    def run(self) -> RunReport:
        self._phase_drafts()
        self._persist()
        self._phase_search()
        self._persist()
        candidates = self._phase_finalize()
        self._persist()
        best = candidates[0] if candidates else None
        return RunReport(
            journal=self.journal,
            candidates=candidates,
            best=best,
            elapsed_s=time.time() - self._started_at,
        )

    # ── phase 1: drafts ─────────────────────────────────────────────────

    def _phase_drafts(self) -> None:
        n = self.cfg.search.num_drafts
        logger.info(f"[phase1] generating {n} drafts in parallel")

        def _gen(idx: int) -> SearchNode | None:
            if self._budget_exhausted():
                return None
            try:
                temp = self._draft_temperature(idx)
                code = generate_draft_code(
                    llm=self.llm,
                    task_desc=self.ctx.task_desc,
                    data_files=self.ctx.data_files,
                    data_preview=self.ctx.data_preview,
                    contract_summary=self.ctx.contract_summary,
                    env_summary=self.ctx.env_summary,
                    time_remaining_s=self._remaining(),
                    disposition=self.ctx.disposition,
                    variant=idx,
                    temperature=temp,
                    label=f"draft_v{idx}",
                )
            except Exception as e:
                logger.exception(f"[phase1] draft {idx} generation failed: {e}")
                return None
            node_id = self.journal.next_id("draft")
            node = SearchNode(
                id=node_id,
                stage="draft",
                code=code or "",
                parent_id=None,
                branch_root_id=node_id,
            )
            return node

        with ThreadPoolExecutor(max_workers=max(1, self.cfg.search.max_parallel)) as ex:
            drafts = [d for d in ex.map(_gen, range(n)) if d is not None]

        self._execute_many(drafts)

    def _draft_temperature(self, variant: int) -> float | None:
        if variant < len(self.ctx.variant_temperatures):
            return max(0.1, min(1.5, self.ctx.variant_temperatures[variant]))
        return self.cfg.llm.temperature

    # ── phase 2: pipelined search ───────────────────────────────────────

    def _phase_search(self) -> None:
        max_steps = self.cfg.search.max_steps
        max_parallel = max(1, self.cfg.search.max_parallel)

        in_flight: set[Future] = set()

        def _try_spawn(executor: ThreadPoolExecutor) -> bool:
            with self._lock:
                if len(self.journal) + len(in_flight) >= max_steps:
                    return False
                if self._budget_exhausted():
                    return False
                action = self.selector.pick(
                    self.journal,
                    excluded=set(self._in_flight_parents),
                    maximize=self.ctx.maximize,
                )
                if action is None:
                    return False
                self._in_flight_parents.add(action.parent.id)
            fut = executor.submit(self._step_worker, action.kind, action.parent)
            in_flight.add(fut)
            return True

        with ThreadPoolExecutor(max_workers=max_parallel) as ex:
            for _ in range(max_parallel):
                if not _try_spawn(ex):
                    break
            while in_flight:
                done, _ = wait(in_flight, return_when=FIRST_COMPLETED, timeout=60.0)
                if not done:
                    continue
                for fut in done:
                    in_flight.discard(fut)
                    try:
                        fut.result()
                    except Exception as e:
                        logger.exception(f"[phase2] worker raised: {e}")
                self._persist()
                while len(in_flight) < max_parallel:
                    if not _try_spawn(ex):
                        break

    def _step_worker(self, kind: str, parent: SearchNode) -> None:
        try:
            if kind == "debug":
                child = self._make_debug(parent)
            else:
                child = self._make_improve(parent)
            if child is not None:
                self._execute_and_record(child)
        finally:
            with self._lock:
                self._in_flight_parents.discard(parent.id)

    def _make_improve(self, parent: SearchNode) -> SearchNode | None:
        if self._budget_exhausted():
            return None
        stdout_tail = parent.result.tail(max_chars=2500) if parent.result else ""
        hint_index = pick_hint(self.journal, rng=self._rng)
        code = generate_improve_code(
            llm=self.llm,
            parent_code=parent.code,
            parent_cv=parent.cv_score,
            parent_holdout=parent.holdout_score,
            parent_stdout_tail=stdout_tail,
            direction=self.ctx.direction_label,
            hint_index=hint_index,
            contract_summary=self.ctx.contract_summary,
            time_remaining_s=self._remaining(),
            fraction_used=self._fraction_used(),
            temperature=self.cfg.llm.temperature,
            label=f"improve<-{parent.id}",
        )
        node_id = self.journal.next_id("improve")
        node = SearchNode(
            id=node_id,
            stage="improve",
            code=code,
            parent_id=parent.id,
            branch_root_id=parent.branch_root_id or parent.id,
            improve_hint_index=hint_index,
        )
        return node

    def _make_debug(self, parent: SearchNode) -> SearchNode | None:
        if self._budget_exhausted():
            return None
        log_tail = parent.result.tail(max_chars=3000) if parent.result else ""
        code = generate_debug_code(
            llm=self.llm,
            parent_code=parent.code,
            error_summary=parent.result.error_summary if parent.result else "",
            log_tail=log_tail,
            contract_summary=self.ctx.contract_summary,
            time_remaining_s=self._remaining(),
            temperature=self.cfg.llm.temperature,
            label=f"debug<-{parent.id}",
        )
        node_id = self.journal.next_id("debug")
        node = SearchNode(
            id=node_id,
            stage="debug",
            code=code,
            parent_id=parent.id,
            branch_root_id=parent.branch_root_id or parent.id,
        )
        return node

    # ── execution + parsing ─────────────────────────────────────────────

    def _execute_many(self, nodes: list[SearchNode]) -> None:
        if not nodes:
            return
        workers = min(self.cfg.search.max_parallel, len(nodes))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(self._execute_and_record, nodes))

    def _execute_and_record(self, node: SearchNode) -> None:
        if not node.code:
            node.is_buggy = True
            node.notes = "empty code"
            self.journal.add(node)
            return

        try:
            result = self.interpreter.run(node.code, node.id)
        except Exception as e:
            logger.exception(f"[exec] interpreter raised on {node.id}: {e}")
            from ..exec.interpreter import ExecResult
            result = ExecResult(
                return_code=-1,
                stdout="",
                stderr=f"InterpreterError: {type(e).__name__}: {e}",
                duration_seconds=0.0,
                error_summary=f"InterpreterError: {type(e).__name__}: {e}",
            )
        node.result = result

        if not result.is_success:
            node.is_buggy = True
            self.journal.add(node)
            return

        if not result.has_submission:
            node.is_buggy = True
            node.notes = "no submission.csv produced"
            self.journal.add(node)
            return

        # Fake-success detector runs first — cheap and catches the obvious.
        fake = detect_fake_success(result.submission_path, self.ctx.sample_submission_path)
        if fake:
            node.is_buggy = True
            node.notes = f"fake-success: {fake}"
            self.journal.add(node)
            return

        # LLM parser reads the stdout + code and returns cv/holdout/bug.
        parsed: ParsedOutcome = parse_outcome(
            llm=self.llm,
            code=node.code,
            stdout=result.stdout,
            stderr=result.stderr,
            maximize=self.ctx.maximize,
        )
        node.cv_score = parsed.cv_score
        node.holdout_score = parsed.holdout_score
        node.maximize = self.ctx.maximize
        node.notes = parsed.notes

        if parsed.bug or parsed.cv_score is None or parsed.holdout_score is None:
            node.is_buggy = True
            if not node.notes:
                node.notes = "parser flagged as missing scores"
            self.journal.add(node)
            return

        self.journal.add(node)

    # ── phase 3: finalize ───────────────────────────────────────────────

    def _phase_finalize(self) -> list[SearchNode]:
        valid = self.journal.all_valid()
        if not valid:
            return []

        maximize = self.ctx.maximize

        def score_key(v: float | None) -> float:
            if v is None:
                return float("-inf")
            return v if maximize else -v

        # Initial sort by holdout, then cv, then recency.
        ranked = sorted(
            valid,
            key=lambda n: (score_key(n.holdout_score), score_key(n.cv_score), n.created_at),
            reverse=True,
        )
        top_k = ranked[: self.cfg.search.final_top_k]
        top_scored_id = top_k[0].id if top_k else ""

        # Reviewer agent runs once per top-K candidate. Verdict feeds into
        # the final rerank: clean > suspicious > leaky.
        for node in top_k:
            if node.review_verdict:
                continue
            verdict: ReviewVerdict = review_candidate(
                llm=self.llm,
                code=node.code,
                task_desc=self.ctx.task_desc,
                contract_summary=self.ctx.contract_summary,
                cv_score=node.cv_score,
                holdout_score=node.holdout_score,
                label=f"review<-{node.id}",
            )
            if (
                node.id == top_scored_id
                and verdict.verdict == "suspicious"
                and verdict.confidence in {"medium", "high"}
            ):
                first_verdict = verdict.verdict
                first_confidence = verdict.confidence
                second = review_candidate(
                    llm=self.llm,
                    code=node.code,
                    task_desc=self.ctx.task_desc,
                    contract_summary=self.ctx.contract_summary,
                    cv_score=node.cv_score,
                    holdout_score=node.holdout_score,
                    label=f"review2<-{node.id}",
                    temperature=0.0,
                )
                agreement = second.verdict in {"suspicious", "leaky"}
                if agreement:
                    merged_verdict = (
                        "leaky" if "leaky" in {verdict.verdict, second.verdict} else "suspicious"
                    )
                    merged_conf = _max_confidence(verdict.confidence, second.confidence)
                    merged_reasons = _merge_unique(verdict.reasons + second.reasons)
                    verdict = ReviewVerdict(
                        verdict=merged_verdict,
                        confidence=merged_conf,
                        reasons=merged_reasons,
                        summary=verdict.summary or second.summary,
                    )
                    logger.info(
                        f"[review] {node.id} consensus=agree "
                        f"first={first_verdict}/{first_confidence} "
                        f"second={second.verdict}/{second.confidence}"
                    )
                else:
                    verdict = ReviewVerdict(
                        verdict="suspicious",
                        confidence="low",
                        reasons=_merge_unique(
                            verdict.reasons
                            + second.reasons
                            + ["second reviewer disagreed with suspicion"]
                        ),
                        summary=verdict.summary,
                    )
                    logger.info(
                        f"[review] {node.id} consensus=disagree "
                        f"first={first_verdict}/{first_confidence} "
                        f"second={second.verdict}/{second.confidence}"
                    )

            node.review_verdict = verdict.verdict
            node.review_confidence = verdict.confidence
            node.review_reasons = list(verdict.reasons)
            if verdict.verdict in {"suspicious", "leaky"}:
                node.is_suspicious = True
                node.suspicion_reasons.extend(verdict.reasons)
            logger.info(f"[review] {node.id} verdict={verdict.verdict} confidence={verdict.confidence}")

        def final_key(n: SearchNode) -> tuple:
            hard_bad = hard_leakage_flag(n.review_verdict, n.review_confidence)
            penalty = adjusted_review_penalty(n, top_k, maximize=maximize)
            return (
                -hard_bad,                 # medium/high-confidence leaky always demoted
                -penalty,
                score_key(n.holdout_score),
                score_key(n.cv_score),
                n.created_at,
            )

        top_k.sort(key=final_key, reverse=True)
        return top_k

    # ── helpers ─────────────────────────────────────────────────────────

    def _persist(self) -> None:
        try:
            snapshot_path = self.workspace_dir / "journal.json"
            snapshot_path.write_text(
                json.dumps(self.journal.snapshot(), indent=2, default=str),
                encoding="utf-8",
            )
        except Exception as e:
            logger.debug(f"[persist] journal snapshot failed: {e}")

    def _remaining(self) -> float:
        return max(0.0, self.cfg.time_limit - (time.time() - self._started_at))

    def _fraction_used(self) -> float:
        if self.cfg.time_limit <= 0:
            return 1.0
        return min(1.0, (time.time() - self._started_at) / self.cfg.time_limit)

    def _budget_exhausted(self) -> bool:
        return self._remaining() <= self.cfg.search.grace_seconds


def _max_confidence(a: str, b: str) -> str:
    order = {"low": 0, "medium": 1, "high": 2}
    ia = order.get((a or "").lower(), 0)
    ib = order.get((b or "").lower(), 0)
    return "high" if max(ia, ib) >= 2 else ("medium" if max(ia, ib) >= 1 else "low")


def _merge_unique(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in items:
        s = str(raw).strip()
        if not s:
            continue
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out[:6]
