"""Panel — parallel pass@K fan-out.

Each panel seat runs its own isolated TreeLoop with a unique disposition,
seed, and workspace. Seats run concurrently in a thread pool — the LLM
calls are I/O bound and subprocess execution already uses its own
processes per node, so threads are the right granularity here.

After all seats finish, the panel merges top-K candidates from each run
and returns them in a single globally-ranked list. The reviewer agent
already ran inside each TreeLoop's finalize, so panel-level rerank just
re-applies the same (verdict, holdout, cv, recency) key.
"""

from __future__ import annotations

import copy
import json
import logging
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from .config import Config
from .exec import Interpreter
from .llm import LLMClient
from .prompts import disposition_for_run
from .tree import SearchNode, TreeLoop
from .tree.loop import RunContext, RunReport

logger = logging.getLogger("purple_next")


@dataclass
class SeatResult:
    seat_index: int
    disposition: str
    report: RunReport | None
    error: str = ""


@dataclass
class PanelResult:
    seats: list[SeatResult]
    final_candidates: list[SearchNode]
    total_elapsed_s: float


def run_panel(
    *,
    cfg: Config,
    data_dir: Path,
    workspace_root: Path,
    build_context: callable,
) -> PanelResult:
    """Fan out pass_k independent tree runs and merge candidates.

    ``build_context(cfg, seat_index)`` returns a ``RunContext`` for the seat.
    It lets the caller (runner) own task description, data preview, and
    contract summary construction once and reuse it across seats.
    """
    pass_k = max(1, cfg.search.pass_k)
    started = time.time()
    workspace_root.mkdir(parents=True, exist_ok=True)

    if pass_k == 1:
        ctx = build_context(cfg, 0)
        result = _run_seat(
            cfg=cfg,
            data_dir=data_dir,
            workspace_dir=workspace_root,
            context=ctx,
            seat_index=0,
        )
        candidates = _merge_candidates([result], cfg=cfg)
        return PanelResult(
            seats=[result],
            final_candidates=candidates,
            total_elapsed_s=time.time() - started,
        )

    per_seat_time = cfg.time_limit / pass_k

    def _seat_task(seat_index: int) -> SeatResult:
        seat_cfg = copy.deepcopy(cfg)
        seat_cfg.time_limit = per_seat_time
        seat_cfg.seed = cfg.seed + seat_index
        if seat_index < len(cfg.search.seat_temperatures):
            seat_cfg.llm.temperature = cfg.search.seat_temperatures[seat_index]
        logger.info(
            f"[panel] seat {seat_index + 1} temperature={seat_cfg.llm.temperature}"
        )
        seat_workspace = workspace_root / f"seat_{seat_index + 1:02d}"
        seat_workspace.mkdir(parents=True, exist_ok=True)
        ctx = build_context(seat_cfg, seat_index)
        return _run_seat(
            cfg=seat_cfg,
            data_dir=data_dir,
            workspace_dir=seat_workspace,
            context=ctx,
            seat_index=seat_index,
        )

    logger.info(f"[panel] fanning out {pass_k} seats, per-seat budget {per_seat_time:.0f}s")

    seats: list[SeatResult] = [None] * pass_k  # type: ignore[list-item]
    with ThreadPoolExecutor(max_workers=pass_k) as ex:
        futures: dict[Future, int] = {ex.submit(_seat_task, i): i for i in range(pass_k)}
        for fut in as_completed(futures):
            i = futures[fut]
            try:
                seats[i] = fut.result()
            except Exception as e:
                logger.exception(f"[panel] seat {i} raised: {e}")
                seats[i] = SeatResult(
                    seat_index=i,
                    disposition=disposition_for_run(i, cfg.search.dispositions or None),
                    report=None,
                    error=str(e),
                )

    final_candidates = _merge_candidates(seats, cfg=cfg)

    manifest = {
        "seats": [
            {
                "seat_index": s.seat_index,
                "disposition": s.disposition,
                "candidates": [n.short() for n in (s.report.candidates if s.report else [])],
                "elapsed_s": s.report.elapsed_s if s.report else 0.0,
                "error": s.error,
            }
            for s in seats
        ],
        "final": [n.short() for n in final_candidates],
        "total_elapsed_s": time.time() - started,
    }
    try:
        (workspace_root / "panel.json").write_text(
            json.dumps(manifest, indent=2, default=str), encoding="utf-8"
        )
    except Exception as e:
        logger.debug(f"[panel] manifest write failed: {e}")

    return PanelResult(
        seats=seats,
        final_candidates=final_candidates,
        total_elapsed_s=time.time() - started,
    )


def _run_seat(
    *,
    cfg: Config,
    data_dir: Path,
    workspace_dir: Path,
    context: RunContext,
    seat_index: int,
) -> SeatResult:
    disposition_first_line = context.disposition.splitlines()[0] if context.disposition else ""
    logger.info(
        f"[panel] seat {seat_index + 1} starting: disposition={disposition_first_line!r}"
    )
    llm = LLMClient(cfg.llm)
    interpreter = Interpreter(
        workspace_dir=workspace_dir,
        data_dir=data_dir,
        timeout=cfg.exec.timeout,
    )
    loop = TreeLoop(
        cfg=cfg,
        llm=llm,
        interpreter=interpreter,
        workspace_dir=workspace_dir,
        run_context=context,
    )
    try:
        report = loop.run()
    except Exception as e:
        logger.exception(f"[panel] seat {seat_index + 1} run failed: {e}")
        return SeatResult(
            seat_index=seat_index,
            disposition=context.disposition,
            report=None,
            error=str(e),
        )
    return SeatResult(seat_index=seat_index, disposition=context.disposition, report=report)


def _merge_candidates(seats: list[SeatResult], *, cfg: Config) -> list[SearchNode]:
    all_candidates: list[SearchNode] = []
    for s in seats:
        if s is None or s.report is None:
            continue
        all_candidates.extend(s.report.candidates)

    if not all_candidates:
        return []

    maximize = next(
        (n.maximize for n in all_candidates if n.maximize is not None),
        True,
    )

    def score_key(v: float | None) -> float:
        if v is None:
            return float("-inf")
        return v if maximize else -v

    verdict_order = {"clean": 0, "suspicious": 1, "leaky": 2}

    def final_key(n: SearchNode) -> tuple:
        bad = verdict_order.get(n.review_verdict or "clean", 0)
        return (
            -bad,
            score_key(n.holdout_score),
            score_key(n.cv_score),
            n.created_at,
        )

    ranked = sorted(all_candidates, key=final_key, reverse=True)

    # Dedupe by submission file content if paths are available.
    seen: set[bytes] = set()
    final: list[SearchNode] = []
    for node in ranked:
        sub = node.submission_path
        if sub is not None and sub.exists():
            try:
                raw = sub.read_bytes()
                if raw in seen:
                    continue
                seen.add(raw)
            except Exception:
                pass
        final.append(node)
        if len(final) >= cfg.search.final_top_k * max(1, cfg.search.pass_k):
            break
    return final
