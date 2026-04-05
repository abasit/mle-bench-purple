import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()

from mlevolve.engine.agent_search import AgentSearch
from mlevolve.engine.executor import Interpreter
from mlevolve.engine.search_node import Journal
from mlevolve.config import _load_cfg, prep_cfg, load_task_desc, prep_agent_workspace, save_run
from mlevolve.utils.seed import set_global_seed
from mlevolve.engine.coldstart import build_guidance_description

logger = logging.getLogger("mle-bench-purple")

_CONFIG_PATH = Path(__file__).parent.parent / "mlevolve.yaml"


def run_competition(work_dir: Path) -> bytes | None:
    """Run mlevolve on a competition and return the best submission as bytes.

    Args:
        work_dir: Root directory where the competition tar was extracted.
                  data_dir = work_dir / "home" / "data"
                  workspace_dir = work_dir / "mlevolve"

    Returns:
        Best submission CSV as bytes, or None if no valid submission was produced.
    """
    work_dir = Path(work_dir)
    data_dir = work_dir / "home" / "data"
    workspace_dir = work_dir / "mlevolve"

    coldstart_dir = Path(__file__).parent / "mlevolve" / "engine" / "coldstart"

    cfg = _load_cfg(path=_CONFIG_PATH, use_cli_args=False)
    cfg.data_dir = str(data_dir)
    cfg.coldstart.task_json_path = str(coldstart_dir / "competition_tag_classified.json")
    cfg.coldstart.model_json_path = str(coldstart_dir / "models_guidance_classified.json")

    desc_file = data_dir / "description.md"
    if desc_file.exists():
        cfg.desc_file = str(desc_file)
    else:
        cfg.goal = "Solve the machine learning competition in the data directory."

    cfg.log_dir = str(workspace_dir)
    cfg.workspace_dir = str(workspace_dir)
    cfg.agent.use_global_memory = False  # skip embedding model download

    cfg = prep_cfg(cfg)
    set_global_seed(cfg.agent.seed)

    if cfg.coldstart.use_coldstart:
        cfg.coldstart.description = build_guidance_description(cfg)

    prep_agent_workspace(cfg)

    task_desc = load_task_desc(cfg)
    journal = Journal()
    agent = AgentSearch(task_desc=task_desc, cfg=cfg, journal=journal)
    interpreter = Interpreter(cfg.workspace_dir, timeout=cfg.exec.timeout, cfg=cfg)

    def exec_callback(*args, **kwargs):
        return interpreter.run(*args, **kwargs)

    def step_task(node=None):
        return agent.step(exec_callback=exec_callback, node=node)

    total_steps = cfg.agent.steps
    initial_draft_count = min(cfg.agent.initial_drafts, total_steps)
    max_workers = interpreter.max_parallel_run
    lock = threading.Lock()
    completed = 0
    grace_period = getattr(cfg.agent, 'timeout_grace_period', 120)
    search_start = time.time()

    def _time_remaining() -> float:
        return cfg.agent.time_limit - (time.time() - search_start)

    def _within_time_limit() -> bool:
        return _time_remaining() > grace_period

    # Phase 1: generate draft code sequentially (deferred — no execution yet)
    pending_draft_nodes = []
    for i in range(initial_draft_count):
        if not _within_time_limit():
            logger.warning(f"Time limit reached during draft generation, stopping at {i}/{initial_draft_count}")
            break
        try:
            node = agent.step(exec_callback=exec_callback, node=None, execute_immediately=False)
            pending_draft_nodes.append(node)
            logger.info(f"Draft {i + 1}/{initial_draft_count} generated: {node.id}")
        except Exception:
            logger.exception(f"Draft {i + 1} generation failed")

    if not pending_draft_nodes:
        logger.error("No drafts generated, aborting search")
        return None

    # Phase 2: execute drafts in parallel, then continue stepping
    def execute_draft(node):
        try:
            return agent.execute_deferred_node(node, exec_callback)
        except Exception:
            logger.exception(f"Draft node {node.id} execution failed")
            return None

    _stop = threading.Event()

    executor = ThreadPoolExecutor(max_workers=max_workers)
    try:
        futures = set()
        for i, node in enumerate(pending_draft_nodes):
            futures.add(executor.submit(execute_draft, node))
            if i < len(pending_draft_nodes) - 1:
                time.sleep(2)  # brief stagger to avoid workspace init conflicts

        _last_heartbeat = time.time()
        _HEARTBEAT_INTERVAL = 30  # log "still waiting" every 30s of silence

        while completed < total_steps and not _stop.is_set():
            if not _within_time_limit():
                logger.info(
                    f"Time limit reached ({cfg.agent.time_limit - grace_period:.0f}s used), "
                    f"stopping search at {completed}/{total_steps} steps"
                )
                break

            done, _ = wait(futures, return_when=FIRST_COMPLETED, timeout=1.0)
            if not done:
                now = time.time()
                if futures and now - _last_heartbeat >= _HEARTBEAT_INTERVAL:
                    elapsed = now - search_start
                    time_remaining = _time_remaining()
                    logger.info(
                        f"[waiting] {len(futures)} step(s) still running — "
                        f"{elapsed:.0f}s elapsed, {time_remaining:.0f}s remaining, "
                        f"{completed}/{total_steps} steps done"
                    )
                    _last_heartbeat = now
                continue

            _last_heartbeat = time.time()
            for fut in done:
                futures.discard(fut)
                try:
                    cur_node = fut.result()
                except Exception:
                    logger.exception("Task execution raised an exception")
                    cur_node = None

                with lock:
                    save_run(cfg, journal)
                    completed = len(journal) - 1  # exclude virtual root

                if completed + len(futures) < total_steps and _within_time_limit() and not _stop.is_set():
                    futures.add(executor.submit(step_task, cur_node))

                elapsed = time.time() - search_start
                logger.info(
                    f"Progress: {completed}/{total_steps} steps, {len(futures)} running, "
                    f"{elapsed:.0f}s / {cfg.agent.time_limit}s elapsed"
                )

    except KeyboardInterrupt:
        logger.warning("KeyboardInterrupt received — stopping search and killing subprocesses")
        _stop.set()
        interpreter.cleanup_session(-1)  # kill all running subprocesses immediately
    finally:
        for fut in list(futures):
            fut.cancel()
        # wait=False so we don't block on threads stuck in subprocess calls
        executor.shutdown(wait=False, cancel_futures=True)

    interpreter.cleanup_session(-1)

    submission_path = cfg.workspace_dir / "best_submission" / "submission.csv"
    if not submission_path.exists():
        logger.warning("No submission produced by mlevolve")
        return None

    # Attempt final ensemble: blend top-K submissions for a free boost
    try:
        blended = _blend_top_submissions(cfg.workspace_dir, agent.metric_maximize)
    except Exception:
        logger.exception("[blend] Blending raised an unexpected exception — using raw best submission")
        blended = None
    if blended is not None:
        blended_path = cfg.workspace_dir / "best_submission" / "submission.csv"
        blended_path.write_bytes(blended)
        logger.info(f"Blended submission saved to {blended_path}")
        return blended

    logger.info(f"Best submission: {submission_path}")
    return submission_path.read_bytes()


def _blend_top_submissions(workspace_dir: Path, metric_maximize: Optional[bool], top_n: int = 3) -> Optional[bytes]:
    """Blend top-N diverse submissions and return the best-blended CSV bytes.

    Handles three column types correctly:
    - **ID columns** (object/string): kept as-is from the first submission.
    - **Boolean columns** (True/False): majority-vote across submissions,
      then serialised back to 'True'/'False' strings — NOT as floats.
      This is the critical correctness fix: competitions like Spaceship Titanic
      expect boolean strings; pd.api.types.is_numeric_dtype returns True for
      bool dtype, which previously caused 0.0/1.0 floats to leak into the CSV
      and fail the green-agent validator.
    - **Continuous numeric columns**: rank-averaged (metric-agnostic default).
      If all values are in [0, 1] across every submission, also tries arithmetic
      mean and picks whichever has higher prediction variance.

    Returns None if fewer than 2 valid submissions with matching columns are found.
    """
    top_solution_dir = workspace_dir / "top_solution"
    if not top_solution_dir.exists():
        return None

    # Only blend the top_n best submissions (directories are named top1, top2, ...)
    submission_dirs = sorted(top_solution_dir.iterdir())[:top_n]
    dfs = []
    ref_cols = None

    for rank_dir in submission_dirs:
        csv_path = rank_dir / "submission.csv"
        if not csv_path.exists():
            continue
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            logger.warning(f"[blend] Failed to read {csv_path}: {e}")
            continue

        if ref_cols is None:
            ref_cols = list(df.columns)
        elif list(df.columns) != ref_cols:
            logger.info(f"[blend] Skipping {rank_dir.name}: different columns")
            continue

        dfs.append(df)

    if len(dfs) < 2:
        logger.info(f"[blend] Only {len(dfs)} compatible submission(s) found — skipping blend")
        return None

    logger.info(f"[blend] Blending {len(dfs)} submissions")

    ref_df = dfs[0].copy()

    # ── Classify columns ─────────────────────────────────────────────────
    # Boolean columns get majority-vote treatment, NOT rank-averaging.
    # Separating them avoids the 0.0/1.0 float corruption bug where
    # pandas treats bool as numeric and rank-averaging emits floats.
    #
    # A column is treated as boolean if:
    #   (a) its dtype is bool in the reference df, OR
    #   (b) it only contains the values {0, 1} / {True, False} across ALL dfs.
    def _is_binary_col(col: str) -> bool:
        if pd.api.types.is_bool_dtype(ref_df[col]):
            return True
        if pd.api.types.is_numeric_dtype(ref_df[col]):
            unique_vals = set()
            for df in dfs:
                unique_vals.update(df[col].dropna().unique())
            return unique_vals <= {0, 1, 0.0, 1.0, True, False}
        return False

    bool_cols = [c for c in ref_cols if _is_binary_col(c)]
    id_cols = [
        c for c in ref_cols
        if not pd.api.types.is_numeric_dtype(ref_df[c]) and c not in bool_cols
    ]
    numeric_cols = [
        c for c in ref_cols
        if c not in bool_cols and c not in id_cols
    ]

    logger.info(f"[blend] id_cols={id_cols}, bool_cols={bool_cols}, numeric_cols={numeric_cols}")

    blended = dfs[0][id_cols].copy()

    # ── Boolean columns: majority vote ────────────────────────────────────
    # Preserves the original dtype (bool → True/False, int → 0/1) so the
    # output CSV matches the format the competition validator expects.
    for col in bool_cols:
        votes = sum(df[col].astype(float) for df in dfs) / len(dfs)
        majority = votes >= 0.5
        # Preserve original serialisation format
        if pd.api.types.is_bool_dtype(ref_df[col]):
            blended[col] = majority  # bool → True/False in CSV
        else:
            blended[col] = majority.astype(int)  # int → 0/1 in CSV

    # ── Continuous columns: rank-average (+ optional arith mean) ─────────
    if numeric_cols:
        ranked = [df[numeric_cols].rank(pct=True) for df in dfs]
        avg_ranks = sum(ranked) / len(ranked)

        rank_blended_vals = {col: avg_ranks[col].values for col in numeric_cols}

        # Try arithmetic mean when all values are in [0, 1]
        arith_blended_vals = None
        try:
            all_in_01 = all(
                dfs[i][col].between(0.0, 1.0, inclusive="both").all()
                for i in range(len(dfs))
                for col in numeric_cols
            )
            if all_in_01:
                mean_preds = sum(df[numeric_cols].values for df in dfs) / len(dfs)
                arith_blended_vals = {col: mean_preds[:, j] for j, col in enumerate(numeric_cols)}
                logger.info("[blend] Arithmetic mean applicable (all columns in [0,1])")
        except Exception as e:
            logger.warning(f"[blend] Arithmetic mean strategy failed: {e}")

        # Pick strategy with higher prediction variance (more discriminative)
        if arith_blended_vals is not None:
            rank_var = float(np.mean([np.var(rank_blended_vals[c]) for c in numeric_cols]))
            arith_var = float(np.mean([np.var(arith_blended_vals[c]) for c in numeric_cols]))
            if arith_var > rank_var:
                logger.info(f"[blend] Using arithmetic mean (var={arith_var:.6f} > {rank_var:.6f})")
                chosen_vals = arith_blended_vals
            else:
                logger.info(f"[blend] Using rank average (var={rank_var:.6f} >= {arith_var:.6f})")
                chosen_vals = rank_blended_vals
        else:
            chosen_vals = rank_blended_vals

        for col in numeric_cols:
            blended[col] = chosen_vals[col]
    else:
        logger.info("[blend] No continuous numeric columns — only boolean majority-vote applied")

    # Reorder columns to match original submission
    blended = blended[ref_cols]
    return blended.to_csv(index=False).encode()
