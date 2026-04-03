import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path

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

    # Phase 1: generate draft code sequentially (deferred — no execution yet)
    pending_draft_nodes = []
    for i in range(initial_draft_count):
        try:
            node = agent.step(exec_callback=exec_callback, node=None, execute_immediately=False)
            pending_draft_nodes.append(node)
            logger.info(f"Draft {i + 1}/{initial_draft_count} generated: {node.id}")
        except Exception:
            logger.exception(f"Draft {i + 1} generation failed")

    # Phase 2: execute drafts in parallel, then continue stepping
    def execute_draft(node):
        try:
            return agent.execute_deferred_node(node, exec_callback)
        except Exception:
            logger.exception(f"Draft node {node.id} execution failed")
            return None

    executor = ThreadPoolExecutor(max_workers=max_workers)
    try:
        futures = set()
        for i, node in enumerate(pending_draft_nodes):
            futures.add(executor.submit(execute_draft, node))
            if i < len(pending_draft_nodes) - 1:
                time.sleep(10)  # stagger to avoid init conflicts

        while completed < total_steps:
            done, _ = wait(futures, return_when=FIRST_COMPLETED, timeout=1.0)
            if not done:
                continue

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

                if completed + len(futures) < total_steps:
                    futures.add(executor.submit(step_task, cur_node))

                logger.info(f"Progress: {completed}/{total_steps} steps, {len(futures)} running")
    finally:
        executor.shutdown(wait=True)

    interpreter.cleanup_session(-1)

    submission_path = cfg.workspace_dir / "best_submission" / "submission.csv"
    if submission_path.exists():
        logger.info(f"Best submission: {submission_path}")
        return submission_path.read_bytes()

    logger.warning("No submission produced by mlevolve")
    return None
