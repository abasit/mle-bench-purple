import asyncio
import sys
import threading
import time
from pathlib import Path

import pandas as pd
from a2a.types import TaskState
from a2a.utils import get_message_text

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent import A2AProgressBridge  # noqa: E402
from solver.interpreter import Interpreter  # noqa: E402


class DummyUpdater:
    def __init__(self):
        self.messages: list[tuple[TaskState, str]] = []

    async def update_status(self, state, message):
        self.messages.append((state, get_message_text(message) or ""))


def test_interpreter_restores_parent_session_state(tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    workspace_dir = tmp_path / "workspace"
    interpreter = Interpreter(workspace_dir=workspace_dir, data_dir=data_dir, timeout=30)

    parent_code = """
import pandas as pd

cached_model = {"bias": 41}
pd.DataFrame({"id": [1], "target": [cached_model["bias"]]}).to_csv("submission.csv", index=False)
print("FINAL VAL SCORE: 0.10")
print("METRIC DIRECTION: maximize")
""".strip()
    parent_result = interpreter.run(parent_code, "d001")
    assert parent_result.is_success
    assert parent_result.session_state_path is not None
    assert parent_result.session_state_path.exists()

    child_code = """
import pandas as pd

assert SESSION_RESTORED is True
assert SESSION_PARENT_NODE_ID == "d001"
cached_model["bias"] += 1
pd.DataFrame({"id": [1], "target": [cached_model["bias"]]}).to_csv("submission.csv", index=False)
print(f"FINAL VAL SCORE: {cached_model['bias'] / 100:.2f}")
print("METRIC DIRECTION: maximize")
""".strip()
    child_result = interpreter.run(
        child_code,
        "i002",
        parent_state_path=parent_result.session_state_path,
        session_parent_node_id="d001",
    )

    assert child_result.is_success
    assert child_result.submission_path is not None
    out = pd.read_csv(child_result.submission_path)
    assert out["target"].tolist() == [42]


def test_interpreter_discards_failed_session_state(tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    workspace_dir = tmp_path / "workspace"
    interpreter = Interpreter(workspace_dir=workspace_dir, data_dir=data_dir, timeout=30)

    failed_code = """
cached_model = {"bias": 41}
raise RuntimeError("boom")
""".strip()
    failed_result = interpreter.run(failed_code, "d001")

    assert not failed_result.is_success
    assert failed_result.session_state_path is None
    assert not (workspace_dir / "nodes" / "d001" / "session_state.pkl").exists()


def test_a2a_progress_bridge_streams_status_updates():
    loop = asyncio.new_event_loop()
    updater = DummyUpdater()
    bridge = A2AProgressBridge(loop, updater)
    loop_thread = threading.Thread(target=loop.run_forever, daemon=True)
    loop_thread.start()

    try:
        worker = threading.Thread(
            target=lambda: (
                bridge.on_phase("drafts", "Generating 3 drafts"),
                bridge.on_step(2, 10, "valid=1 buggy=1 best=0.81234"),
                bridge.on_best("d001", 0.81234, "d001(draft val=0.8123)"),
            ),
            daemon=True,
        )
        worker.start()
        worker.join(timeout=2)

        deadline = time.time() + 2.0
        while len(updater.messages) < 3 and time.time() < deadline:
            time.sleep(0.05)

        texts = [text for state, text in updater.messages if state == TaskState.working]
        assert any("Solver phase [drafts]" in text for text in texts)
        assert any("Solver step 2/10" in text for text in texts)
        assert any("New best candidate: d001" in text for text in texts)
    finally:
        loop.call_soon_threadsafe(loop.stop)
        loop_thread.join(timeout=2)
        loop.close()
