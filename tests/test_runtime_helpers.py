import base64
import asyncio
import io
import sys
import tarfile
from pathlib import Path

from a2a.types import FilePart, FileWithBytes, Message, Part, Role, TextPart

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import agent as agent_module  # noqa: E402
from agent import Agent, _safe_extract_tar_bytes, _validation_is_success  # noqa: E402


def _tar_bytes(entries: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, content in entries.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))
    return buf.getvalue()


def test_safe_extract_tar_bytes_rejects_path_traversal(tmp_path: Path):
    payload = _tar_bytes({"../escape.txt": b"bad"})
    target = tmp_path / "work"

    try:
        _safe_extract_tar_bytes(payload, target)
        assert False, "Expected path traversal tar to be rejected"
    except ValueError as exc:
        assert "Unsafe tar member path" in str(exc)


def test_safe_extract_tar_bytes_extracts_valid_archive(tmp_path: Path):
    payload = _tar_bytes({"home/data/description.md": b"hello"})
    target = tmp_path / "work"

    _safe_extract_tar_bytes(payload, target)

    assert (target / "home" / "data" / "description.md").read_text() == "hello"


def test_validation_result_parser():
    assert _validation_is_success("Submission is valid.")
    assert not _validation_is_success("Submission invalid! bad columns")


def test_agent_follow_up_resolves_pending_validation():
    async def _run():
        agent = Agent()
        future = asyncio.get_running_loop().create_future()
        agent._pending_validation = future

        msg = Message(
            kind="message",
            role=Role.user,
            parts=[Part(root=TextPart(text="Submission is valid."))],
            message_id="m1",
        )

        should_complete = await agent.run(msg, updater=None)  # type: ignore[arg-type]
        assert should_complete is False
        assert future.done()
        assert future.result() == "Submission is valid."

    asyncio.run(_run())


def test_agent_requests_validation_and_reports_submission_artifact():
    class StubUpdater:
        def __init__(self, agent: Agent):
            self.agent = agent
            self.statuses: list[tuple[object, Message]] = []
            self.artifacts: list[tuple[str, list[Part]]] = []

        def new_agent_message(self, parts: list[Part]) -> Message:
            return Message(
                kind="message",
                role=Role.agent,
                parts=parts,
                message_id="validate-msg",
            )

        async def update_status(self, state, message: Message) -> None:
            self.statuses.append((state, message))
            text_parts = [
                part.root.text
                for part in message.parts
                if isinstance(part.root, TextPart)
            ]
            if any("validate" in text.lower() for text in text_parts):
                follow_up = Message(
                    kind="message",
                    role=Role.user,
                    parts=[Part(root=TextPart(text="Submission is valid."))],
                    message_id="m-follow-up",
                )
                await self.agent.run(follow_up, self)  # type: ignore[arg-type]

        async def add_artifact(self, parts: list[Part], name: str) -> None:
            self.artifacts.append((name, parts))

    async def _run():
        agent = Agent()
        updater = StubUpdater(agent)

        tar_payload = _tar_bytes({"home/data/description.md": b"benchmark task"})
        msg = Message(
            kind="message",
            role=Role.user,
            parts=[
                Part(root=TextPart(text="Solve this competition.")),
                Part(
                    root=FilePart(
                        file=FileWithBytes(
                            bytes=base64.b64encode(tar_payload).decode("ascii"),
                            name="competition.tar.gz",
                            mime_type="application/gzip",
                        )
                    )
                ),
            ],
            message_id="m-init",
        )

        original = agent_module.run_competition_candidates
        agent_module.run_competition_candidates = lambda work_dir: [b"id,target\n1,0.5\n"]
        try:
            should_complete = await agent.run(msg, updater)  # type: ignore[arg-type]
        finally:
            agent_module.run_competition_candidates = original

        assert should_complete is True
        assert updater.artifacts
        name, parts = updater.artifacts[-1]
        assert name == "Submission"
        file_part = parts[0].root
        assert isinstance(file_part, FilePart)
        assert file_part.file.name == "submission.csv"
        assert base64.b64decode(file_part.file.bytes) == b"id,target\n1,0.5\n"
        assert any(
            any(
                isinstance(part.root, TextPart) and "validate" in part.root.text.lower()
                for part in message.parts
            )
            for _, message in updater.statuses
        )

    asyncio.run(_run())


def test_agent_tries_next_candidate_after_validation_rejection():
    class StubUpdater:
        def __init__(self, agent: Agent):
            self.agent = agent
            self.statuses: list[tuple[object, Message]] = []
            self.artifacts: list[tuple[str, list[Part]]] = []
            self.validation_attempts = 0

        def new_agent_message(self, parts: list[Part]) -> Message:
            return Message(
                kind="message",
                role=Role.agent,
                parts=parts,
                message_id=f"validate-msg-{self.validation_attempts}",
            )

        async def update_status(self, state, message: Message) -> None:
            self.statuses.append((state, message))
            text_parts = [
                part.root.text
                for part in message.parts
                if isinstance(part.root, TextPart)
            ]
            if any("validate" in text.lower() for text in text_parts):
                self.validation_attempts += 1
                result_text = (
                    "Submission invalid: bad columns"
                    if self.validation_attempts == 1
                    else "Submission is valid."
                )
                follow_up = Message(
                    kind="message",
                    role=Role.user,
                    parts=[Part(root=TextPart(text=result_text))],
                    message_id=f"m-follow-up-{self.validation_attempts}",
                )
                await self.agent.run(follow_up, self)  # type: ignore[arg-type]

        async def add_artifact(self, parts: list[Part], name: str) -> None:
            self.artifacts.append((name, parts))

    async def _run():
        agent = Agent()
        updater = StubUpdater(agent)

        tar_payload = _tar_bytes({"home/data/description.md": b"benchmark task"})
        msg = Message(
            kind="message",
            role=Role.user,
            parts=[
                Part(root=TextPart(text="Solve this competition.")),
                Part(
                    root=FilePart(
                        file=FileWithBytes(
                            bytes=base64.b64encode(tar_payload).decode("ascii"),
                            name="competition.tar.gz",
                            mime_type="application/gzip",
                        )
                    )
                ),
            ],
            message_id="m-init",
        )

        original = agent_module.run_competition_candidates
        agent_module.run_competition_candidates = lambda work_dir: [
            b"id,target\n1,0.1\n",
            b"id,target\n1,0.9\n",
        ]
        try:
            should_complete = await agent.run(msg, updater)  # type: ignore[arg-type]
        finally:
            agent_module.run_competition_candidates = original

        assert should_complete is True
        assert updater.validation_attempts == 2
        name, parts = updater.artifacts[-1]
        assert name == "Submission"
        file_part = parts[0].root
        assert isinstance(file_part, FilePart)
        assert base64.b64decode(file_part.file.bytes) == b"id,target\n1,0.9\n"
        assert any(
            any(
                isinstance(part.root, TextPart)
                and "trying the next-best solution" in part.root.text.lower()
                for part in message.parts
            )
            for _, message in updater.statuses
        )

    asyncio.run(_run())
