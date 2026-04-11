"""A2A agent: parse the incoming task, delegate solving to ``purple_next``."""

import asyncio
import base64
import io
import logging
import shutil
import tarfile
from pathlib import Path, PurePosixPath
from uuid import uuid4

from a2a.server.tasks import TaskUpdater
from a2a.types import FilePart, FileWithBytes, Message, Part, TaskState, TextPart
from a2a.utils import new_agent_text_message

from messenger import Messenger
from purple_next.runner import run_competition

logger = logging.getLogger("mle-bench-purple")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
    logger.addHandler(_handler)


class Agent:
    def __init__(self):
        self.work_dir: Path | None = None
        self._lock = asyncio.Lock()
        self.messenger = Messenger()

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        tar_bytes = self._extract_tar_bytes(message)
        if not tar_bytes:
            raise ValueError("Missing competition.tar.gz payload")

        work_dir = await self._prepare_work_dir()
        _safe_extract_tar(tar_bytes, work_dir)
        logger.info(f"Extracted competition data to: {work_dir}")

        await updater.update_status(
            state=TaskState.working,
            message=new_agent_text_message(
                "Competition data ready. Starting purple_next panel..."
            ),
        )

        loop = asyncio.get_running_loop()
        submission_bytes = await loop.run_in_executor(None, run_competition, work_dir)

        if submission_bytes is None:
            raise RuntimeError("purple_next produced no submission")

        await updater.update_status(
            state=TaskState.working,
            message=new_agent_text_message("Panel finished. Submitting best result..."),
        )

        await updater.add_artifact(
            parts=[
                Part(
                    root=FilePart(
                        file=FileWithBytes(
                            bytes=base64.b64encode(submission_bytes).decode("ascii"),
                            name="submission.csv",
                            mime_type="text/csv",
                        )
                    )
                )
            ],
            name="Submission",
        )
        logger.info("Done.")

    @staticmethod
    def _extract_tar_bytes(message: Message) -> bytes | None:
        for part in message.parts:
            if not isinstance(part.root, FilePart):
                continue
            file_data = part.root.file
            if isinstance(file_data, FileWithBytes) and file_data.name == "competition.tar.gz":
                return base64.b64decode(file_data.bytes)
        return None

    async def _prepare_work_dir(self) -> Path:
        async with self._lock:
            root = Path.cwd() / "work_dir"
            root.mkdir(exist_ok=True)
            if self.work_dir is not None and self.work_dir.exists():
                shutil.rmtree(self.work_dir, ignore_errors=True)
            self.work_dir = root / f"session-{uuid4().hex[:12]}"
            self.work_dir.mkdir(parents=True, exist_ok=False)
            return self.work_dir


def _safe_extract_tar(tar_bytes: bytes, destination: Path) -> None:
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tar:
        members = tar.getmembers()
        for member in members:
            member_path = PurePosixPath(member.name)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise ValueError(f"Unsafe tar member path: {member.name}")
            if member.issym() or member.islnk():
                raise ValueError(f"Tar links are not allowed: {member.name}")
            resolved_target = (destination / Path(*member_path.parts)).resolve()
            if destination != resolved_target and destination not in resolved_target.parents:
                raise ValueError(f"Tar member escapes destination: {member.name}")
        tar.extractall(destination, members=members)
