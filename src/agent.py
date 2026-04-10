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
from a2a.utils import get_message_text, new_agent_text_message

from messenger import Messenger
from solver import run_competition_candidates
from solver.utils import find_sample_submission

logger = logging.getLogger("mle-bench-purple")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
if not logger.handlers:
    logger.addHandler(handler)

_VALIDATION_TIMEOUT_SECONDS = 180.0


class Agent:
    def __init__(self):
        self.messenger = Messenger()
        self.work_dir: Path | None = None
        self._state_lock = asyncio.Lock()
        self._pending_validation: asyncio.Future[str] | None = None

    async def run(self, message: Message, updater: TaskUpdater) -> bool:
        """Handle an incoming A2A message.

        Returns:
            True when the executor should auto-complete the task after this call.
            False for follow-up validation messages that only acknowledge an
            in-flight solve without terminating the main task.
        """
        has_tar = any(
            isinstance(part.root, FilePart)
            and isinstance(part.root.file, FileWithBytes)
            and part.root.file.name == "competition.tar.gz"
            for part in message.parts
        )

        if not has_tar:
            return await self._handle_follow_up(message)

        logger.info("Received task, extracting data...")
        instructions, tar_bytes = self._parse_initial_message(message)
        if not tar_bytes:
            raise ValueError("Missing competition.tar.gz payload")

        work_dir = await self._prepare_work_dir()
        _safe_extract_tar_bytes(tar_bytes, work_dir)
        logger.info(f"Extracted competition data to: {work_dir}")

        data_dir = work_dir / "home" / "data"
        description_path = data_dir / "description.md"
        description = (
            description_path.read_text(encoding="utf-8", errors="replace")
            if description_path.exists()
            else "No description found"
        )
        logger.info(f"Received instructions: {instructions[:200]}...")
        logger.info(f"Read competition description ({len(description)} chars)")

        await updater.update_status(
            state=TaskState.working,
            message=new_agent_text_message(
                "Competition data ready. Starting solver tree search..."
            ),
        )

        logger.info("Running solver...")
        loop = asyncio.get_running_loop()
        submission_candidates = await loop.run_in_executor(
            None,
            lambda: run_competition_candidates(work_dir),
        )

        if not submission_candidates:
            logger.warning("Solver produced no submission, falling back to sample submission")
            await updater.update_status(
                state=TaskState.working,
                message=new_agent_text_message(
                    "Warning: solver produced no valid submission. Falling back to sample submission."
                ),
            )
            submission_bytes = self._fallback_submission(data_dir)
        else:
            submission_bytes = None
            total = len(submission_candidates)
            for idx, candidate in enumerate(submission_candidates, start=1):
                validation_result = await self._request_validation(updater, candidate)
                if validation_result is None:
                    submission_bytes = candidate
                    logger.info(
                        f"No green validation response for candidate {idx}/{total}; "
                        "keeping the current best candidate."
                    )
                    break
                logger.info(f"Green validation result for candidate {idx}/{total}: {validation_result}")
                if _validation_is_success(validation_result):
                    submission_bytes = candidate
                    break
                if idx < total:
                    logger.warning(
                        f"Green validation rejected candidate {idx}/{total}; trying the next-best submission."
                    )
                    await updater.update_status(
                        state=TaskState.working,
                        message=new_agent_text_message(
                            f"Green validation rejected candidate {idx}/{total}. Trying the next-best solution."
                        ),
                    )
            if submission_bytes is None:
                logger.warning(
                    "Green validation rejected every candidate submission; "
                    "falling back to sample submission."
                )
                await updater.update_status(
                    state=TaskState.working,
                    message=new_agent_text_message(
                        "Green validation rejected every candidate submission. Falling back to sample submission."
                    ),
                )
                submission_bytes = self._fallback_submission(data_dir)

            await updater.update_status(
                state=TaskState.working,
                message=new_agent_text_message(
                    "Solver search complete. Submitting best solution..."
                ),
            )

        logger.info("Submitting final artifact...")
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
        return True

    async def _handle_follow_up(self, message: Message) -> bool:
        text = (get_message_text(message) or "").strip()
        logger.info(f"Received follow-up message: {text[:200] or '(empty)'}")

        async with self._state_lock:
            waiter = self._pending_validation

        if waiter is not None and not waiter.done():
            waiter.set_result(text)
        else:
            logger.info("No pending validation request; follow-up acknowledged and ignored.")
        return False

    def _parse_initial_message(self, message: Message) -> tuple[str, bytes | None]:
        instructions = ""
        tar_bytes = None
        for part in message.parts:
            if isinstance(part.root, TextPart):
                instructions = part.root.text
            elif isinstance(part.root, FilePart):
                file_data = part.root.file
                if (
                    isinstance(file_data, FileWithBytes)
                    and file_data.name == "competition.tar.gz"
                ):
                    tar_bytes = base64.b64decode(file_data.bytes)
        return instructions, tar_bytes

    async def _prepare_work_dir(self) -> Path:
        async with self._state_lock:
            root = Path.cwd() / "work_dir"
            root.mkdir(exist_ok=True)
            if self.work_dir is not None and self.work_dir.exists():
                shutil.rmtree(self.work_dir, ignore_errors=True)
            self.work_dir = root / f"session-{uuid4().hex[:12]}"
            self.work_dir.mkdir(parents=True, exist_ok=False)
            return self.work_dir

    async def _request_validation(
        self,
        updater: TaskUpdater,
        submission_bytes: bytes,
    ) -> str | None:
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        async with self._state_lock:
            self._pending_validation = future

        validation_message = updater.new_agent_message(
            parts=[
                Part(root=TextPart(text="validate")),
                Part(
                    root=FilePart(
                        file=FileWithBytes(
                            bytes=base64.b64encode(submission_bytes).decode("ascii"),
                            name="submission.csv",
                            mime_type="text/csv",
                        )
                    )
                ),
            ]
        )
        await updater.update_status(
            state=TaskState.working,
            message=validation_message,
        )

        try:
            return await asyncio.wait_for(future, timeout=_VALIDATION_TIMEOUT_SECONDS)
        except TimeoutError:
            logger.warning("Timed out waiting for green validation response.")
            return None
        finally:
            async with self._state_lock:
                if self._pending_validation is future:
                    self._pending_validation = None

    @staticmethod
    def _fallback_submission(data_dir: Path) -> bytes:
        sample_submission = find_sample_submission(data_dir)
        if sample_submission is not None:
            return sample_submission.read_bytes()
        return b"id,target\n"


def _validation_is_success(result_text: str) -> bool:
    return "submission is valid" in (result_text or "").strip().lower()


def _safe_extract_tar_bytes(tar_bytes: bytes, destination: Path) -> None:
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
