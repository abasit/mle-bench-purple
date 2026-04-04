import asyncio
import base64
import io
import tarfile
import logging
from pathlib import Path

from a2a.server.tasks import TaskUpdater
from a2a.types import (
    FileWithBytes,
    FilePart,
    Message,
    Part,
    TextPart,
)
from messenger import Messenger
from mlevolve_runner import run_competition

logger = logging.getLogger("mle-bench-purple")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
logger.addHandler(handler)


class Agent:
    def __init__(self):
        self.messenger = Messenger()

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        # Check if this is a follow-up message (no tar file)
        has_tar = any(
            isinstance(part.root, FilePart) and
            isinstance(part.root.file, FileWithBytes) and
            part.root.file.name == "competition.tar.gz"
            for part in message.parts
        )

        if not has_tar:
            logger.info("Ignoring follow-up message (no competition tar)")
            return

        logger.info("Received task, extracting data...")

        # Parse incoming message
        instructions = ""
        tar_bytes = None
        for part in message.parts:
            if isinstance(part.root, TextPart):
                instructions = part.root.text
                logger.info("Received instructions: " + instructions[:200] + "...")
            elif isinstance(part.root, FilePart):
                file_data = part.root.file
                if isinstance(file_data, FileWithBytes):
                    tar_bytes = base64.b64decode(file_data.bytes)

        # Extract tar
        work_dir = Path.cwd() / "work_dir"
        work_dir.mkdir(exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode='r:gz') as tar:
            tar.extractall(work_dir)
        logger.info(f"Extracted competition data to: {work_dir}")

        data_dir = work_dir / "home" / "data"

        # Read description
        description_path = data_dir / "description.md"
        description = description_path.read_text() if description_path.exists() else "No description found"
        logger.info(f"Read competition description ({len(description)} chars)")

        # Run mlevolve to produce a submission
        logger.info("Running mlevolve...")
        loop = asyncio.get_running_loop()
        submission_bytes = await loop.run_in_executor(
            None, lambda: run_competition(work_dir)
        )

        if submission_bytes is None:
            logger.warning("mlevolve produced no submission, falling back to sample submission")
            all_files = [str(p.relative_to(data_dir)) for p in data_dir.rglob("*") if p.is_file()]
            sample_submission = next(
                (data_dir / f for f in all_files if "sample" in f.lower() and f.endswith(".csv")),
                None,
            )
            submission_bytes = sample_submission.read_bytes() if sample_submission else b"id,target\n"

        logger.info("Submitting final artifact...")
        await updater.add_artifact(
            parts=[
                Part(root=FilePart(
                    file=FileWithBytes(
                        bytes=base64.b64encode(submission_bytes).decode('ascii'),
                        name="submission.csv",
                        mime_type="text/csv",
                    )
                ))
            ],
            name="Submission",
        )
        logger.info("Done.")