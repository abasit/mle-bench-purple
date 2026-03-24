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
    TaskState,
    TextPart,
)
from a2a.utils import new_agent_text_message
from messenger import Messenger

logger = logging.getLogger("mle-bench-purple")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
logger.addHandler(handler)


class Agent:
    def __init__(self):
        self.messenger = Messenger()

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        logger.info("Received task, extracting data...")

        # 1. Parse incoming message: extract instructions text and tar bytes
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

        if not tar_bytes:
            logger.info("Error: No competition data received")
            return

        # 2. Extract tar to working directory (for easy inspection)
        work_dir = Path.cwd() / "work_dir"
        work_dir.mkdir(exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode='r:gz') as tar:
            tar.extractall(work_dir)
        logger.info(f"Extracted competition data to: {work_dir}")

        data_dir = work_dir / "home" / "data"

        # 3. Read description.md
        description_path = data_dir / "description.md"
        description = description_path.read_text() if description_path.exists() else "No description found"
        logger.info(f"Read competition description ({len(description)} chars)")

        # 4. List available files
        all_files = [str(p.relative_to(data_dir)) for p in data_dir.rglob("*") if p.is_file()]
        logger.info(f"Found {len(all_files)} files: {all_files[:20]}")

        # 5. Find sample submission as baseline
        sample_submission = None
        for f in all_files:
            if "sample" in f.lower() and f.endswith(".csv"):
                sample_submission = data_dir / f
                break

        if sample_submission and sample_submission.exists():
            submission_bytes = sample_submission.read_bytes()
            logger.info(f"Using sample submission: {sample_submission.name}")
        else:
            submission_bytes = b"id,target\n"
            logger.info("No sample submission found, submitting fallback")

        # # 6. Validate submission before final submit
        # logger.info("Requesting validation from green agent...")
        # validation_msg = Message(
        #     kind="message",
        #     role="agent",
        #     parts=[
        #         Part(root=TextPart(text="validate")),
        #         Part(root=FilePart(
        #             file=FileWithBytes(
        #                 bytes=base64.b64encode(submission_bytes).decode('ascii'),
        #                 name="submission.csv",
        #                 mime_type="text/csv",
        #             )
        #         ))
        #     ],
        #     message_id="validation-request",
        # )
        # await updater.update_status(TaskState.working, validation_msg)

        # 7. Submit final artifact
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