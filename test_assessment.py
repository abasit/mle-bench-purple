"""
Test script to run an MLE-bench assessment locally.

Usage:
    1. Start the green agent:  uv run src/server.py --port 9009
    2. Start the purple agent: uv run src/server.py --port 9010
    3. Run this script:        python test_assessment.py

You can override ports/competition:
    python test_assessment.py --green-port 9009 --purple-port 9010 --competition spaceship-titanic
"""
import argparse
import asyncio
import json

import httpx
from a2a.client import A2ACardResolver, ClientConfig, ClientFactory
from a2a.types import (
    AgentCard,
    Message,
    TaskStatusUpdateEvent,
    TaskArtifactUpdateEvent,
    TextPart,
    DataPart,
    FilePart,
)


def merge_parts(parts) -> str:
    chunks = []
    for part in parts:
        if isinstance(part.root, TextPart):
            chunks.append(part.root.text)
        elif isinstance(part.root, DataPart):
            chunks.append(json.dumps(part.root.data, indent=2))
        elif isinstance(part.root, FilePart):
            chunks.append(f"[File: {part.root.file.name}]")
    return "\n".join(chunks)


async def run_assessment(green_url: str, purple_url: str, competition_id: str):
    request = {
        "participants": {"agent": purple_url},
        "config": {"competition_id": competition_id},
    }
    request_json = json.dumps(request)

    print(f"Green agent:  {green_url}")
    print(f"Purple agent: {purple_url}")
    print(f"Competition:  {competition_id}")
    print("=" * 60)

    async with httpx.AsyncClient(timeout=3600) as httpx_client:
        resolver = A2ACardResolver(httpx_client=httpx_client, base_url=green_url)
        agent_card = await resolver.get_agent_card()
        config = ClientConfig(httpx_client=httpx_client, streaming=True)
        factory = ClientFactory(config)
        client = factory.create(agent_card)

        from a2a.types import Message, Part, Role, TextPart
        from uuid import uuid4

        msg = Message(
            kind="message",
            role=Role.user,
            parts=[Part(root=TextPart(text=request_json))],
            message_id=uuid4().hex,
        )

        async for event in client.send_message(msg):
            match event:
                case Message() as m:
                    print(f"[Message] {merge_parts(m.parts)}")

                case (task, TaskStatusUpdateEvent() as status_event):
                    state = status_event.status.state.value
                    parts_text = ""
                    if status_event.status.message:
                        parts_text = merge_parts(status_event.status.message.parts)
                    print(f"[{state}] {parts_text}")

                    if state == "completed":
                        if task.artifacts:
                            for artifact in task.artifacts:
                                print(f"[Artifact: {artifact.name}] {merge_parts(artifact.parts)}")

                    elif state not in ("submitted", "working"):
                        print(f"Assessment ended with status: {state}")
                        return

                case (task, TaskArtifactUpdateEvent() as artifact_event):
                    art = artifact_event.artifact
                    print(f"[Artifact: {art.name}] {merge_parts(art.parts)}")

                case _:
                    pass

    print("=" * 60)
    print("Assessment complete.")


def main():
    parser = argparse.ArgumentParser(description="Test MLE-bench assessment locally")
    parser.add_argument("--green-port", type=int, default=9009)
    parser.add_argument("--purple-port", type=int, default=9010)
    parser.add_argument("--competition", type=str, default="spaceship-titanic")
    args = parser.parse_args()

    green_url = f"http://localhost:{args.green_port}"
    purple_url = f"http://localhost:{args.purple_port}"

    asyncio.run(run_assessment(green_url, purple_url, args.competition))


if __name__ == "__main__":
    main()
