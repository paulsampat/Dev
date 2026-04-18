"""
RUNTIME — run this for every conversation.

Usage:
    python agent_run.py
    python agent_run.py "Explain Black-Scholes in simple terms"
"""

import os
import sys
import anthropic
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic()

AGENT_ID = os.environ["AGENT_ID"]
ENVIRONMENT_ID = os.environ["ENVIRONMENT_ID"]


def run_session(user_message: str):
    # 1. Create a new session for this conversation
    session = client.beta.sessions.create(
        agent=AGENT_ID,                # string shorthand = latest version
        environment_id=ENVIRONMENT_ID,
    )
    print(f"Session: {session.id}\n")

    # 2. Open the stream BEFORE sending the message (stream-first ordering)
    with client.beta.sessions.events.stream(session_id=session.id) as stream:

        # 3. Send the user message
        client.beta.sessions.events.send(
            session_id=session.id,
            events=[{
                "type": "user.message",
                "content": [{"type": "text", "text": user_message}],
            }],
        )

        # 4. Process the event stream
        for event in stream:
            if event.type == "agent.message":
                for block in event.content:
                    if block.type == "text":
                        print(block.text, end="", flush=True)

            elif event.type == "session.status_idle":
                stop = event.stop_reason.type if hasattr(event, "stop_reason") else "unknown"
                if stop != "requires_action":
                    break  # agent finished

            elif event.type == "session.status_terminated":
                break

    print()  # newline after streamed response

    # 5. Download any files the agent wrote to /mnt/session/outputs/
    import time
    time.sleep(2)  # brief wait for file indexing
    files = client.beta.files.list(session_id=session.id)
    if files.data:
        os.makedirs("agent_outputs", exist_ok=True)
        print("\nDownloading files from agent:")
        for f in files.data:
            content = client.beta.files.download(f.id)
            path = os.path.join("agent_outputs", f.filename)
            with open(path, "wb") as out:
                out.write(content.read())
            print(f"  Saved: {path}")


if __name__ == "__main__":
    message = sys.argv[1] if len(sys.argv) > 1 else "Hello! What can you help me with?"
    run_session(message)
