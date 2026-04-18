"""
Phase 5: Conversation persistence for the coding agent.

Saves and loads session state (message history, task, metadata) as JSON
files under sessions/. This lets you:
  - Resume a session that was interrupted
  - Review what the agent did in a past session
  - Debug by replaying a session's message history

Each session is stored as:
  sessions/<session_id>.json

Message content blocks (TextBlock, ToolUseBlock, etc.) are Pydantic models
from the Anthropic SDK. We serialise them to plain dicts for JSON storage.
The SDK accepts plain dicts when loading back — no reconstruction needed.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

SESSIONS_DIR = Path(__file__).parent.parent / "sessions"
SESSIONS_DIR.mkdir(exist_ok=True)


# ── Serialisation helpers ─────────────────────────────────────────────────────

def _serialise_content(content) -> list[dict] | str:
    """
    Convert a message's content to a JSON-serialisable form.

    content can be:
      - a plain string  (user task message)
      - a list of SDK content block objects (TextBlock, ToolUseBlock, etc.)
      - a list of plain dicts (tool_result messages we built ourselves)
    """
    if isinstance(content, str):
        return content

    result = []
    for block in content:
        if isinstance(block, dict):
            result.append(block)
        elif hasattr(block, "model_dump"):
            # Anthropic SDK Pydantic model — serialise to dict
            result.append(block.model_dump())
        else:
            result.append(str(block))
    return result


def _serialise_messages(messages: list[dict]) -> list[dict]:
    """Serialise a full message list for JSON storage."""
    return [
        {"role": msg["role"], "content": _serialise_content(msg["content"])}
        for msg in messages
    ]


# ── Public API ────────────────────────────────────────────────────────────────

def save_session(
    session_id: str,
    task: str,
    messages: list[dict],
    total_cost: float,
    iterations: int,
    status: str = "completed",
) -> Path:
    """
    Save a session to disk.

    Args:
        session_id : Unique session identifier.
        task       : The original task string.
        messages   : Full message history (may contain SDK objects).
        total_cost : Total cost accumulated during the session.
        iterations : Number of loop iterations completed.
        status     : 'completed', 'interrupted', or 'error'.

    Returns:
        Path to the saved session file.
    """
    path = SESSIONS_DIR / f"{session_id}.json"

    payload = {
        "session_id": session_id,
        "task": task,
        "status": status,
        "total_cost": round(total_cost, 6),
        "iterations": iterations,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "messages": _serialise_messages(messages),
    }

    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_session(session_id: str) -> dict:
    """
    Load a saved session from disk.

    Args:
        session_id: The session ID to load.

    Returns:
        The session dict with keys: session_id, task, status,
        total_cost, iterations, saved_at, messages.

    Raises:
        FileNotFoundError if the session does not exist.
    """
    path = SESSIONS_DIR / f"{session_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Session not found: {session_id}")
    return json.loads(path.read_text(encoding="utf-8"))


def list_sessions() -> list[dict]:
    """
    List all saved sessions, most recent first.

    Returns:
        List of session summary dicts (session_id, task, status,
        total_cost, iterations, saved_at).
    """
    sessions = []
    for path in sorted(SESSIONS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            sessions.append({
                "session_id": data.get("session_id"),
                "task":       data.get("task"),
                "status":     data.get("status"),
                "total_cost": data.get("total_cost"),
                "iterations": data.get("iterations"),
                "saved_at":   data.get("saved_at"),
            })
        except (json.JSONDecodeError, KeyError):
            continue
    return sessions
