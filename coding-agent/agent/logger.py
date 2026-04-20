"""
Phase 5: Structured file logging for the coding agent.

Writes structured logs to logs/agent.log so you have a full audit trail
of every session — tasks, tool calls, errors, and costs — without
cluttering the terminal.

Two loggers:
  agent_logger  — general agent events (session start/end, iterations, errors)
  cost_logger   — token usage and cost per iteration (easy to grep/analyse)
"""

import logging
from pathlib import Path

# ── Setup ─────────────────────────────────────────────────────────────────────

LOGS_DIR = Path(__file__).parent.parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)

LOG_FILE = LOGS_DIR / "agent.log"

_fmt = logging.Formatter(
    fmt="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

def _make_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured (e.g. on reimport)
    logger.setLevel(logging.DEBUG)
    handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    handler.setFormatter(_fmt)
    logger.addHandler(handler)
    logger.propagate = False  # don't bubble up to root logger
    return logger

agent_logger = _make_logger("agent")
cost_logger  = _make_logger("cost")


# ── Public logging functions ──────────────────────────────────────────────────

def log_session_start(session_id: str, task: str) -> None:
    agent_logger.info(f"SESSION START  id={session_id}  task={task!r}")


def log_session_end(session_id: str, total_cost: float, iterations: int) -> None:
    agent_logger.info(
        f"SESSION END    id={session_id}  "
        f"iterations={iterations}  total_cost=${total_cost:.4f}"
    )


def log_iteration(session_id: str, iteration: int, stop_reason: str) -> None:
    agent_logger.debug(
        f"ITERATION      id={session_id}  "
        f"iteration={iteration}  stop_reason={stop_reason}"
    )


def log_tool_call(session_id: str, tool_name: str, tool_input: dict) -> None:
    # Log tool name always; truncate input to keep log readable
    input_repr = str(tool_input)
    if len(input_repr) > 200:
        input_repr = input_repr[:200] + "…"
    agent_logger.debug(
        f"TOOL CALL      id={session_id}  tool={tool_name}  input={input_repr}"
    )


def log_tool_result(session_id: str, tool_name: str, result: str) -> None:
    preview = result.splitlines()[0] if result else "(empty)"
    if len(preview) > 200:
        preview = preview[:200] + "…"
    agent_logger.debug(
        f"TOOL RESULT    id={session_id}  tool={tool_name}  result={preview!r}"
    )


def log_usage(
    session_id: str,
    iteration: int,
    input_tokens: int,
    cache_write: int,
    cache_read: int,
    output_tokens: int,
    iteration_cost: float,
    total_cost: float,
) -> None:
    cost_logger.info(
        f"USAGE  id={session_id}  iter={iteration}  "
        f"in={input_tokens}  cache_write={cache_write}  "
        f"cache_read={cache_read}  out={output_tokens}  "
        f"iter_cost=${iteration_cost:.4f}  total=${total_cost:.4f}"
    )


def log_error(session_id: str, error: Exception, context: str = "") -> None:
    agent_logger.error(
        f"ERROR          id={session_id}  "
        f"context={context!r}  "
        f"type={type(error).__name__}  msg={error}"
    )


def log_warning(session_id: str, message: str) -> None:
    agent_logger.warning(f"WARNING        id={session_id}  msg={message}")
