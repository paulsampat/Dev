"""
Phase 2: Tool implementations for the coding agent.

Each function here corresponds to a tool defined in tools.py.
The dispatch() function at the bottom routes Claude's tool_use
requests to the right implementation and returns a result string.
"""

import fnmatch
import os
import re
import subprocess
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────

# Default working directory for the agent. Tools resolve relative paths
# against this. Override by passing working_dir to run_bash, or root to
# search_files / grep_files.
DEFAULT_WORKING_DIR = Path.cwd()

# Hard cap on how many bytes read_file will return. Prevents accidentally
# dumping a huge binary or generated file into Claude's context window.
MAX_READ_BYTES = 100_000  # 100 KB

# Hard cap on bash output returned to Claude.
MAX_OUTPUT_BYTES = 50_000  # 50 KB


# ── Helpers ───────────────────────────────────────────────────────────────────

def _resolve(path: str, base: Path = DEFAULT_WORKING_DIR) -> Path:
    """Resolve a path relative to base, preventing directory traversal."""
    resolved = (base / path).resolve()
    # Ensure the resolved path is still under the base directory.
    # This prevents Claude from accidentally reading /etc/passwd etc.
    try:
        resolved.relative_to(base.resolve())
    except ValueError:
        raise PermissionError(
            f"Path '{path}' resolves outside the working directory '{base}'. "
            "Access denied."
        )
    return resolved


def _truncate(text: str, max_bytes: int, label: str = "output") -> str:
    """Return text truncated to max_bytes with a notice if truncated."""
    encoded = text.encode("utf-8", errors="replace")
    if len(encoded) <= max_bytes:
        return text
    truncated = encoded[:max_bytes].decode("utf-8", errors="replace")
    return truncated + f"\n\n[{label} truncated at {max_bytes} bytes]"


# ── Tool implementations ───────────────────────────────────────────────────────

def read_file(path: str, start_line: int | None = None, end_line: int | None = None) -> str:
    """Read a file, optionally returning only a slice of lines."""
    resolved = _resolve(path)

    if not resolved.exists():
        return f"Error: file not found: {path}"
    if not resolved.is_file():
        return f"Error: '{path}' is a directory, not a file."

    try:
        content = resolved.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        return f"Error reading file: {e}"

    if start_line is not None or end_line is not None:
        lines = content.splitlines(keepends=True)
        # Convert from 1-based (user-facing) to 0-based (Python list)
        lo = (start_line - 1) if start_line is not None else 0
        hi = end_line if end_line is not None else len(lines)
        lo = max(0, lo)
        hi = min(len(lines), hi)
        content = "".join(lines[lo:hi])
        header = f"[Lines {lo + 1}–{hi} of {len(lines)} in {path}]\n"
        content = header + content

    return _truncate(content, MAX_READ_BYTES, label="file content")


def write_file(path: str, content: str) -> str:
    """Write content to a file, creating parent directories as needed."""
    resolved = _resolve(path)

    try:
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content, encoding="utf-8")
    except OSError as e:
        return f"Error writing file: {e}"

    lines = content.count("\n") + (1 if content and not content.endswith("\n") else 0)
    return f"Written {lines} lines to {path}"


def str_replace_file(path: str, old_string: str, new_string: str) -> str:
    """Replace an exact string in a file."""
    resolved = _resolve(path)

    if not resolved.exists():
        return f"Error: file not found: {path}"

    try:
        content = resolved.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        return f"Error reading file: {e}"

    count = content.count(old_string)
    if count == 0:
        return (
            f"Error: old_string not found in {path}.\n"
            "Check that whitespace and indentation match exactly."
        )
    if count > 1:
        return (
            f"Error: old_string appears {count} times in {path}. "
            "Provide more surrounding context to make it unique."
        )

    new_content = content.replace(old_string, new_string, 1)
    try:
        resolved.write_text(new_content, encoding="utf-8")
    except OSError as e:
        return f"Error writing file: {e}"

    return f"Replaced 1 occurrence in {path}"


def run_bash(command: str, working_dir: str | None = None, timeout: int = 30) -> str:
    """Run a shell command and return combined stdout + stderr."""
    cwd = Path(working_dir).resolve() if working_dir else DEFAULT_WORKING_DIR

    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return f"Error: command timed out after {timeout}s.\nCommand: {command}"
    except OSError as e:
        return f"Error running command: {e}"

    output_parts = []
    if result.stdout:
        output_parts.append(result.stdout)
    if result.stderr:
        output_parts.append(f"[stderr]\n{result.stderr}")

    combined = "\n".join(output_parts).strip() or "(no output)"
    combined = _truncate(combined, MAX_OUTPUT_BYTES, label="command output")

    return f"[exit code: {result.returncode}]\n{combined}"


def search_files(pattern: str, root: str | None = None) -> str:
    """Find files matching a glob pattern."""
    base = Path(root).resolve() if root else DEFAULT_WORKING_DIR

    if not base.exists():
        return f"Error: directory not found: {root}"

    matches = sorted(str(p.relative_to(base)) for p in base.glob(pattern) if p.is_file())

    if not matches:
        return f"No files found matching '{pattern}' under {base}"

    result = f"Found {len(matches)} file(s) matching '{pattern}':\n"
    result += "\n".join(matches)
    return _truncate(result, MAX_OUTPUT_BYTES, label="search results")


def grep_files(
    pattern: str,
    root: str | None = None,
    file_pattern: str | None = None,
    case_sensitive: bool = True,
    max_results: int = 50,
) -> str:
    """Search file contents for lines matching a regex."""
    base = Path(root).resolve() if root else DEFAULT_WORKING_DIR

    if not base.exists():
        return f"Error: directory not found: {root}"

    flags = 0 if case_sensitive else re.IGNORECASE
    try:
        regex = re.compile(pattern, flags)
    except re.error as e:
        return f"Error: invalid regex pattern '{pattern}': {e}"

    matches = []
    files_searched = 0

    for filepath in sorted(base.rglob("*")):
        if not filepath.is_file():
            continue
        if file_pattern and not fnmatch.fnmatch(filepath.name, file_pattern):
            continue
        # Skip binary-looking files
        try:
            text = filepath.read_text(encoding="utf-8", errors="strict")
        except (UnicodeDecodeError, OSError):
            continue

        files_searched += 1
        for lineno, line in enumerate(text.splitlines(), start=1):
            if regex.search(line):
                rel = filepath.relative_to(base)
                matches.append(f"{rel}:{lineno}: {line.rstrip()}")
                if len(matches) >= max_results:
                    break
        if len(matches) >= max_results:
            break

    if not matches:
        return f"No matches for '{pattern}' in {files_searched} file(s) under {base}"

    header = f"Found {len(matches)} match(es) across {files_searched} file(s):\n"
    result = header + "\n".join(matches)
    if len(matches) == max_results:
        result += f"\n\n[Results capped at {max_results}. Refine your pattern or use file_pattern to narrow the search.]"

    return result


# ── Dispatcher ────────────────────────────────────────────────────────────────

def dispatch(tool_name: str, tool_input: dict) -> str:
    """
    Route a tool_use request from Claude to the correct implementation.

    Returns a string result to be sent back as a tool_result message.
    Any unhandled exception is caught and returned as an error string
    so the agent loop stays alive.
    """
    try:
        match tool_name:
            case "read_file":
                return read_file(**tool_input)
            case "write_file":
                return write_file(**tool_input)
            case "str_replace_file":
                return str_replace_file(**tool_input)
            case "run_bash":
                return run_bash(**tool_input)
            case "search_files":
                return search_files(**tool_input)
            case "grep_files":
                return grep_files(**tool_input)
            case _:
                return f"Error: unknown tool '{tool_name}'"
    except PermissionError as e:
        return f"Permission denied: {e}"
    except TypeError as e:
        return f"Error: invalid arguments for tool '{tool_name}': {e}"
    except Exception as e:
        return f"Unexpected error in tool '{tool_name}': {type(e).__name__}: {e}"
