"""
Hard controls for the coding agent.

All checks raise PermissionError on violation — the dispatcher in
tool_executor.py catches these and returns a clear error string to Claude,
which then explains the block to the user.

Two guard types:
  check_command(command)  — blocks dangerous bash commands before execution
  check_path(path)        — blocks reads/writes to sensitive files
"""

import re
from pathlib import Path

# ── Bash command blocks ───────────────────────────────────────────────────────
#
# Each entry is a (pattern, reason) tuple. The pattern is matched against the
# full command string (case-insensitive, after collapsing whitespace).
# The first match wins and raises PermissionError.

_BLOCKED_COMMAND_PATTERNS: list[tuple[str, str]] = [
    # Destructive file operations
    (r"\brm\s+.*-[a-z]*r[a-z]*f", "recursive force delete (rm -rf) is not allowed"),
    (r"\brm\s+.*-[a-z]*f[a-z]*r", "recursive force delete (rm -rf) is not allowed"),
    (r"\bmkfs\b", "filesystem formatting (mkfs) is not allowed"),
    (r"\bdd\b.*of=", "disk write via dd is not allowed"),
    (r"\bshred\b", "shred is not allowed"),

    # Git operations that affect remotes
    (r"\bgit\s+push\b", "git push is not allowed — commit locally and push manually"),
    (r"\bgit\s+.*--force\b", "force git operations are not allowed"),
    (r"\bgit\s+.*-f\b", "force git operations are not allowed"),
    (r"\bgit\s+reset\s+--hard\b", "git reset --hard is not allowed"),
    (r"\bgit\s+clean\s+.*-f\b", "git clean -f is not allowed"),
    (r"\bgit\s+rebase\b", "git rebase is not allowed"),

    # Privilege escalation
    (r"\bsudo\b", "sudo is not allowed"),
    (r"\bsu\s", "su is not allowed"),
    (r"\bdoas\b", "doas is not allowed"),

    # Network operations
    (r"\bcurl\b", "curl is not allowed — use Python requests or httpx in code instead"),
    (r"\bwget\b", "wget is not allowed"),
    (r"\bnc\b", "netcat (nc) is not allowed"),
    (r"\bncat\b", "ncat is not allowed"),
    (r"\bssh\b", "ssh is not allowed"),
    (r"\bscp\b", "scp is not allowed"),
    (r"\brsync\b", "rsync is not allowed"),
    (r"\btelnet\b", "telnet is not allowed"),

    # System administration
    (r"\bchmod\s+.*[0-9]{3,4}\b", "chmod with numeric mode is not allowed"),
    (r"\bchown\b", "chown is not allowed"),
    (r"\bkill\s+-9\b", "kill -9 is not allowed"),
    (r"\bkillall\b", "killall is not allowed"),
    (r"\bshutdown\b", "shutdown is not allowed"),
    (r"\breboot\b", "reboot is not allowed"),
    (r"\bhalt\b", "halt is not allowed"),
    (r"\bpoweroff\b", "poweroff is not allowed"),

    # Package managers installing globally
    (r"\bnpm\s+install\s+-g\b", "global npm installs are not allowed"),
    (r"\bpip\s+install\b(?!.*--user)(?!.*uv)", "pip install outside uv is not allowed — use uv add"),
    (r"\bpip3\s+install\b", "pip3 install is not allowed — use uv add"),

    # Fork bomb
    (r":\(\)\s*\{.*\|.*&\s*\}", "fork bomb pattern detected"),

    # Environment variable exfiltration
    (r"\benv\b.*>", "redirecting env output is not allowed"),
    (r"\bprintenv\b.*>", "redirecting printenv output is not allowed"),
]

# Pre-compile patterns for performance
_COMPILED_COMMAND_BLOCKS = [
    (re.compile(pattern, re.IGNORECASE), reason)
    for pattern, reason in _BLOCKED_COMMAND_PATTERNS
]


# ── Protected path patterns ───────────────────────────────────────────────────
#
# Glob patterns matched against the resolved file path string.
# Matching paths cannot be written (or in some cases read).

_PROTECTED_WRITE_PATTERNS: list[tuple[str, str]] = [
    # Secrets and credentials
    ("**/.env",           "writing .env files is not allowed"),
    ("**/.env.*",         "writing .env files is not allowed"),
    ("**/*.pem",          "writing certificate/key files is not allowed"),
    ("**/*.key",          "writing key files is not allowed"),
    ("**/*.p12",          "writing certificate files is not allowed"),
    ("**/*.pfx",          "writing certificate files is not allowed"),
    ("**/*.crt",          "writing certificate files is not allowed"),
    ("**/*.cer",          "writing certificate files is not allowed"),
    ("**/secrets/**",     "writing to secrets directories is not allowed"),
    ("**/.ssh/**",        "writing to .ssh directories is not allowed"),

    # Git internals
    ("**/.git/**",        "writing to .git internals is not allowed"),

    # Production / deployment configs
    ("**/production.*",   "writing production config files is not allowed"),
    ("**/prod.*",         "writing production config files is not allowed"),
    ("**/*.tfstate",      "writing Terraform state files is not allowed"),
    ("**/*.tfstate.*",    "writing Terraform state files is not allowed"),
]

_PROTECTED_READ_PATTERNS: list[tuple[str, str]] = [
    # Only hard-block reads on the most sensitive files
    ("**/.env",           "reading .env files is not allowed"),
    ("**/.env.*",         "reading .env files is not allowed"),
    ("**/*.pem",          "reading certificate/key files is not allowed"),
    ("**/*.key",          "reading key files is not allowed"),
    ("**/.ssh/**",        "reading .ssh files is not allowed"),
]


def _matches_any(path: Path, patterns: list[tuple[str, str]]) -> str | None:
    """Return the reason string if the path matches any pattern, else None."""
    path_str = str(path)
    for pattern, reason in patterns:
        if path.match(pattern) or Path(path_str).match(pattern):
            return reason
    return None


# ── Public API ────────────────────────────────────────────────────────────────

def check_command(command: str) -> None:
    """
    Raise PermissionError if the command matches any blocked pattern.

    Args:
        command: The shell command string about to be executed.

    Raises:
        PermissionError: With a message explaining what was blocked and why.
    """
    # Collapse whitespace so double-spaces don't evade pattern matching
    normalised = " ".join(command.split())

    for pattern, reason in _COMPILED_COMMAND_BLOCKS:
        if pattern.search(normalised):
            raise PermissionError(
                f"Blocked command: {reason}.\n"
                f"Command: {command!r}"
            )


def check_path_write(path: Path) -> None:
    """
    Raise PermissionError if the path is protected from writes.

    Args:
        path: Resolved absolute Path about to be written.

    Raises:
        PermissionError: With a message explaining what was blocked and why.
    """
    reason = _matches_any(path, _PROTECTED_WRITE_PATTERNS)
    if reason:
        raise PermissionError(
            f"Blocked write to protected path: {reason}.\n"
            f"Path: {path}"
        )


def check_path_read(path: Path) -> None:
    """
    Raise PermissionError if the path is protected from reads.

    Args:
        path: Resolved absolute Path about to be read.

    Raises:
        PermissionError: With a message explaining what was blocked and why.
    """
    reason = _matches_any(path, _PROTECTED_READ_PATTERNS)
    if reason:
        raise PermissionError(
            f"Blocked read of protected path: {reason}.\n"
            f"Path: {path}"
        )
