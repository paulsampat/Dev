"""
Phase 1: Tool definitions for the coding agent.

These are the JSON schemas Claude uses to understand what tools are available,
what each tool does, and what parameters it expects. Nothing executes here —
tool implementations come in Phase 2.

For a coding agent we need:
  - read_file    : read the contents of a file
  - write_file   : create or overwrite a file
  - run_bash     : execute a shell command
  - search_files : find files by glob pattern
  - grep_files   : search file contents by regex
"""

TOOLS = [
    {
        "name": "read_file",
        "description": (
            "Read the contents of a file at a given path. "
            "Use this to inspect source code, configs, or any text file. "
            "Returns the file contents as a string."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute or relative path to the file to read.",
                },
                "start_line": {
                    "type": "integer",
                    "description": (
                        "Optional. 1-based line number to start reading from. "
                        "Use with end_line to read a slice of a large file."
                    ),
                },
                "end_line": {
                    "type": "integer",
                    "description": (
                        "Optional. 1-based line number to stop reading at (inclusive). "
                        "Use with start_line to read a slice of a large file."
                    ),
                },
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": (
            "Write content to a file, creating it (and any missing parent directories) "
            "if it does not exist, or overwriting it if it does. "
            "Use this to create new files or make complete rewrites. "
            "For targeted edits to an existing file, prefer str_replace_file."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute or relative path where the file should be written.",
                },
                "content": {
                    "type": "string",
                    "description": "The full content to write to the file.",
                },
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "str_replace_file",
        "description": (
            "Replace an exact string in a file with new content. "
            "Use this for targeted edits — changing a function, fixing a bug, "
            "updating a value — without rewriting the whole file. "
            "The old_string must match exactly (including whitespace and indentation). "
            "Fails if old_string is not found, or if it appears more than once."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to edit.",
                },
                "old_string": {
                    "type": "string",
                    "description": "The exact string to search for and replace.",
                },
                "new_string": {
                    "type": "string",
                    "description": "The string to replace it with.",
                },
            },
            "required": ["path", "old_string", "new_string"],
        },
    },
    {
        "name": "run_bash",
        "description": (
            "Execute a shell command and return its stdout, stderr, and exit code. "
            "Use this to run tests, install packages, compile code, check git status, "
            "or perform any system operation. "
            "Prefer non-interactive commands. Avoid commands that run indefinitely."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute.",
                },
                "working_dir": {
                    "type": "string",
                    "description": (
                        "Optional. Directory to run the command in. "
                        "Defaults to the agent's working directory."
                    ),
                },
                "timeout": {
                    "type": "integer",
                    "description": (
                        "Optional. Maximum seconds to wait before killing the command. "
                        "Defaults to 30."
                    ),
                },
            },
            "required": ["command"],
        },
    },
    {
        "name": "search_files",
        "description": (
            "Find files matching a glob pattern under a given directory. "
            "Use this to discover project structure, locate files by name or extension, "
            "or check if a file exists. "
            "Returns a list of matching file paths."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": (
                        "Glob pattern to match against. "
                        "Examples: '**/*.py', 'src/**/*.ts', 'tests/test_*.py'."
                    ),
                },
                "root": {
                    "type": "string",
                    "description": (
                        "Optional. Directory to search under. "
                        "Defaults to the agent's working directory."
                    ),
                },
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "lint_files",
        "description": (
            "Run the ruff linter on a file or directory and return any violations found. "
            "Use this after writing or editing code to check for errors, style issues, and "
            "unused imports before considering a task complete. "
            "Set fix=true to automatically fix any auto-fixable violations in place."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File or directory to lint. Defaults to the working directory.",
                },
                "fix": {
                    "type": "boolean",
                    "description": "Optional. If true, auto-fix violations where possible. Defaults to false.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "grep_files",
        "description": (
            "Search the contents of files for lines matching a regex pattern. "
            "Use this to find where a function is defined, where a variable is used, "
            "or where a string appears across a codebase. "
            "Returns matching lines with their file path and line number."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Regex pattern to search for.",
                },
                "root": {
                    "type": "string",
                    "description": (
                        "Optional. Directory to search under. "
                        "Defaults to the agent's working directory."
                    ),
                },
                "file_pattern": {
                    "type": "string",
                    "description": (
                        "Optional. Glob pattern to restrict which files are searched. "
                        "Example: '*.py' to search only Python files."
                    ),
                },
                "case_sensitive": {
                    "type": "boolean",
                    "description": "Optional. Whether the search is case-sensitive. Defaults to true.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Optional. Maximum number of matching lines to return. Defaults to 50.",
                },
            },
            "required": ["pattern"],
        },
    },
]

# Tool names as a set — used in Phase 3 to dispatch calls
TOOL_NAMES = {tool["name"] for tool in TOOLS}
