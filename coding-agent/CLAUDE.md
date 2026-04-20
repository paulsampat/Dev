# Coding Agent — Project Guidelines

## Code style
- Always use type hints on function signatures
- Use dataclasses over plain dicts for structured data
- Keep functions small and single-purpose

## Testing
- Tests go in tests/ using pytest
- Every new function should have at least one test

## Dependencies
- Use uv to manage dependencies (pyproject.toml is the source of truth)
- Do not add dependencies unless necessary

## Project structure
- Agent logic lives in agent/
- Tool definitions and implementations live in tools/
- Design notes go in design/ as .txt files
- When generating new code based on the task I am given, create a new directory under generated_code directory with a suitable short name and put all generate code in that directory

