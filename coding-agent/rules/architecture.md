# Architecture Rules

All code generated must follow these architectural patterns:

## Separation of concerns
- Business logic must not live inside API route handlers or CLI entry points
- Route handlers and CLI commands should only: validate input, call a service function, return a response
- Database queries must not appear directly in route handlers — use a repository or service layer

## Dependencies
- Do not import from a higher layer into a lower layer (e.g. a utility module must not import from a route module)
- Avoid circular imports — if two modules need each other, extract shared code into a third module

## Configuration
- Never hardcode secrets, URLs, ports, or environment-specific values in source code
- All configuration must come from environment variables or a config object loaded at startup

## Type safety
- All function signatures must have type hints on parameters and return type
- Do not use `Any` as a type hint unless absolutely unavoidable, and add a comment explaining why
