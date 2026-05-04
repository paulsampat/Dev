# Resilience Rules

All code generated must follow these resilience patterns:

## HTTP calls
- Every HTTP call (requests, httpx, aiohttp, urllib) must have an explicit timeout parameter
- Never use a bare `requests.get(url)` — always `requests.get(url, timeout=30)`

## Error handling
- Never use a bare `except:` clause — always catch specific exceptions (e.g. `except ValueError:`)
- Never silently swallow exceptions with an empty `except` block containing only `pass`
- Always log or re-raise caught exceptions — silent failure is not acceptable

## Retries
- Any function that calls an external service must either implement retry logic
  or explicitly document in a comment why retries are not needed

## Timeouts
- Long-running operations must have a timeout mechanism
- Background threads or tasks must have a defined maximum lifetime
