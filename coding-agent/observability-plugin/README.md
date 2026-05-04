# Observability Plugin for Claude Code

A Claude Code plugin that guides teams through implementing production-grade observability — structured logging, metrics, distributed tracing, health endpoints, and alerting — for any service.

## What it covers

- **Structured logging** — JSON log format, required fields, log levels, centralised aggregation
- **Metrics (RED method)** — Rate, Errors, Duration for every endpoint; resource and business metrics
- **Distributed tracing** — OpenTelemetry instrumentation, trace propagation, span creation
- **Health endpoints** — `/health/live` (liveness) and `/health/ready` (readiness) with correct semantics
- **Alerting** — SLO-based alerting, avoiding alert fatigue, making every alert actionable
- **Pre-production checklist** — full checklist to verify before any service goes to production

## Installation

```bash
claude plugins install path/to/observability-plugin
```

Or from a Git URL once published:

```bash
claude plugins install https://github.com/paulsampat/observability-plugin
```

## Usage

Once installed, invoke the skill from any Claude Code session:

```
/observability:observability
```

Claude will walk you through the three pillars of observability — logging, metrics, and tracing — with concrete implementation guidance tailored to your service.

### Example prompts

```
/observability:observability
Build me a FastAPI service with full observability

/observability:observability
Review my service for observability gaps before we go to production

/observability:observability
How should I instrument my message consumer for distributed tracing?
```

## Skill: `observability`

| Field       | Value                                                                                           |
|-------------|-----------------------------------------------------------------------------------------------|
| Invoke with | `/observability:observability`                                                                  |
| Best for    | Building a new service, reviewing an existing service, preparing for a production launch        |
| Standards   | OpenTelemetry, W3C Trace Context, Prometheus metrics format, Kubernetes health probe semantics |

## Plugin manifest

```json
{
  "name": "observability",
  "version": "1.0.0",
  "author": { "name": "Paul Sampat" },
  "license": "MIT"
}
```

## License

MIT
