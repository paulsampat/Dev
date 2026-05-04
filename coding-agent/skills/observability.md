# Observability Skill

**Purpose:** Guide teams through implementing structured logging, metrics, distributed tracing, health endpoints, and alerting for any service.
**Use when:** Building a new service, reviewing an existing service for observability gaps, or preparing for a production launch.
**Invoke with:** `--skill observability` or `/observability`

---

Observability must be built in from day one — not retrofitted. In any service or distributed system, you cannot `console.log` your way to understanding what went wrong in production. Apply these standards to every service you build or review.

Observability has three pillars: structured logging, metrics, and distributed tracing. All three are required. A system with only one or two is not observable.

---

## Pillar 1 — Structured Logging

Every log entry must be structured (JSON), not free-form text. A log line that is human-readable but machine-unreadable cannot be queried, aggregated, or alerted on at scale.

**Every log entry must include:**
- `timestamp` — ISO 8601, UTC
- `service` — the name of the service emitting the log
- `severity` — one of: DEBUG, INFO, WARN, ERROR
- `trace_id` — the distributed trace ID for the current request (propagated from upstream)
- `span_id` — the current span within the trace
- Relevant business context — e.g. `order_id`, `user_id`, `book_id`, `ticker` — whatever identifies the unit of work

**Log levels — use them correctly:**
- `DEBUG` — detailed internals, enabled only in development or on-demand in production
- `INFO` — business events that always happen in normal operation (request received, order placed, job completed)
- `WARN` — something degraded but handled (retry attempt, fallback activated, slow response)
- `ERROR` — a failure that requires attention (exception thrown, downstream call failed after retries, data inconsistency)

**Never log:**
- Secrets, API keys, passwords, PII (names, emails, account numbers) in plain text
- Unstructured stack traces as a top-level log entry — attach them as a structured field

**Centralised aggregation:**
Ship all logs to a centralised system. Common choices:
- Elasticsearch + Kibana (ELK/OpenSearch)
- Datadog Logs
- Grafana Loki
- AWS CloudWatch Logs

Logs that only exist on a pod that has since been recycled are useless.

---

## Pillar 2 — Metrics

Define and emit metrics for every service from day one. Metrics are the fastest signal that something is wrong — before you know *what* is wrong.

### The RED Method (apply to every service endpoint)

- **Rate** — requests per second (throughput)
- **Errors** — error rate, broken down by 4xx (client errors) and 5xx (server/service errors) separately
- **Duration** — latency distribution: p50, p95, p99. Never use average — it hides tail latency. A p99 of 10 seconds means 1 in 100 users is waiting 10 seconds.

### Resource metrics (collect for every instance)
- CPU utilisation
- Memory utilisation
- Connection pool usage (DB connections, HTTP client connections)
- Queue depth (messages pending, jobs in queue)
- Disk I/O if relevant

### Business metrics
These are often the most actionable. Define domain-specific metrics that reflect what the system is supposed to do:
- Orders placed per minute
- P&L calculations completed per second
- Positions ingested per market session
- Failed valuations per book

A spike in technical error rate tells you something is broken. A drop in "P&L calculations per second" tells you the business is impacted — and you should know this before your users do.

### Exposure and collection
- Expose a `GET /metrics` endpoint in Prometheus format, or push to your metrics backend
- Common backends: Prometheus + Grafana, Datadog, Dynatrace, AWS CloudWatch

### Alerting
- Alert on **symptoms** (user-facing error rate elevated, p99 latency breached SLO) — not causes (CPU is high)
- Every alert must be **actionable** — if the on-call engineer cannot do anything when it fires, it should not be an alert
- Define **SLOs** (Service Level Objectives) — e.g. "99.9% of requests complete within 500ms" — and alert when you are burning through your error budget at an unsustainable rate
- Avoid alert fatigue: too many low-signal alerts train engineers to ignore them

---

## Pillar 3 — Distributed Tracing

Tracing is the most important observability tool for any system with multiple services or async processing. It lets you follow a single request or unit of work across every service, queue, and database it touches.

**Every service must:**
1. On every incoming request, check for an existing trace ID in the request headers (`traceparent` per W3C Trace Context standard, or `X-Trace-ID`)
2. If no trace ID exists, generate one (this is the trace root)
3. Propagate the trace ID in **all outbound calls** — HTTP headers, message queue metadata, async job payloads
4. Create a **span** for every significant operation:
   - Inbound HTTP request handling
   - Every outbound service call
   - Every database query
   - Every cache read/write
   - Every message published or consumed
5. Emit completed spans to a tracing backend

**Use OpenTelemetry** as the instrumentation standard. It is vendor-neutral, widely supported, and means you can swap backends without re-instrumenting your code. Supported backends include Jaeger, Zipkin, Grafana Tempo, Datadog APM, AWS X-Ray.

**What tracing gives you:**
- End-to-end latency breakdown — which service or database is responsible for the slow p99
- Dependency maps — which services call which, and how often
- Error propagation — which upstream call caused a downstream failure
- Async flow visibility — trace a message from producer through queue to consumer

---

## Health Endpoints

Every service must expose health endpoints. These are not optional — they are required for Kubernetes and any load balancer to function correctly.

**`GET /health/live`** — Liveness probe
- Returns `200 OK` if the process is running and not deadlocked
- Should be extremely fast and never block
- Used by Kubernetes to decide whether to restart the pod
- Do NOT check downstream dependencies here — a database being down should not cause the pod to restart

**`GET /health/ready`** — Readiness probe
- Returns `200 OK` only when the service is ready to accept traffic
- Should check: database connection pool is healthy, required downstream dependencies are reachable, any startup initialisation is complete
- Used by Kubernetes to decide whether to send traffic to the pod
- A pod that fails its readiness probe is removed from the load balancer but not restarted

**Response format:**
```json
{
  "status": "ok",
  "checks": {
    "database": "ok",
    "cache": "ok",
    "downstream_service": "degraded"
  }
}
```

---

## Pre-Production Observability Checklist

Before any service goes to production, verify all of the following:

**Logging**
- [ ] All log entries are structured JSON
- [ ] Every log entry includes trace_id and service name
- [ ] No secrets or PII in logs
- [ ] Logs are shipping to centralised aggregation
- [ ] Log levels are correctly applied (not everything at INFO or ERROR)

**Metrics**
- [ ] RED metrics (Rate, Errors, Duration) instrumented for every endpoint
- [ ] Business metrics defined and emitting
- [ ] `/metrics` endpoint exposed or push configured
- [ ] Dashboards created for the service
- [ ] Alerts defined for error rate and p99 latency SLO breaches

**Tracing**
- [ ] OpenTelemetry SDK instrumented
- [ ] Trace ID propagated on all inbound and outbound calls
- [ ] Spans created for DB queries, cache ops, and outbound service calls
- [ ] Traces visible in tracing backend

**Health**
- [ ] `GET /health/live` implemented and wired to Kubernetes liveness probe
- [ ] `GET /health/ready` implemented, checks real dependencies, wired to readiness probe

**Runbook**
- [ ] A runbook exists describing: how to deploy, how to roll back, what the top 3 alerts mean and how to respond
