"""Dynamic batching for serving.

A model that takes 2 ms on one row usually takes barely more on eight, because
the cost is dominated by streaming weights out of memory rather than by the
arithmetic. Under concurrent traffic, running requests one at a time therefore
wastes most of the machine.

This scheduler collects requests arriving close together into a single batched
call. The tuning knob is the pair (``max_batch_size``, ``max_wait_ms``): the
first bounds how large a batch can get, the second bounds how long an early
request will wait for company. Setting ``max_wait_ms`` too high trades tail
latency for throughput.
"""

from __future__ import annotations

import queue
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .api import InferenceSession


@dataclass
class _Request:
    feeds: dict[str, np.ndarray]
    rows: int
    future: Future


@dataclass
class SchedulerStats:
    requests: int = 0
    batches: int = 0
    rows: int = 0
    wait_seconds: float = 0.0
    run_seconds: float = 0.0

    @property
    def mean_batch_size(self) -> float:
        return self.rows / self.batches if self.batches else 0.0

    def __str__(self) -> str:
        return (
            f"{self.requests} requests in {self.batches} batches "
            f"(mean batch {self.mean_batch_size:.2f}), "
            f"{self.run_seconds * 1e3:.1f} ms compute, "
            f"{self.wait_seconds * 1e3:.1f} ms batching delay"
        )


class BatchScheduler:
    """Serialises requests onto one session, batching what arrives together."""

    def __init__(
        self,
        session: InferenceSession,
        max_batch_size: int = 8,
        max_wait_ms: float = 5.0,
        batch_axis: int = 0,
    ):
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be >= 1")
        if batch_axis != 0:
            raise ValueError(
                "only batch_axis=0 is supported; models with the batch "
                "dimension elsewhere need a transpose at the graph boundary"
            )
        self.session = session
        self.max_batch_size = max_batch_size
        self.max_wait = max_wait_ms / 1e3
        self.stats = SchedulerStats()

        self._queue: queue.Queue[Optional[_Request]] = queue.Queue()
        self._worker: Optional[threading.Thread] = None
        self._running = False

    # -- lifecycle -------------------------------------------------------

    def start(self) -> "BatchScheduler":
        if self._running:
            return self
        self._running = True
        self._worker = threading.Thread(
            target=self._loop, name="inference-batcher", daemon=True
        )
        self._worker.start()
        return self

    def stop(self, timeout: Optional[float] = 5.0) -> None:
        if not self._running:
            return
        self._running = False
        self._queue.put(None)  # wake the worker so it can exit
        if self._worker is not None:
            self._worker.join(timeout)
            self._worker = None

    def __enter__(self) -> "BatchScheduler":
        return self.start()

    def __exit__(self, *exc_info) -> None:
        self.stop()

    # -- submission ------------------------------------------------------

    def submit(self, feeds: dict[str, np.ndarray]) -> Future:
        if not self._running:
            raise RuntimeError("scheduler is not running; call start() first")

        rows = {int(np.asarray(v).shape[0]) for v in feeds.values()}
        if len(rows) != 1:
            raise ValueError(
                "inputs disagree on batch size along axis 0: "
                + ", ".join(f"{k}={np.asarray(v).shape}" for k, v in feeds.items())
            )

        future: Future = Future()
        self._queue.put(_Request(feeds=feeds, rows=rows.pop(), future=future))
        return future

    def run(self, feeds: dict[str, np.ndarray], timeout: Optional[float] = None):
        """Blocking convenience wrapper around :meth:`submit`."""
        return self.submit(feeds).result(timeout)

    # -- worker ----------------------------------------------------------

    def _collect(self) -> list[_Request]:
        """Block for one request, then gather whatever else arrives in time."""
        first = self._queue.get()
        if first is None:
            return []

        batch = [first]
        deadline = time.perf_counter() + self.max_wait
        rows = first.rows
        while rows < self.max_batch_size:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                break
            try:
                nxt = self._queue.get(timeout=remaining)
            except queue.Empty:
                break
            if nxt is None:
                self._running = False
                break
            if rows + nxt.rows > self.max_batch_size:
                # Would overflow the batch; run what we have and keep this one
                # at the head of the queue for the next round.
                self._queue.put(nxt)
                break
            batch.append(nxt)
            rows += nxt.rows
        return batch

    def _loop(self) -> None:
        while self._running:
            started_waiting = time.perf_counter()
            batch = self._collect()
            if not batch:
                continue
            self.stats.wait_seconds += time.perf_counter() - started_waiting

            try:
                self._execute(batch)
            except Exception as exc:  # noqa: BLE001 - propagate to every caller
                for request in batch:
                    if not request.future.done():
                        request.future.set_exception(exc)

    def _execute(self, batch: list[_Request]) -> None:
        names = set(batch[0].feeds)
        for request in batch[1:]:
            if set(request.feeds) != names:
                raise ValueError(
                    "batched requests must feed the same inputs; got "
                    f"{sorted(names)} and {sorted(request.feeds)}"
                )

        if len(batch) == 1:
            merged = batch[0].feeds
        else:
            merged = {
                name: np.concatenate([r.feeds[name] for r in batch], axis=0)
                for name in names
            }

        started = time.perf_counter()
        outputs = self.session.run(merged)
        self.stats.run_seconds += time.perf_counter() - started
        self.stats.batches += 1
        self.stats.requests += len(batch)
        self.stats.rows += sum(r.rows for r in batch)

        total_rows = sum(r.rows for r in batch)
        for name, array in outputs.items():
            if array.shape[0] != total_rows:
                raise ValueError(
                    f"output {name!r} has leading dimension {array.shape[0]} "
                    f"but the batch holds {total_rows} rows; this model cannot "
                    f"be batched along axis 0"
                )

        offset = 0
        for request in batch:
            stop = offset + request.rows
            request.future.set_result(
                {name: array[offset:stop] for name, array in outputs.items()}
            )
            offset = stop
