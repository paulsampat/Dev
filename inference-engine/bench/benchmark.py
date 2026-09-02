"""Measure what the engine's parts are actually worth.

    python -m bench.benchmark
    python -m bench.benchmark --model transformer --batch 1 8 32

Reports, per configuration, median latency and throughput, plus the numerical
drift against the reference implementation -- because a speedup that changes
the answer is not a speedup.
"""

from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from engine.api import InferenceSession
from engine.executor import ExecutionPlan
from engine.scheduler import BatchScheduler
from tools.models import BUILDERS


@dataclass
class Result:
    label: str
    median_ms: float
    p90_ms: float
    rows_per_second: float
    max_error: Optional[float] = None

    def row(self) -> str:
        error = "-" if self.max_error is None else f"{self.max_error:.2e}"
        return (
            f"{self.label:<34} {self.median_ms:>9.3f} {self.p90_ms:>9.3f} "
            f"{self.rows_per_second:>12,.0f} {error:>11}"
        )


HEADER = (
    f"{'configuration':<34} {'p50 (ms)':>9} {'p90 (ms)':>9} "
    f"{'rows/s':>12} {'max err':>11}"
)


def measure(
    fn: Callable[[], object],
    rows: int,
    label: str,
    iterations: int = 50,
    warmup: int = 5,
) -> Result:
    for _ in range(warmup):  # let the allocator pool reach steady state
        fn()

    samples = []
    for _ in range(iterations):
        started = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - started)

    samples.sort()
    median = statistics.median(samples)
    return Result(
        label=label,
        median_ms=median * 1e3,
        p90_ms=samples[int(len(samples) * 0.9) - 1] * 1e3,
        rows_per_second=rows / median,
    )


def benchmark_model(name: str, batches: list[int], iterations: int) -> None:
    demo = BUILDERS[name]()
    reference_input = demo.example_input
    feature_shape = reference_input.shape[1:]
    rng = np.random.default_rng(0)

    print(f"\n=== {name} ===")
    print(ExecutionPlan.compile(demo.graph).describe())

    for batch in batches:
        inputs = rng.standard_normal((batch,) + feature_shape).astype(np.float32)
        feeds = {demo.input_name: inputs}
        expected = demo.reference(inputs)

        print(f"\nbatch={batch}")
        print(HEADER)

        configurations = [
            ("unoptimized, no arena", dict(optimize_graph=False, use_arena=False)),
            ("unoptimized, arena", dict(optimize_graph=False, use_arena=True)),
            ("fused, no arena", dict(optimize_graph=True, use_arena=False)),
            ("fused + arena", dict(optimize_graph=True, use_arena=True)),
            ("fused + arena + int8", dict(optimize_graph=True, use_arena=True,
                                          quantize=True)),
        ]

        for label, options in configurations:
            with InferenceSession(demo.graph, **options) as session:
                output_name = session.outputs[0].name
                result = measure(
                    lambda: session.run(feeds), batch, label, iterations=iterations
                )
                output = session.run(feeds)[output_name]
                result.max_error = float(np.abs(output - expected).max())
                print(result.row())


def benchmark_batching(name: str, iterations: int = 200) -> None:
    """Single-row requests, served one at a time versus dynamically batched."""
    demo = BUILDERS[name]()
    rng = np.random.default_rng(1)
    rows = [
        rng.standard_normal((1,) + demo.example_input.shape[1:]).astype(np.float32)
        for _ in range(iterations)
    ]

    print(f"\n=== dynamic batching ({name}, {iterations} single-row requests) ===")

    with InferenceSession(demo.graph) as session:
        started = time.perf_counter()
        for row in rows:
            session.run({demo.input_name: row})
        serial = time.perf_counter() - started
        print(
            f"one at a time      {serial * 1e3:8.1f} ms total, "
            f"{iterations / serial:8,.0f} req/s"
        )

    with InferenceSession(demo.graph) as session:
        with BatchScheduler(session, max_batch_size=32, max_wait_ms=2) as scheduler:
            started = time.perf_counter()
            futures = [scheduler.submit({demo.input_name: row}) for row in rows]
            for future in futures:
                future.result(timeout=60)
            batched = time.perf_counter() - started
            print(
                f"dynamic batching   {batched * 1e3:8.1f} ms total, "
                f"{iterations / batched:8,.0f} req/s"
            )
            print(f"  {scheduler.stats}")
            print(f"  speedup: {serial / batched:.2f}x")


def profile_model(name: str) -> None:
    demo = BUILDERS[name]()
    with InferenceSession(demo.graph) as session:
        feeds = {demo.input_name: demo.example_input}
        for _ in range(20):
            session.run(feeds, profile=True)
        print(f"\n=== per-node profile ({name}) ===")
        print(session.plan.profile_table())
        print(f"\narena: {session.arena_stats}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(BUILDERS) + ["all"], default="all")
    parser.add_argument("--batch", type=int, nargs="+", default=[1, 8, 64])
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--skip-batching", action="store_true")
    args = parser.parse_args(argv)

    names = sorted(BUILDERS) if args.model == "all" else [args.model]
    for name in names:
        benchmark_model(name, args.batch, args.iterations)
        profile_model(name)
        if not args.skip_batching:
            benchmark_batching(name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
