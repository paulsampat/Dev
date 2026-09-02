"""Execution planning and the run loop.

Building a plan once and reusing it across calls is what separates an engine
from a for-loop over ops. The plan precomputes, per model:

* the topological order of nodes,
* the resolved kernel for each node (no registry lookup per call),
* each value's *last use*, so buffers can be recycled the instant they die.

The liveness analysis is what makes the arena useful: peak memory tracks the
widest point of the graph rather than the sum of every intermediate.

Aliasing is handled explicitly. ``Reshape`` and ``Transpose`` return views of
their input, so releasing that input's buffer would corrupt the view. Before
returning a buffer to the pool, the run loop checks whether any still-live
value is a view into it, and if so transfers ownership instead of freeing.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from . import ops
from .allocator import Arena, Block
from .backends.base import Backend
from .backends.numpy_backend import NumpyBackend
from .graph import Graph, Node
from .tensor import format_shape

def _root_buffer(array: np.ndarray) -> Any:
    """Walk the ``.base`` chain to the object that actually owns the memory."""
    obj: Any = array
    while getattr(obj, "base", None) is not None:
        obj = obj.base
    return obj


@dataclass
class Step:
    node: Node
    spec: ops.OpSpec
    # Inputs that die after this step and can be returned to the arena.
    frees: tuple[str, ...] = ()


@dataclass
class ProfileEntry:
    op_type: str
    node: str
    calls: int = 0
    seconds: float = 0.0

    @property
    def ms_per_call(self) -> float:
        return (self.seconds / self.calls * 1e3) if self.calls else 0.0


@dataclass
class ExecutionPlan:
    """A compiled, reusable execution schedule for one graph."""

    graph: Graph
    backend: Backend
    steps: list[Step]
    use_arena: bool = True
    arena: Arena = field(default_factory=Arena)
    profile: dict[str, ProfileEntry] = field(default_factory=dict)

    # -- construction ----------------------------------------------------

    @staticmethod
    def compile(
        graph: Graph,
        backend: Optional[Backend] = None,
        use_arena: bool = True,
    ) -> "ExecutionPlan":
        graph.validate()
        backend = backend or NumpyBackend()
        ordered = graph.topological_order()

        # Last use of each value, by step index.
        last_use: dict[str, int] = {}
        for index, node in enumerate(ordered):
            for value in node.inputs:
                if value:
                    last_use[value] = index

        never_free = set(graph.output_names()) | set(graph.initializers)

        steps: list[Step] = []
        for index, node in enumerate(ordered):
            spec = ops.get(node.op_type)
            spec.check_arity(len(node.inputs), node.name)
            frees = tuple(
                value
                for value in dict.fromkeys(node.inputs)
                if value and last_use.get(value) == index and value not in never_free
            )
            steps.append(Step(node=node, spec=spec, frees=frees))

        return ExecutionPlan(
            graph=graph, backend=backend, steps=steps, use_arena=use_arena
        )

    # -- running ---------------------------------------------------------

    def _bind_feeds(self, feeds: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        expected = {spec.name: spec for spec in self.graph.inputs}
        missing = [name for name in expected if name not in feeds]
        if missing:
            raise ValueError(
                f"missing input(s): {', '.join(sorted(missing))}; expected "
                + ", ".join(
                    f"{s.name}{format_shape(s.shape)}" for s in self.graph.inputs
                )
            )
        unexpected = [name for name in feeds if name not in expected]
        if unexpected:
            raise ValueError(
                f"unexpected input(s): {', '.join(sorted(unexpected))}; this "
                f"model takes {', '.join(sorted(expected))}"
            )

        env: dict[str, np.ndarray] = dict(self.graph.initializers)
        for name, spec in expected.items():
            array = np.ascontiguousarray(feeds[name])
            problem = spec.matches(array)
            if problem:
                raise ValueError(problem)
            env[name] = self.backend.to_device(array)
        return env

    def run(
        self, feeds: dict[str, np.ndarray], profile: bool = False
    ) -> dict[str, np.ndarray]:
        env = self._bind_feeds(feeds)
        blocks: dict[str, Block] = {}
        output_names = set(self.graph.output_names())

        for step in self.steps:
            node = step.node
            inputs = [env[name] for name in node.inputs]

            out_buffer = None
            out_block: Optional[Block] = None
            if (
                self.use_arena
                and step.spec.supports_out
                and node.outputs[0] not in output_names
            ):
                shape = step.spec.infer_shape(
                    [tuple(a.shape) for a in inputs], node.attrs
                )
                dtype = step.spec.infer_dtype([a.dtype for a in inputs], node.attrs)
                # Acquired before the kernel runs and before any input is
                # freed, so the arena can never hand back a buffer this node is
                # still reading from.
                out_buffer, out_block = self.arena.acquire(shape, dtype)

            started = time.perf_counter() if profile else 0.0
            result = step.spec.compute(
                self.backend, inputs, node.attrs, out=out_buffer
            )
            if profile:
                entry = self.profile.setdefault(
                    node.name, ProfileEntry(node.op_type, node.name)
                )
                entry.calls += 1
                entry.seconds += time.perf_counter() - started

            if out_block is not None and result is not out_buffer:
                # The kernel ignored the buffer we offered; give it straight back.
                self.arena.release(out_block)
                out_block = None

            env[node.outputs[0]] = result
            if out_block is not None:
                blocks[node.outputs[0]] = out_block

            for name in step.frees:
                self._retire(name, env, blocks)

        outputs = {}
        for name in self.graph.output_names():
            array = env[name]
            # An output may be a view into arena memory (e.g. a trailing
            # Reshape). Copy it out before the pool is recycled.
            if self.arena.owns(_root_buffer(array)):
                array = np.array(array, copy=True)
            outputs[name] = self.backend.to_host(array)

        self.arena.release_all()
        return outputs

    def _retire(
        self,
        name: str,
        env: dict[str, np.ndarray],
        blocks: dict[str, Block],
    ) -> None:
        """Drop a dead value, freeing its buffer unless a live view holds it."""
        block = blocks.pop(name, None)
        env.pop(name, None)
        if block is None:
            return
        for other, array in env.items():
            if other not in blocks and _root_buffer(array) is block.buffer:
                blocks[other] = block  # ownership moves to the surviving view
                return
        self.arena.release(block)

    # -- introspection ---------------------------------------------------

    def peak_live_values(self) -> int:
        """Widest point of the graph: the most values live at once."""
        live: set[str] = set(self.graph.input_names())
        peak = len(live)
        for step in self.steps:
            live.add(step.node.outputs[0])
            peak = max(peak, len(live))
            live.difference_update(step.frees)
        return peak

    def profile_table(self) -> str:
        if not self.profile:
            return "no profile data; run with profile=True"
        rows = sorted(self.profile.values(), key=lambda e: -e.seconds)
        total = sum(e.seconds for e in rows) or 1.0
        lines = [f"{'node':<24} {'op':<12} {'ms/call':>9} {'share':>7}"]
        for entry in rows:
            lines.append(
                f"{entry.node:<24} {entry.op_type:<12} "
                f"{entry.ms_per_call:>9.4f} {entry.seconds / total:>6.1%}"
            )
        return "\n".join(lines)

    def describe(self) -> str:
        return (
            f"{self.graph.summary()}\n"
            f"backend={self.backend.name}, steps={len(self.steps)}, "
            f"arena={'on' if self.use_arena else 'off'}, "
            f"peak live values={self.peak_live_values()}"
        )
