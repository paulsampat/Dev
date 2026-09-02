"""The user-facing API.

Everything below this line is machinery; this is the surface a caller touches::

    with InferenceSession("model.iem") as session:
        outputs = session.run({"x": batch})
"""

from __future__ import annotations

import threading
from typing import Optional, Union

import numpy as np

from .backends.base import Backend
from .backends.numpy_backend import NumpyBackend
from .executor import ExecutionPlan
from .format import ModelFile
from .graph import Graph
from .loader import load_graph
from .optimizer import OptimizationReport, optimize
from .tensor import TensorSpec, format_shape

BACKENDS: dict[str, type[Backend]] = {"numpy": NumpyBackend}


def get_backend(name: str) -> Backend:
    try:
        return BACKENDS[name]()
    except KeyError:
        raise ValueError(
            f"unknown backend {name!r}; available: {', '.join(sorted(BACKENDS))}"
        ) from None


class InferenceSession:
    """Loads a model once, then runs it many times.

    A session owns a compiled plan and its memory arena, and serialises calls
    with a lock -- the arena is stateful, so concurrent ``run`` calls on one
    session would hand the same buffer to two computations. For parallelism,
    create one session per thread, or put a :class:`BatchScheduler` in front.
    """

    def __init__(
        self,
        model: Union[str, Graph],
        *,
        backend: Union[str, Backend] = "numpy",
        optimize_graph: bool = True,
        quantize: bool = False,
        use_arena: bool = True,
    ):
        self._file: Optional[ModelFile] = None
        if isinstance(model, str):
            graph, self._file = load_graph(model)
        else:
            graph = model.copy()
            graph.validate()

        self.backend = backend if isinstance(backend, Backend) else get_backend(backend)
        self.report: Optional[OptimizationReport] = None
        if optimize_graph or quantize:
            graph, self.report = optimize(graph, quantize=quantize)

        self.graph = graph
        self.plan = ExecutionPlan.compile(graph, self.backend, use_arena=use_arena)
        self._lock = threading.Lock()

    # -- introspection ---------------------------------------------------

    @property
    def inputs(self) -> list[TensorSpec]:
        return list(self.graph.inputs)

    @property
    def outputs(self) -> list[TensorSpec]:
        return list(self.graph.outputs)

    def signature(self) -> str:
        def side(specs):
            return ", ".join(f"{s.name}: {format_shape(s.shape)}" for s in specs)

        return f"({side(self.inputs)}) -> ({side(self.outputs)})"

    def describe(self) -> str:
        lines = [self.plan.describe(), f"signature: {self.signature()}"]
        if self.report is not None:
            lines.append(str(self.report))
        return "\n".join(lines)

    # -- execution -------------------------------------------------------

    def run(
        self, feeds: dict[str, np.ndarray], profile: bool = False
    ) -> dict[str, np.ndarray]:
        with self._lock:
            return self.plan.run(feeds, profile=profile)

    @property
    def arena_stats(self):
        return self.plan.arena.stats

    # -- lifecycle -------------------------------------------------------

    def close(self) -> None:
        self.plan.arena.reset()
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self) -> "InferenceSession":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()
