"""Graph-level optimization passes.

These run once, at load time, and rewrite the graph into a cheaper but
numerically equivalent form. The wins come from doing less work per call:

* **Constant folding** evaluates subgraphs that depend only on weights.
* **Fusion** collapses ``MatMul -> Add -> Relu`` into a single ``Gemm`` node,
  which turns three passes over the activation into one and removes two
  intermediate buffers.
* **Dead code elimination** drops nodes nothing downstream reads.
* **Quantization** (opt-in) rewrites ``Gemm`` weights to int8, cutting weight
  memory ~4x at the cost of a small accuracy change.

Every pass takes a graph and returns a new one plus a count of what it changed,
so the pass manager can report exactly what happened rather than leaving the
user to diff two graphs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from . import ops
from .backends.base import Backend
from .backends.numpy_backend import NumpyBackend
from .graph import Graph, Node, reachable_nodes

Pass = Callable[[Graph], tuple[Graph, int]]

# Activations that can be folded into a Gemm/QGemm as an attribute.
FUSIBLE_ACTIVATIONS = {"Relu": "relu", "Gelu": "gelu",
                       "Sigmoid": "sigmoid", "Tanh": "tanh"}


@dataclass
class OptimizationReport:
    applied: list[tuple[str, int]] = field(default_factory=list)
    nodes_before: int = 0
    nodes_after: int = 0
    weight_bytes_before: int = 0
    weight_bytes_after: int = 0

    def __str__(self) -> str:
        if not self.applied:
            return "optimizer: no changes"
        changes = ", ".join(f"{name} x{count}" for name, count in self.applied)
        return (
            f"optimizer: {self.nodes_before} -> {self.nodes_after} nodes "
            f"({changes}); weights "
            f"{self.weight_bytes_before / 1e6:.2f} -> "
            f"{self.weight_bytes_after / 1e6:.2f} MB"
        )


def _single_consumer(graph: Graph, value: str) -> Optional[Node]:
    """The one node reading ``value``, or None if it is read 0 or 2+ times.

    Fusing across a value with several consumers, or one that leaves the graph,
    would delete a result somebody else still needs.
    """
    if value in graph.output_names():
        return None
    consumers = graph.consumers(value)
    return consumers[0] if len(consumers) == 1 else None


# -- passes ----------------------------------------------------------------


def constant_fold(graph: Graph, backend: Optional[Backend] = None) -> tuple[Graph, int]:
    """Evaluate nodes whose inputs are all constants, at load time."""
    backend = backend or NumpyBackend()
    graph = graph.copy()
    folded = 0

    for node in list(graph.topological_order()):
        if node.op_type == "QGemmInt8":  # already the product of a rewrite
            continue
        if not node.inputs or not all(graph.is_constant(i) for i in node.inputs):
            continue
        if any(out in graph.output_names() for out in node.outputs):
            continue

        spec = ops.get(node.op_type)
        inputs = [graph.initializers[i] for i in node.inputs]
        try:
            result = spec.compute(backend, inputs, node.attrs)
        except Exception:
            # A node that cannot be folded is left for run time rather than
            # failing the whole load.
            continue

        graph.initializers[node.outputs[0]] = np.ascontiguousarray(result)
        graph.nodes.remove(node)
        folded += 1

    return graph, folded


def fuse_matmul_bias(graph: Graph) -> tuple[Graph, int]:
    """``MatMul(x, w) -> Add(_, b)`` becomes ``Gemm(x, w, b)``."""
    graph = graph.copy()
    fused = 0

    for node in list(graph.nodes):
        if node.op_type != "MatMul":
            continue
        consumer = _single_consumer(graph, node.outputs[0])
        if consumer is None or consumer.op_type != "Add":
            continue

        other = [i for i in consumer.inputs if i != node.outputs[0]]
        if len(other) != 1 or not graph.is_constant(other[0]):
            continue
        bias = graph.initializers[other[0]]
        if bias.ndim != 1:  # only a broadcast row-vector bias folds cleanly
            continue

        node.op_type = "Gemm"
        node.inputs = list(node.inputs) + [other[0]]
        node.outputs = list(consumer.outputs)
        graph.nodes.remove(consumer)
        fused += 1

    return graph, fused


def fuse_activation(graph: Graph) -> tuple[Graph, int]:
    """Fold a trailing elementwise activation into the Gemm that feeds it."""
    graph = graph.copy()
    fused = 0

    for node in list(graph.nodes):
        if node.op_type not in ("Gemm", "MatMul", "QGemmInt8"):
            continue
        if node.attrs.get("activation"):
            continue
        consumer = _single_consumer(graph, node.outputs[0])
        if consumer is None or consumer.op_type not in FUSIBLE_ACTIVATIONS:
            continue

        if node.op_type == "MatMul":
            node.op_type = "Gemm"
        node.attrs = dict(node.attrs)
        node.attrs["activation"] = FUSIBLE_ACTIVATIONS[consumer.op_type]
        node.outputs = list(consumer.outputs)
        graph.nodes.remove(consumer)
        fused += 1

    return graph, fused


def eliminate_dead_nodes(graph: Graph) -> tuple[Graph, int]:
    """Drop nodes and weights that no graph output depends on."""
    graph = graph.copy()
    keep = reachable_nodes(graph, graph.output_names())
    removed = [n for n in graph.nodes if n.name not in keep]
    graph.nodes = [n for n in graph.nodes if n.name in keep]

    live_values = set(graph.output_names())
    for node in graph.nodes:
        live_values.update(node.inputs)
    dead_weights = [k for k in graph.initializers if k not in live_values]
    for key in dead_weights:
        del graph.initializers[key]

    return graph, len(removed) + len(dead_weights)


def quantize_int8(graph: Graph) -> tuple[Graph, int]:
    """Rewrite ``Gemm`` to int8 weights with per-output-column scales.

    Weights quantize statically here; activations quantize per row at run time
    inside the kernel. Per-column scales matter -- a single scale for the whole
    matrix loses far more accuracy when columns have different magnitudes.
    """
    graph = graph.copy()
    converted = 0

    for node in list(graph.nodes):
        if node.op_type != "Gemm" or len(node.inputs) < 2:
            continue
        weight_name = node.inputs[1]
        if not graph.is_constant(weight_name):
            continue
        weight = graph.initializers[weight_name]
        if weight.ndim != 2 or weight.dtype != np.float32:
            continue

        scale = np.maximum(np.max(np.abs(weight), axis=0), 1e-12) / 127.0
        quantized = np.rint(weight / scale).clip(-127, 127).astype(np.int8)

        q_name = f"{weight_name}_q"
        s_name = f"{weight_name}_scale"
        graph.initializers[q_name] = np.ascontiguousarray(quantized)
        graph.initializers[s_name] = np.ascontiguousarray(scale.astype(np.float32))
        del graph.initializers[weight_name]

        node.op_type = "QGemmInt8"
        node.inputs = [node.inputs[0], q_name, s_name] + list(node.inputs[2:])
        converted += 1

    return graph, converted


# -- pass manager ----------------------------------------------------------

DEFAULT_PASSES: list[tuple[str, Pass]] = [
    ("constant_fold", constant_fold),
    ("fuse_matmul_bias", fuse_matmul_bias),
    ("fuse_activation", fuse_activation),
    ("eliminate_dead_nodes", eliminate_dead_nodes),
]


def optimize(
    graph: Graph,
    passes: Optional[list[tuple[str, Pass]]] = None,
    quantize: bool = False,
) -> tuple[Graph, OptimizationReport]:
    """Run the pass pipeline, validating the graph after each pass."""
    report = OptimizationReport(
        nodes_before=len(graph.nodes),
        weight_bytes_before=sum(a.nbytes for a in graph.initializers.values()),
    )

    pipeline = list(passes if passes is not None else DEFAULT_PASSES)
    if quantize:
        # Quantize after fusion so activations are already folded into the Gemm
        # attrs, then re-run DCE to drop the replaced float weights.
        pipeline.append(("quantize_int8", quantize_int8))
        pipeline.append(("eliminate_dead_nodes", eliminate_dead_nodes))

    for name, pass_fn in pipeline:
        graph, changes = pass_fn(graph)
        if changes:
            report.applied.append((name, changes))
            graph.validate()

    report.nodes_after = len(graph.nodes)
    report.weight_bytes_after = sum(a.nbytes for a in graph.initializers.values())
    return graph, report
