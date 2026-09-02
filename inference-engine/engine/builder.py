"""A small helper for constructing graphs in code.

Real engines get graphs from a converter (ONNX, GGUF, a PyTorch export). This
builder plays that role for the demo models and the tests: it names values,
tracks weights and produces a validated :class:`Graph`.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .graph import Graph, Node
from .tensor import Shape, TensorSpec, dtype_from_name


class GraphBuilder:
    def __init__(self, name: str = "model"):
        self.name = name
        self._nodes: list[Node] = []
        self._inputs: list[TensorSpec] = []
        self._outputs: list[TensorSpec] = []
        self._initializers: dict[str, np.ndarray] = {}
        self._counter = 0

    def _fresh(self, prefix: str) -> str:
        self._counter += 1
        return f"{prefix}_{self._counter}"

    def input(self, name: str, shape: Shape, dtype: str = "float32") -> str:
        self._inputs.append(
            TensorSpec(name=name, dtype=dtype_from_name(dtype), shape=tuple(shape))
        )
        return name

    def constant(self, name: str, array: np.ndarray) -> str:
        if name in self._initializers:
            raise ValueError(f"constant {name!r} already exists")
        self._initializers[name] = np.ascontiguousarray(array)
        return name

    def op(
        self,
        op_type: str,
        inputs: Sequence[str],
        out: Optional[str] = None,
        node_name: Optional[str] = None,
        **attrs,
    ) -> str:
        out = out or self._fresh(op_type.lower())
        self._nodes.append(
            Node(
                name=node_name or self._fresh("node"),
                op_type=op_type,
                inputs=list(inputs),
                outputs=[out],
                attrs={k: v for k, v in attrs.items() if v is not None},
            )
        )
        return out

    # Convenience wrappers for the common patterns.

    def linear(
        self,
        x: str,
        weight: np.ndarray,
        bias: Optional[np.ndarray] = None,
        prefix: str = "fc",
        activation: Optional[str] = None,
    ) -> str:
        w_name = self.constant(self._fresh(f"{prefix}_w"), weight)
        y = self.op("MatMul", [x, w_name])
        if bias is not None:
            b_name = self.constant(self._fresh(f"{prefix}_b"), bias)
            y = self.op("Add", [y, b_name])
        if activation is not None:
            y = self.op(activation, [y])
        return y

    def output(self, value: str, shape: Shape, dtype: str = "float32") -> str:
        self._outputs.append(
            TensorSpec(name=value, dtype=dtype_from_name(dtype), shape=tuple(shape))
        )
        return value

    def build(self) -> Graph:
        if not self._outputs:
            raise ValueError("graph has no declared outputs")
        graph = Graph(
            name=self.name,
            nodes=self._nodes,
            inputs=self._inputs,
            outputs=self._outputs,
            initializers=self._initializers,
        )
        graph.validate()
        return graph
