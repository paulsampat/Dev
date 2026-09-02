"""The computation graph IR.

A model is a directed acyclic graph of ``Node``s. Each node names an operator,
its input values and its output values; values are just strings. Constant
tensors (weights) live in ``Graph.initializers``. Everything else -- the
optimizer, the memory planner, the executor -- is a transformation or a
traversal over this structure.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Iterable, Iterator, Optional

import numpy as np

from .tensor import TensorSpec, dtype_to_name


@dataclass
class Node:
    name: str
    op_type: str
    inputs: list[str]
    outputs: list[str]
    attrs: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "op_type": self.op_type,
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
            "attrs": dict(self.attrs),
        }

    @staticmethod
    def from_dict(data: dict) -> "Node":
        return Node(
            name=data["name"],
            op_type=data["op_type"],
            inputs=list(data["inputs"]),
            outputs=list(data["outputs"]),
            attrs=dict(data.get("attrs", {})),
        )


class GraphError(ValueError):
    """Raised when a graph is structurally invalid."""


@dataclass
class Graph:
    name: str
    nodes: list[Node]
    inputs: list[TensorSpec]
    outputs: list[TensorSpec]
    initializers: dict[str, np.ndarray] = field(default_factory=dict)

    # -- traversal -------------------------------------------------------

    def producer(self, value: str) -> Optional[Node]:
        for node in self.nodes:
            if value in node.outputs:
                return node
        return None

    def consumers(self, value: str) -> list[Node]:
        return [n for n in self.nodes if value in n.inputs]

    def input_names(self) -> set[str]:
        return {spec.name for spec in self.inputs}

    def output_names(self) -> list[str]:
        return [spec.name for spec in self.outputs]

    def is_constant(self, value: str) -> bool:
        return value in self.initializers

    def iter_values(self) -> Iterator[str]:
        seen: set[str] = set()
        for name in list(self.input_names()) + list(self.initializers):
            if name not in seen:
                seen.add(name)
                yield name
        for node in self.nodes:
            for value in node.outputs:
                if value not in seen:
                    seen.add(value)
                    yield value

    # -- structure -------------------------------------------------------

    def topological_order(self) -> list[Node]:
        """Return nodes in dependency order, raising on cycles."""
        available = self.input_names() | set(self.initializers)
        pending = list(self.nodes)
        ordered: list[Node] = []

        while pending:
            progressed = False
            still_pending = []
            for node in pending:
                if all(inp in available for inp in node.inputs if inp):
                    ordered.append(node)
                    available.update(node.outputs)
                    progressed = True
                else:
                    still_pending.append(node)
            pending = still_pending
            if not progressed:
                stuck = ", ".join(n.name for n in pending)
                missing = sorted(
                    {i for n in pending for i in n.inputs if i and i not in available}
                )
                raise GraphError(
                    f"graph has a cycle or dangling inputs; unresolved nodes: "
                    f"{stuck}; missing values: {', '.join(missing) or 'none'}"
                )
        return ordered

    def validate(self) -> None:
        seen_nodes: set[str] = set()
        produced: set[str] = self.input_names() | set(self.initializers)

        overlap = self.input_names() & set(self.initializers)
        if overlap:
            raise GraphError(
                f"values are both graph inputs and initializers: {sorted(overlap)}"
            )

        for node in self.nodes:
            if node.name in seen_nodes:
                raise GraphError(f"duplicate node name {node.name!r}")
            seen_nodes.add(node.name)
            for out in node.outputs:
                if out in produced:
                    raise GraphError(
                        f"value {out!r} is produced more than once "
                        f"(node {node.name!r})"
                    )
                produced.add(out)

        for spec in self.outputs:
            if spec.name not in produced:
                raise GraphError(f"graph output {spec.name!r} is never produced")

        # Also surfaces cycles and dangling node inputs.
        self.topological_order()

    def copy(self) -> "Graph":
        return Graph(
            name=self.name,
            nodes=[replace(n, inputs=list(n.inputs), outputs=list(n.outputs),
                           attrs=dict(n.attrs)) for n in self.nodes],
            inputs=list(self.inputs),
            outputs=list(self.outputs),
            initializers=dict(self.initializers),
        )

    # -- serialisation ---------------------------------------------------

    def to_header(self, offsets: dict[str, dict]) -> dict:
        """Serialise structure only; weight bytes are described by ``offsets``."""
        return {
            "name": self.name,
            "inputs": [s.to_dict() for s in self.inputs],
            "outputs": [s.to_dict() for s in self.outputs],
            "nodes": [n.to_dict() for n in self.nodes],
            "initializers": offsets,
        }

    def summary(self) -> str:
        counts: dict[str, int] = {}
        for node in self.nodes:
            counts[node.op_type] = counts.get(node.op_type, 0) + 1
        ops = ", ".join(f"{k}x{v}" for k, v in sorted(counts.items()))
        weight_bytes = sum(a.nbytes for a in self.initializers.values())
        dtypes = sorted({dtype_to_name(a.dtype) for a in self.initializers.values()})
        return (
            f"{self.name}: {len(self.nodes)} nodes [{ops}], "
            f"{len(self.initializers)} initializers "
            f"({weight_bytes / 1e6:.2f} MB, {'/'.join(dtypes) or 'none'})"
        )


def reachable_nodes(graph: Graph, from_values: Iterable[str]) -> set[str]:
    """Names of nodes contributing to ``from_values`` (backwards reachability)."""
    producers = {out: n for n in graph.nodes for out in n.outputs}
    keep: set[str] = set()
    stack = list(from_values)
    while stack:
        value = stack.pop()
        node = producers.get(value)
        if node is None or node.name in keep:
            continue
        keep.add(node.name)
        stack.extend(i for i in node.inputs if i)
    return keep
