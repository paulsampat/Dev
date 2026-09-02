"""Turning a model file into a validated in-memory ``Graph``."""

from __future__ import annotations

from typing import Optional

import numpy as np

from .format import FormatError, ModelFile, write_model
from .graph import Graph, Node
from .tensor import TensorSpec


def load_graph(path: str) -> tuple[Graph, ModelFile]:
    """Load a ``.iem`` file.

    The ``ModelFile`` is returned alongside the graph because it owns the mmap
    that the weight arrays are views into -- the caller must keep it alive for
    as long as the graph is used.
    """
    handle = ModelFile(path)
    try:
        header = handle.header
        if "graph" not in header:
            raise FormatError(f"{path}: header has no 'graph' section")
        spec = header["graph"]

        graph = Graph(
            name=spec.get("name", "model"),
            nodes=[Node.from_dict(n) for n in spec["nodes"]],
            inputs=[TensorSpec.from_dict(s) for s in spec["inputs"]],
            outputs=[TensorSpec.from_dict(s) for s in spec["outputs"]],
            initializers=handle.tensors(),
        )
        graph.validate()
        return graph, handle
    except Exception:
        handle.close()
        raise


def save_graph(path: str, graph: Graph, extra: Optional[dict] = None) -> None:
    """Serialise a graph and its weights to a ``.iem`` file."""
    graph.validate()
    header = dict(extra or {})
    header["graph"] = {
        "name": graph.name,
        "inputs": [s.to_dict() for s in graph.inputs],
        "outputs": [s.to_dict() for s in graph.outputs],
        "nodes": [n.to_dict() for n in graph.nodes],
    }
    tensors = {
        name: np.ascontiguousarray(array)
        for name, array in graph.initializers.items()
    }
    write_model(path, header, tensors)
