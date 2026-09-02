"""A from-scratch ML inference engine.

Load a trained model, optimize its graph, and run forward passes efficiently::

    from engine import InferenceSession

    with InferenceSession("model.iem") as session:
        outputs = session.run({"x": batch})

The pieces, in the order data flows through them:

* :mod:`engine.format`     -- the ``.iem`` file format (mmap'd, zero-copy)
* :mod:`engine.loader`     -- file -> :class:`~engine.graph.Graph`
* :mod:`engine.optimizer`  -- constant folding, fusion, DCE, quantization
* :mod:`engine.executor`   -- planning, liveness analysis, the run loop
* :mod:`engine.allocator`  -- buffer recycling across the plan
* :mod:`engine.ops`        -- operator registry (compute + shape inference)
* :mod:`engine.backends`   -- the kernels a hardware target must provide
* :mod:`engine.scheduler`  -- dynamic batching for serving
"""

from .allocator import Arena
from .api import InferenceSession, get_backend
from .builder import GraphBuilder
from .executor import ExecutionPlan
from .format import FormatError, ModelFile
from .graph import Graph, GraphError, Node
from .loader import load_graph, save_graph
from .optimizer import optimize
from .scheduler import BatchScheduler
from .tensor import TensorSpec

__all__ = [
    "Arena",
    "BatchScheduler",
    "ExecutionPlan",
    "FormatError",
    "Graph",
    "GraphBuilder",
    "GraphError",
    "InferenceSession",
    "ModelFile",
    "Node",
    "TensorSpec",
    "get_backend",
    "load_graph",
    "optimize",
    "save_graph",
]
