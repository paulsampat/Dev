import numpy as np
import pytest

from engine.graph import Graph, GraphError, Node, reachable_nodes
from engine.tensor import TensorSpec, broadcast_shapes, dtype_from_name


def spec(name, shape, dtype="float32"):
    return TensorSpec(name, dtype_from_name(dtype), tuple(shape))


def linear_graph():
    return Graph(
        name="g",
        nodes=[
            Node("n0", "MatMul", ["x", "w"], ["h"]),
            Node("n1", "Relu", ["h"], ["y"]),
        ],
        inputs=[spec("x", (None, 4))],
        outputs=[spec("y", (None, 8))],
        initializers={"w": np.zeros((4, 8), dtype=np.float32)},
    )


def test_topological_order_follows_dependencies():
    graph = linear_graph()
    graph.nodes.reverse()  # declared out of order
    assert [n.name for n in graph.topological_order()] == ["n0", "n1"]


def test_validate_accepts_a_well_formed_graph():
    linear_graph().validate()


def test_cycle_is_rejected():
    graph = linear_graph()
    graph.nodes[0].inputs = ["x", "w", "y"]  # n0 now depends on n1's output
    with pytest.raises(GraphError, match="cycle"):
        graph.validate()


def test_dangling_input_is_rejected():
    graph = linear_graph()
    graph.nodes[0].inputs = ["x", "missing"]
    with pytest.raises(GraphError, match="missing"):
        graph.validate()


def test_duplicate_output_is_rejected():
    graph = linear_graph()
    graph.nodes.append(Node("n2", "Relu", ["x"], ["h"]))
    with pytest.raises(GraphError, match="produced more than once"):
        graph.validate()


def test_duplicate_node_name_is_rejected():
    graph = linear_graph()
    graph.nodes.append(Node("n0", "Relu", ["y"], ["z"]))
    with pytest.raises(GraphError, match="duplicate node name"):
        graph.validate()


def test_unproduced_output_is_rejected():
    graph = linear_graph()
    graph.outputs = [spec("nope", (None, 8))]
    with pytest.raises(GraphError, match="never produced"):
        graph.validate()


def test_value_that_is_both_input_and_initializer_is_rejected():
    graph = linear_graph()
    graph.initializers["x"] = np.zeros((1, 4), dtype=np.float32)
    with pytest.raises(GraphError, match="both graph inputs and initializers"):
        graph.validate()


def test_reachable_nodes_ignores_dead_branches():
    graph = linear_graph()
    graph.nodes.append(Node("dead", "Relu", ["x"], ["unused"]))
    assert reachable_nodes(graph, ["y"]) == {"n0", "n1"}


def test_copy_is_deep_enough_to_edit_safely():
    graph = linear_graph()
    clone = graph.copy()
    clone.nodes[0].inputs.append("extra")
    clone.nodes[0].attrs["a"] = 1
    assert graph.nodes[0].inputs == ["x", "w"]
    assert graph.nodes[0].attrs == {}


def test_consumers_and_producer():
    graph = linear_graph()
    assert graph.producer("h").name == "n0"
    assert [n.name for n in graph.consumers("h")] == ["n1"]
    assert graph.producer("x") is None


@pytest.mark.parametrize(
    "a,b,expected",
    [
        ((3, 4), (4,), (3, 4)),
        ((2, 1, 4), (3, 4), (2, 3, 4)),
        ((5,), (1,), (5,)),
        ((), (2, 2), (2, 2)),
    ],
)
def test_broadcast_shapes(a, b, expected):
    assert broadcast_shapes(a, b) == expected


def test_broadcast_shapes_rejects_incompatible():
    with pytest.raises(ValueError, match="not broadcastable"):
        broadcast_shapes((3, 4), (5, 4))


def test_tensor_spec_matches_reports_problems():
    s = spec("x", (None, 4))
    assert s.matches(np.zeros((2, 4), dtype=np.float32)) is None
    assert "dtype" in s.matches(np.zeros((2, 4), dtype=np.float64))
    assert "rank" in s.matches(np.zeros((2, 4, 1), dtype=np.float32))
    assert "axis 1" in s.matches(np.zeros((2, 5), dtype=np.float32))
