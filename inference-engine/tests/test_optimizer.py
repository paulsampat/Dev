import numpy as np
import pytest

from engine.builder import GraphBuilder
from engine.executor import ExecutionPlan
from engine.optimizer import (
    constant_fold,
    eliminate_dead_nodes,
    fuse_activation,
    fuse_matmul_bias,
    optimize,
    quantize_int8,
)
from tools.models import build_mlp, build_transformer_block

RNG = np.random.default_rng(3)


def rand(*shape):
    return RNG.standard_normal(shape).astype(np.float32)


def op_types(graph):
    return sorted(n.op_type for n in graph.nodes)


def run(graph, feeds):
    return ExecutionPlan.compile(graph).run(feeds)


def simple_mlp_graph():
    b = GraphBuilder("g")
    x = b.input("x", (None, 4))
    y = b.linear(x, rand(4, 8), rand(8), activation="Relu")
    b.output(y, (None, 8))
    return b.build()


# -- individual passes -----------------------------------------------------


def test_fuse_matmul_bias_produces_a_gemm():
    graph, count = fuse_matmul_bias(simple_mlp_graph())
    assert count == 1
    assert op_types(graph) == ["Gemm", "Relu"]


def test_fuse_activation_folds_relu_into_the_gemm():
    graph, _ = fuse_matmul_bias(simple_mlp_graph())
    graph, count = fuse_activation(graph)

    assert count == 1
    assert op_types(graph) == ["Gemm"]
    assert graph.nodes[0].attrs["activation"] == "relu"


def test_fusion_preserves_the_graph_output_name():
    original = simple_mlp_graph()
    fused, _ = optimize(original)
    assert fused.output_names() == original.output_names()


def test_fusion_is_skipped_when_the_intermediate_is_also_a_graph_output():
    b = GraphBuilder("g")
    x = b.input("x", (None, 4))
    h = b.op("MatMul", [x, b.constant("w", rand(4, 8))])
    y = b.op("Relu", [h])
    b.output(h, (None, 8))  # h escapes, so folding Relu into the MatMul
    b.output(y, (None, 8))  # would destroy it
    graph = b.build()

    _, count = fuse_activation(graph)
    assert count == 0


def test_fusion_is_skipped_when_the_intermediate_has_two_consumers():
    b = GraphBuilder("g")
    x = b.input("x", (None, 4))
    h = b.op("MatMul", [x, b.constant("w", rand(4, 8))])
    left = b.op("Relu", [h])
    right = b.op("Gelu", [h])
    b.output(b.op("Add", [left, right]), (None, 8))
    graph = b.build()

    _, count = fuse_activation(graph)
    assert count == 0


def test_matmul_bias_fusion_requires_a_constant_rank_one_bias():
    b = GraphBuilder("g")
    x = b.input("x", (None, 4))
    h = b.op("MatMul", [x, b.constant("w", rand(4, 8))])
    b.output(b.op("Add", [h, b.constant("m", rand(1, 8))]), (None, 8))
    graph = b.build()

    _, count = fuse_matmul_bias(graph)
    assert count == 0, "a rank-2 addend is not a bias and must not be folded"


def test_constant_folding_evaluates_weight_only_subgraphs():
    b = GraphBuilder("g")
    x = b.input("x", (None, 4))
    w1 = b.constant("w1", rand(4, 8))
    w2 = b.constant("w2", rand(8, 8))
    folded_weight = b.op("MatMul", [w1, w2])       # depends only on constants
    b.output(b.op("MatMul", [x, folded_weight]), (None, 8))
    graph = b.build()

    optimized, count = constant_fold(graph)
    assert count == 1
    assert op_types(optimized) == ["MatMul"]
    assert np.allclose(
        optimized.initializers[folded_weight],
        graph.initializers["w1"] @ graph.initializers["w2"],
        atol=1e-5,
    )


def test_constant_folding_does_not_change_results():
    b = GraphBuilder("g")
    x = b.input("x", (None, 4))
    w = b.op("MatMul", [b.constant("w1", rand(4, 8)), b.constant("w2", rand(8, 8))])
    b.output(b.op("MatMul", [x, w]), (None, 8))
    graph = b.build()

    feeds = {"x": rand(3, 4)}
    output = graph.output_names()[0]
    optimized, _ = constant_fold(graph)
    before = run(graph, feeds)[output]
    after = run(optimized, feeds)[output]
    assert np.allclose(before, after, atol=1e-5)


def test_dead_code_elimination_drops_unused_nodes_and_weights():
    b = GraphBuilder("g")
    x = b.input("x", (None, 4))
    y = b.op("Relu", [x])
    b.op("MatMul", [x, b.constant("unused_w", rand(4, 8))])  # nothing reads this
    b.output(y, (None, 4))
    graph = b.build()

    optimized, count = eliminate_dead_nodes(graph)
    assert count == 2  # the node and its now-orphaned weight
    assert op_types(optimized) == ["Relu"]
    assert "unused_w" not in optimized.initializers


# -- quantization ----------------------------------------------------------


def test_quantization_shrinks_weights_about_fourfold():
    graph, _ = optimize(build_mlp().graph)
    before = sum(a.nbytes for a in graph.initializers.values())

    quantized, count = quantize_int8(graph)
    quantized, _ = eliminate_dead_nodes(quantized)
    after = sum(a.nbytes for a in quantized.initializers.values())

    assert count == 3
    assert 3.5 < before / after < 4.5


def test_quantization_keeps_predictions_close():
    demo = build_mlp()
    graph, _ = optimize(demo.graph)
    quantized, _ = optimize(demo.graph, quantize=True)

    feeds = {"x": demo.example_input}
    exact = run(graph, feeds)[graph.output_names()[0]]
    approx = run(quantized, feeds)[quantized.output_names()[0]]

    relative = np.abs(approx - exact).max() / np.abs(exact).max()
    assert relative < 0.05, f"int8 drift {relative:.2%} is too large"
    assert np.array_equal(approx.argmax(-1), exact.argmax(-1))


def test_quantization_keeps_the_fused_activation():
    quantized, _ = optimize(build_mlp().graph, quantize=True)
    activations = [n.attrs.get("activation") for n in quantized.nodes]
    assert activations.count("relu") == 2


# -- the pipeline as a whole -----------------------------------------------


@pytest.mark.parametrize("build", [build_mlp, build_transformer_block])
def test_optimized_and_unoptimized_graphs_agree(build):
    demo = build(seed=1)
    feeds = {demo.input_name: demo.example_input}

    baseline = run(demo.graph, feeds)[demo.output_name]
    optimized_graph, report = optimize(demo.graph)
    optimized = run(optimized_graph, feeds)[optimized_graph.output_names()[0]]

    assert report.nodes_after < report.nodes_before
    assert np.allclose(baseline, optimized, atol=1e-5)


def test_optimize_leaves_a_validated_graph():
    optimized, _ = optimize(build_transformer_block().graph)
    optimized.validate()


def test_report_describes_the_changes():
    _, report = optimize(build_mlp().graph)
    text = str(report)
    assert "fuse_matmul_bias" in text and "nodes" in text


def test_optimizing_an_already_optimized_graph_is_a_no_op():
    once, _ = optimize(build_mlp().graph)
    _, second_report = optimize(once)
    assert second_report.applied == []
