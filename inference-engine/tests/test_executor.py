import numpy as np
import pytest

from engine.builder import GraphBuilder
from engine.executor import ExecutionPlan
from tools.models import build_mlp, build_transformer_block

RNG = np.random.default_rng(11)


def rand(*shape):
    return RNG.standard_normal(shape).astype(np.float32)


# -- feed validation -------------------------------------------------------


def plan_for_mlp():
    return ExecutionPlan.compile(build_mlp(in_features=4, hidden=(8,), out_features=2).graph)


def test_missing_input_names_what_was_expected():
    with pytest.raises(ValueError, match="missing input"):
        plan_for_mlp().run({})


def test_unexpected_input_is_rejected():
    with pytest.raises(ValueError, match="unexpected input"):
        plan_for_mlp().run({"x": rand(2, 4), "stray": rand(2, 4)})


def test_wrong_static_dimension_is_rejected():
    with pytest.raises(ValueError, match="axis 1"):
        plan_for_mlp().run({"x": rand(2, 7)})


def test_wrong_rank_is_rejected():
    with pytest.raises(ValueError, match="rank"):
        plan_for_mlp().run({"x": rand(2, 4, 1)})


def test_wrong_dtype_is_rejected():
    with pytest.raises(ValueError, match="dtype"):
        plan_for_mlp().run({"x": np.zeros((2, 4), dtype=np.float64)})


def test_dynamic_batch_dimension_accepts_any_size():
    plan = plan_for_mlp()
    for batch in (1, 3, 16):
        assert plan.run({"x": rand(batch, 4)})[plan.graph.output_names()[0]].shape == (
            batch,
            2,
        )


# -- memory reuse and aliasing --------------------------------------------


def aliasing_graph():
    """A graph where a Reshape output is a view of a buffer that then dies.

    Without ownership transfer in the run loop, ``h``'s buffer is released the
    moment Reshape consumes it, the next MatMul reuses that same buffer, and
    the reshaped view silently reads the wrong data.
    """
    b = GraphBuilder("aliasing")
    x = b.input("x", (None, 4))
    w1 = b.constant("w1", rand(4, 8))
    w2 = b.constant("w2", rand(4, 8))

    h = b.op("MatMul", [x, w1])
    r = b.op("Reshape", [h], shape=[-1, 2, 4])   # view of h's buffer
    t = b.op("MatMul", [x, w2])                  # same size: a reuse candidate
    s = b.op("Reshape", [t], shape=[-1, 2, 4])
    out = b.op("Add", [r, s])
    b.output(out, (None, 2, 4))
    return b.build(), out


def test_a_view_is_not_corrupted_when_its_source_buffer_dies():
    graph, out = aliasing_graph()
    feeds = {"x": rand(5, 4)}

    expected = (
        (feeds["x"] @ graph.initializers["w1"]).reshape(5, 2, 4)
        + (feeds["x"] @ graph.initializers["w2"]).reshape(5, 2, 4)
    )
    result = ExecutionPlan.compile(graph, use_arena=True).run(feeds)[out]
    assert np.allclose(result, expected, atol=1e-5)


def test_transpose_views_survive_buffer_recycling():
    demo = build_transformer_block(seed=5)
    feeds = {demo.input_name: demo.example_input}
    result = ExecutionPlan.compile(demo.graph, use_arena=True).run(feeds)
    assert np.allclose(result[demo.output_name], demo.reference(demo.example_input),
                       atol=1e-4)


@pytest.mark.parametrize("build", [build_mlp, build_transformer_block])
def test_arena_on_and_off_agree_exactly(build):
    demo = build(seed=2)
    feeds = {demo.input_name: demo.example_input}

    with_arena = ExecutionPlan.compile(demo.graph, use_arena=True).run(feeds)
    without = ExecutionPlan.compile(demo.graph, use_arena=False).run(feeds)
    assert np.array_equal(with_arena[demo.output_name], without[demo.output_name])


def test_repeated_runs_are_identical():
    """Recycled buffers hold stale data; kernels must fully overwrite them."""
    demo = build_transformer_block(seed=4)
    plan = ExecutionPlan.compile(demo.graph)
    feeds = {demo.input_name: demo.example_input}

    first = plan.run(feeds)[demo.output_name].copy()
    for _ in range(4):
        plan.run({demo.input_name: rand(*demo.example_input.shape)})
    again = plan.run(feeds)[demo.output_name]
    assert np.array_equal(first, again)


def test_outputs_survive_the_arena_being_recycled():
    demo = build_transformer_block(seed=6)
    plan = ExecutionPlan.compile(demo.graph)

    held = plan.run({demo.input_name: demo.example_input})[demo.output_name]
    snapshot = held.copy()
    plan.run({demo.input_name: rand(*demo.example_input.shape)})
    assert np.array_equal(held, snapshot), "a returned output was recycled underneath"


def test_the_arena_actually_reuses_buffers():
    demo = build_transformer_block()
    plan = ExecutionPlan.compile(demo.graph)
    plan.run({demo.input_name: demo.example_input})
    assert plan.arena.stats.reuses > 0


def test_arena_holds_less_than_the_sum_of_all_intermediates():
    demo = build_transformer_block()
    plan = ExecutionPlan.compile(demo.graph)
    plan.run({demo.input_name: demo.example_input})
    stats = plan.arena.stats
    assert stats.bytes_allocated < stats.bytes_requested


# -- planning --------------------------------------------------------------


def test_plan_frees_each_value_exactly_once():
    demo = build_transformer_block()
    plan = ExecutionPlan.compile(demo.graph)
    freed = [name for step in plan.steps for name in step.frees]
    assert len(freed) == len(set(freed))


def test_graph_outputs_and_weights_are_never_freed():
    demo = build_mlp()
    plan = ExecutionPlan.compile(demo.graph)
    protected = set(demo.graph.output_names()) | set(demo.graph.initializers)
    for step in plan.steps:
        assert not protected & set(step.frees)


def test_peak_live_values_is_below_the_total_value_count():
    demo = build_transformer_block()
    plan = ExecutionPlan.compile(demo.graph)
    assert plan.peak_live_values() < len(list(demo.graph.iter_values()))


def test_profiling_records_every_node():
    demo = build_mlp()
    plan = ExecutionPlan.compile(demo.graph)
    plan.run({demo.input_name: demo.example_input}, profile=True)

    assert len(plan.profile) == len(plan.steps)
    assert all(entry.calls == 1 for entry in plan.profile.values())
    assert "ms/call" in plan.profile_table()


def test_profile_table_without_data_says_so():
    assert "no profile data" in ExecutionPlan.compile(build_mlp().graph).profile_table()


def test_describe_mentions_the_backend_and_step_count():
    text = ExecutionPlan.compile(build_mlp().graph).describe()
    assert "numpy" in text and "steps=" in text
