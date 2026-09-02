"""Whole-pipeline checks: file -> graph -> optimizer -> plan -> numbers.

The load-bearing assertion in this file is agreement with the reference
implementations in ``tools.models``, which compute the same maths directly in
numpy without touching any engine code.
"""

import numpy as np
import pytest

from engine.api import InferenceSession
from engine.loader import save_graph
from tools.export import main as export_main
from tools.models import build_mlp, build_transformer_block

BUILDERS = [build_mlp, build_transformer_block]


@pytest.mark.parametrize("build", BUILDERS)
def test_session_matches_the_reference_implementation(build):
    demo = build(seed=21)
    with InferenceSession(demo.graph) as session:
        result = session.run({demo.input_name: demo.example_input})

    output = result[session.outputs[0].name]
    expected = demo.reference(demo.example_input)
    assert output.shape == expected.shape
    assert np.allclose(output, expected, atol=1e-4), (
        f"max abs error {np.abs(output - expected).max():.2e}"
    )


@pytest.mark.parametrize("build", BUILDERS)
def test_loading_from_disk_matches_the_in_memory_graph(build, tmp_path):
    demo = build(seed=22)
    path = str(tmp_path / "m.iem")
    save_graph(path, demo.graph)

    with InferenceSession(path) as session:
        loaded = session.run({demo.input_name: demo.example_input})[
            session.outputs[0].name
        ]
    assert np.allclose(loaded, demo.reference(demo.example_input), atol=1e-4)


@pytest.mark.parametrize("batch", [1, 2, 7, 33])
def test_results_do_not_depend_on_batch_size(batch):
    demo = build_mlp(seed=23)
    rng = np.random.default_rng(batch)
    inputs = rng.standard_normal((batch, 784)).astype(np.float32)

    with InferenceSession(demo.graph) as session:
        output = session.run({"x": inputs})[session.outputs[0].name]

    assert output.shape == (batch, 10)
    assert np.allclose(output, demo.reference(inputs), atol=1e-4)


def test_rows_are_independent_of_the_batch_they_ride_in():
    """Row i of a batched call must equal that row run on its own."""
    demo = build_mlp(seed=24)
    batch = demo.example_input

    with InferenceSession(demo.graph) as session:
        name = session.outputs[0].name
        together = session.run({"x": batch})[name]
        alone = np.concatenate(
            [session.run({"x": batch[i : i + 1]})[name] for i in range(len(batch))]
        )
    assert np.allclose(together, alone, atol=1e-5)


@pytest.mark.parametrize("build", BUILDERS)
def test_optimization_can_be_turned_off(build):
    demo = build(seed=25)
    feeds = {demo.input_name: demo.example_input}

    with InferenceSession(demo.graph, optimize_graph=False) as plain:
        assert plain.report is None
        assert len(plain.graph.nodes) == len(demo.graph.nodes)
        baseline = plain.run(feeds)[plain.outputs[0].name]

    with InferenceSession(demo.graph) as optimized:
        assert len(optimized.graph.nodes) < len(demo.graph.nodes)
        tuned = optimized.run(feeds)[optimized.outputs[0].name]

    assert np.allclose(baseline, tuned, atol=1e-5)


def test_quantized_session_stays_accurate_and_smaller():
    demo = build_mlp(seed=26)
    feeds = {"x": demo.example_input}

    with InferenceSession(demo.graph) as full:
        exact = full.run(feeds)[full.outputs[0].name]
    with InferenceSession(demo.graph, quantize=True) as small:
        approx = small.run(feeds)[small.outputs[0].name]
        assert small.report.weight_bytes_after < small.report.weight_bytes_before / 3

    assert np.array_equal(approx.argmax(-1), exact.argmax(-1))
    assert np.abs(approx - exact).max() / np.abs(exact).max() < 0.05


def test_session_describes_itself():
    demo = build_mlp()
    with InferenceSession(demo.graph) as session:
        text = session.describe()
        assert "signature:" in text
        assert "x: (?, 784)" in session.signature()


def test_unknown_backend_lists_the_available_ones():
    with pytest.raises(ValueError, match="unknown backend"):
        InferenceSession(build_mlp().graph, backend="cuda")


def test_session_can_be_reused_many_times():
    demo = build_mlp(seed=27)
    with InferenceSession(demo.graph) as session:
        name = session.outputs[0].name
        for _ in range(20):
            output = session.run({"x": demo.example_input})[name]
            assert np.allclose(output, demo.reference(demo.example_input), atol=1e-4)


def test_close_is_idempotent():
    session = InferenceSession(build_mlp().graph)
    session.close()
    session.close()


def test_export_cli_writes_a_loadable_model(tmp_path, capsys):
    path = str(tmp_path / "cli.iem")
    assert export_main(["mlp", path]) == 0
    assert "wrote" in capsys.readouterr().out

    with InferenceSession(path) as session:
        rng = np.random.default_rng(0)
        inputs = rng.standard_normal((2, 784)).astype(np.float32)
        assert session.run({"x": inputs})[session.outputs[0].name].shape == (2, 10)
