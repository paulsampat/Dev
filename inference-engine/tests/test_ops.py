import numpy as np
import pytest

from engine import ops
from engine.backends.numpy_backend import NumpyBackend
from tools.models import ref_gelu, ref_layernorm, ref_softmax

BACKEND = NumpyBackend()
RNG = np.random.default_rng(7)


def run(op_type, inputs, out=None, **attrs):
    return ops.get(op_type).compute(BACKEND, inputs, attrs, out=out)


def rand(*shape):
    return RNG.standard_normal(shape).astype(np.float32)


# -- registry --------------------------------------------------------------


def test_unknown_op_lists_what_is_registered():
    with pytest.raises(ValueError, match="unknown operator 'Nope'"):
        ops.get("Nope")


def test_arity_is_checked_with_a_useful_message():
    with pytest.raises(ValueError, match=r"expects 2 inputs, got 1"):
        ops.get("MatMul").check_arity(1, "n0")


def test_every_registered_op_declares_inference_functions():
    for name, spec in ops.REGISTRY.items():
        assert callable(spec.infer_shape), name
        assert callable(spec.infer_dtype), name
        assert spec.min_inputs >= 1, name


# -- shape inference -------------------------------------------------------


@pytest.mark.parametrize(
    "a,b,expected",
    [
        ((3, 4), (4, 5), (3, 5)),
        ((2, 3, 4), (4, 5), (2, 3, 5)),
        ((2, 1, 3, 4), (2, 4, 5), (2, 2, 3, 5)),
    ],
)
def test_matmul_shape_inference(a, b, expected):
    assert ops.get("MatMul").infer_shape([a, b], {}) == expected


def test_matmul_shape_inference_rejects_mismatched_inner_dims():
    with pytest.raises(ValueError, match="inner dimensions disagree"):
        ops.get("MatMul").infer_shape([(3, 4), (5, 6)], {})


def test_reshape_infers_the_placeholder_dimension():
    infer = ops.get("Reshape").infer_shape
    assert infer([(8, 12)], {"shape": [-1, 4, 3]}) == (8, 4, 3)
    assert infer([(8, 12)], {"shape": [2, -1]}) == (2, 48)


def test_reshape_rejects_impossible_targets():
    infer = ops.get("Reshape").infer_shape
    with pytest.raises(ValueError, match="cannot reshape"):
        infer([(8, 12)], {"shape": [7, -1]})
    with pytest.raises(ValueError, match="cannot reshape"):
        infer([(8, 12)], {"shape": [5, 5]})
    with pytest.raises(ValueError, match="more than one -1"):
        infer([(8, 12)], {"shape": [-1, -1]})


def test_transpose_shape_inference():
    infer = ops.get("Transpose").infer_shape
    assert infer([(2, 3, 4)], {"perm": [0, 2, 1]}) == (2, 4, 3)
    assert infer([(2, 3, 4)], {}) == (4, 3, 2)  # default reverses
    with pytest.raises(ValueError, match="invalid perm"):
        infer([(2, 3, 4)], {"perm": [0, 0, 1]})


def test_gather_shape_inference():
    assert ops.get("Gather").infer_shape([(10, 4), (2, 3)], {"axis": 0}) == (2, 3, 4)


def test_cast_changes_the_inferred_dtype():
    spec = ops.get("Cast")
    assert spec.infer_dtype([np.dtype(np.float32)], {"to": "float16"}) == np.float16


# -- kernels ---------------------------------------------------------------


def test_matmul_matches_numpy():
    a, b = rand(3, 4), rand(4, 5)
    assert np.allclose(run("MatMul", [a, b]), a @ b)


def test_gemm_applies_bias_and_activation():
    x, w, b = rand(3, 4), rand(4, 5), rand(5)
    assert np.allclose(run("Gemm", [x, w, b]), x @ w + b, atol=1e-6)
    assert np.allclose(
        run("Gemm", [x, w, b], activation="relu"), np.maximum(x @ w + b, 0), atol=1e-6
    )
    assert np.allclose(
        run("Gemm", [x, w, b], activation="gelu"), ref_gelu(x @ w + b), atol=1e-5
    )


def test_gemm_rejects_an_unknown_activation():
    with pytest.raises(ValueError, match="unknown fused activation"):
        run("Gemm", [rand(2, 3), rand(3, 3)], activation="swish")


def test_qgemm_int8_stays_close_to_float():
    x, w, b = rand(6, 32), rand(32, 16), rand(16)
    scale = np.maximum(np.max(np.abs(w), axis=0), 1e-12) / 127.0
    w_q = np.rint(w / scale).clip(-127, 127).astype(np.int8)

    quantized = run("QGemmInt8", [x, w_q, scale.astype(np.float32), b])
    exact = x @ w + b
    relative = np.abs(quantized - exact).max() / np.abs(exact).max()
    assert relative < 0.05, f"int8 error {relative:.3%} is larger than expected"


@pytest.mark.parametrize(
    "op_type,fn",
    [
        ("Relu", lambda x: np.maximum(x, 0)),
        ("Sigmoid", lambda x: 1 / (1 + np.exp(-x))),
        ("Tanh", np.tanh),
        ("Gelu", ref_gelu),
    ],
)
def test_activations_match_reference(op_type, fn):
    x = rand(4, 6)
    assert np.allclose(run(op_type, [x]), fn(x), atol=1e-6)


def test_activations_do_not_mutate_their_input():
    x = rand(4, 6)
    original = x.copy()
    for op_type in ("Relu", "Sigmoid", "Tanh", "Gelu"):
        run(op_type, [x])
        assert np.array_equal(x, original), f"{op_type} mutated its input"


def test_softmax_matches_reference_and_sums_to_one():
    x = rand(3, 5) * 10
    result = run("Softmax", [x], axis=-1)
    assert np.allclose(result, ref_softmax(x), atol=1e-6)
    assert np.allclose(result.sum(axis=-1), 1.0, atol=1e-6)


def test_softmax_is_numerically_stable_for_large_inputs():
    x = np.array([[1000.0, 1001.0, 1002.0]], dtype=np.float32)
    result = run("Softmax", [x], axis=-1)
    assert np.isfinite(result).all()
    assert np.allclose(result.sum(), 1.0)


def test_layernorm_matches_reference():
    x, w, b = rand(2, 3, 8), rand(8), rand(8)
    assert np.allclose(
        run("LayerNorm", [x, w, b], epsilon=1e-5), ref_layernorm(x, w, b), atol=1e-5
    )


def test_layernorm_normalises_the_last_axis():
    x = rand(4, 16) * 5 + 3
    ones, zeros = np.ones(16, np.float32), np.zeros(16, np.float32)
    result = run("LayerNorm", [x, ones, zeros], epsilon=1e-5)
    assert np.allclose(result.mean(axis=-1), 0, atol=1e-5)
    assert np.allclose(result.std(axis=-1), 1, atol=1e-3)


def test_scale_gather_and_cast():
    x = rand(2, 3)
    assert np.allclose(run("Scale", [x], factor=0.5), x * 0.5)

    table = rand(10, 4)
    indices = np.array([0, 3, 9], dtype=np.int64)
    assert np.allclose(run("Gather", [table, indices], axis=0), table[indices])

    assert run("Cast", [x], to="float16").dtype == np.float16


# -- the out= contract -----------------------------------------------------


def test_kernels_that_advertise_out_write_into_it():
    x, w = rand(3, 4), rand(4, 5)
    for op_type, inputs in [
        ("MatMul", [x, w]),
        ("Gemm", [x, w, rand(5)]),
        ("Add", [x, x]),
        ("Mul", [x, x]),
        ("Relu", [x]),
        ("Gelu", [x]),
        ("Sigmoid", [x]),
        ("Tanh", [x]),
        ("Scale", [x]),
    ]:
        spec = ops.get(op_type)
        assert spec.supports_out, op_type
        shape = spec.infer_shape([tuple(i.shape) for i in inputs], {})
        buffer = np.empty(shape, dtype=np.float32)
        result = spec.compute(BACKEND, inputs, {"factor": 2.0}, out=buffer)
        assert result is buffer, f"{op_type} ignored the supplied out buffer"
