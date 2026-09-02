"""Demo models, each paired with a plain-numpy reference implementation.

The references exist to answer the question every inference engine has to
answer: *is it still computing the right thing?* They are written directly from
the maths, with no reference to the graph, the optimizer or the executor, so a
mismatch points at the engine rather than at a shared bug.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from engine.builder import GraphBuilder
from engine.graph import Graph


@dataclass
class DemoModel:
    graph: Graph
    reference: Callable[[np.ndarray], np.ndarray]
    example_input: np.ndarray
    input_name: str
    output_name: str


# -- shared maths ----------------------------------------------------------


def ref_gelu(x: np.ndarray) -> np.ndarray:
    inner = np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3))
    return 0.5 * x * (1.0 + np.tanh(inner))


def ref_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def ref_layernorm(x, weight, bias, eps=1e-5):
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.mean((x - mean) ** 2, axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * weight + bias


# -- MLP -------------------------------------------------------------------


def build_mlp(
    in_features: int = 784,
    hidden: tuple[int, ...] = (256, 256),
    out_features: int = 10,
    batch: int = 8,
    seed: int = 0,
) -> DemoModel:
    """A classic feed-forward classifier: MatMul/Add/Relu stacks."""
    rng = np.random.default_rng(seed)
    sizes = (in_features,) + tuple(hidden)
    weights = [
        (
            (rng.standard_normal((a, b)) / np.sqrt(a)).astype(np.float32),
            rng.standard_normal(b).astype(np.float32) * 0.01,
        )
        for a, b in zip(sizes, sizes[1:])
    ]
    head_w = (rng.standard_normal((sizes[-1], out_features))
              / np.sqrt(sizes[-1])).astype(np.float32)
    head_b = (rng.standard_normal(out_features) * 0.01).astype(np.float32)

    builder = GraphBuilder("mlp")
    x = builder.input("x", (None, in_features))
    value = x
    for index, (w, b) in enumerate(weights):
        value = builder.linear(value, w, b, prefix=f"fc{index}", activation="Relu")
    logits = builder.linear(value, head_w, head_b, prefix="head")
    builder.output(logits, (None, out_features))

    def reference(inputs: np.ndarray) -> np.ndarray:
        h = inputs
        for w, b in weights:
            h = np.maximum(h @ w + b, 0)
        return h @ head_w + head_b

    return DemoModel(
        graph=builder.build(),
        reference=reference,
        example_input=rng.standard_normal((batch, in_features)).astype(np.float32),
        input_name="x",
        output_name=logits,
    )


# -- Transformer block -----------------------------------------------------


def build_transformer_block(
    d_model: int = 128,
    n_heads: int = 4,
    seq_len: int = 32,
    ffn_multiplier: int = 4,
    batch: int = 4,
    seed: int = 0,
) -> DemoModel:
    """One pre-norm transformer block: attention + FFN, with residuals.

    This is the shape check that matters -- it exercises reshape/transpose,
    batched matmul, softmax and layernorm together, which is most of what an
    LLM's forward pass is made of.
    """
    if d_model % n_heads:
        raise ValueError(f"d_model {d_model} is not divisible by n_heads {n_heads}")
    head_dim = d_model // n_heads
    scale = 1.0 / np.sqrt(head_dim)
    d_ffn = d_model * ffn_multiplier

    rng = np.random.default_rng(seed)

    def normal(*shape):
        return (rng.standard_normal(shape) / np.sqrt(shape[0])).astype(np.float32)

    wq, wk, wv, wo = (normal(d_model, d_model) for _ in range(4))
    w1, w2 = normal(d_model, d_ffn), normal(d_ffn, d_model)
    b1 = (rng.standard_normal(d_ffn) * 0.01).astype(np.float32)
    b2 = (rng.standard_normal(d_model) * 0.01).astype(np.float32)
    ln1_w = np.ones(d_model, dtype=np.float32)
    ln1_b = np.zeros(d_model, dtype=np.float32)
    ln2_w = np.ones(d_model, dtype=np.float32)
    ln2_b = np.zeros(d_model, dtype=np.float32)

    builder = GraphBuilder("transformer_block")
    x = builder.input("x", (None, seq_len, d_model))

    def project(source: str, weight: np.ndarray, prefix: str) -> str:
        """(B, S, D) -> (B, H, S, dh)"""
        name = builder.constant(f"{prefix}_w", weight)
        projected = builder.op("MatMul", [source, name])
        split = builder.op(
            "Reshape", [projected], shape=[-1, seq_len, n_heads, head_dim]
        )
        return builder.op("Transpose", [split], perm=[0, 2, 1, 3])

    q = project(x, wq, "q")
    k = project(x, wk, "k")
    v = project(x, wv, "v")

    k_t = builder.op("Transpose", [k], perm=[0, 1, 3, 2])
    scores = builder.op("MatMul", [q, k_t])
    scaled = builder.op("Scale", [scores], factor=float(scale))
    probs = builder.op("Softmax", [scaled], axis=-1)

    context = builder.op("MatMul", [probs, v])                       # (B,H,S,dh)
    merged = builder.op("Transpose", [context], perm=[0, 2, 1, 3])   # (B,S,H,dh)
    flat = builder.op("Reshape", [merged], shape=[-1, seq_len, d_model])

    attn_out = builder.op("MatMul", [flat, builder.constant("o_w", wo)])
    residual1 = builder.op("Add", [x, attn_out])
    normed1 = builder.op(
        "LayerNorm",
        [residual1, builder.constant("ln1_w", ln1_w),
         builder.constant("ln1_b", ln1_b)],
        epsilon=1e-5,
    )

    hidden = builder.op("MatMul", [normed1, builder.constant("ffn_w1", w1)])
    hidden = builder.op("Add", [hidden, builder.constant("ffn_b1", b1)])
    hidden = builder.op("Gelu", [hidden])
    ffn_out = builder.op("MatMul", [hidden, builder.constant("ffn_w2", w2)])
    ffn_out = builder.op("Add", [ffn_out, builder.constant("ffn_b2", b2)])

    residual2 = builder.op("Add", [normed1, ffn_out])
    output = builder.op(
        "LayerNorm",
        [residual2, builder.constant("ln2_w", ln2_w),
         builder.constant("ln2_b", ln2_b)],
        epsilon=1e-5,
    )
    builder.output(output, (None, seq_len, d_model))

    def reference(inputs: np.ndarray) -> np.ndarray:
        b = inputs.shape[0]

        def heads(matrix):
            projected = inputs @ matrix
            return projected.reshape(b, seq_len, n_heads, head_dim).transpose(0, 2, 1, 3)

        qh, kh, vh = heads(wq), heads(wk), heads(wv)
        attention = ref_softmax(qh @ kh.transpose(0, 1, 3, 2) * scale, axis=-1)
        context_ref = attention @ vh
        merged_ref = context_ref.transpose(0, 2, 1, 3).reshape(b, seq_len, d_model)
        hidden_state = ref_layernorm(inputs + merged_ref @ wo, ln1_w, ln1_b)
        ffn = ref_gelu(hidden_state @ w1 + b1) @ w2 + b2
        return ref_layernorm(hidden_state + ffn, ln2_w, ln2_b)

    return DemoModel(
        graph=builder.build(),
        reference=reference,
        example_input=rng.standard_normal((batch, seq_len, d_model)).astype(np.float32),
        input_name="x",
        output_name=output,
    )


BUILDERS = {"mlp": build_mlp, "transformer": build_transformer_block}
