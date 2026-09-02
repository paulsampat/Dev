"""CPU backend.

Numpy dispatches matmul to the BLAS it was built against, so these kernels sit
at roughly the performance a hand-written CPU engine reaches without writing
its own microkernels. Kernels accept an ``out=`` buffer wherever numpy supports
one, which is what lets the memory planner recycle intermediate buffers.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .base import Backend

_SQRT_2_OVER_PI = np.float32(np.sqrt(2.0 / np.pi))


def _apply_activation(y: np.ndarray, activation: Optional[str]) -> np.ndarray:
    if activation is None:
        return y
    if activation == "relu":
        return np.maximum(y, 0, out=y)
    if activation == "gelu":
        return NumpyBackend.gelu_inplace(y)
    if activation == "sigmoid":
        return _sigmoid_inplace(y)
    if activation == "tanh":
        return np.tanh(y, out=y)
    raise ValueError(f"unknown fused activation {activation!r}")


def _sigmoid_inplace(x: np.ndarray) -> np.ndarray:
    np.negative(x, out=x)
    np.exp(x, out=x)
    np.add(x, 1.0, out=x)
    np.reciprocal(x, out=x)
    return x


class NumpyBackend(Backend):
    name = "numpy"

    # -- allocation ------------------------------------------------------

    def empty(self, shape, dtype):
        return np.empty(shape, dtype=dtype)

    # -- linear algebra --------------------------------------------------

    def matmul(self, a, b, out=None):
        return np.matmul(a, b, out=out)

    def gemm(self, x, w, bias=None, activation=None, out=None):
        y = np.matmul(x, w, out=out)
        if bias is not None:
            np.add(y, bias, out=y)
        return _apply_activation(y, activation)

    def qgemm_int8(self, x, w_q, w_scale, bias=None, activation=None):
        # Dynamic per-row activation quantisation, static per-column weight
        # scales. Accumulate in int32, then rescale back to float32.
        amax = np.max(np.abs(x), axis=-1, keepdims=True)
        x_scale = np.maximum(amax, 1e-12) / 127.0
        x_q = np.rint(x / x_scale).astype(np.int8)

        acc = np.matmul(x_q.astype(np.int32), w_q.astype(np.int32))
        y = acc.astype(np.float32) * (x_scale * w_scale.astype(np.float32))
        if bias is not None:
            y += bias
        return _apply_activation(y, activation)

    # -- elementwise -----------------------------------------------------

    def add(self, a, b, out=None):
        return np.add(a, b, out=out)

    def mul(self, a, b, out=None):
        return np.multiply(a, b, out=out)

    def relu(self, x, out=None):
        return np.maximum(x, 0, out=out)

    @staticmethod
    def gelu_inplace(x: np.ndarray) -> np.ndarray:
        # tanh approximation, matching the formulation used by BERT/GPT-2.
        inner = _SQRT_2_OVER_PI * (x + np.float32(0.044715) * x * x * x)
        np.tanh(inner, out=inner)
        np.add(inner, 1.0, out=inner)
        np.multiply(x, inner, out=x)
        np.multiply(x, 0.5, out=x)
        return x

    def gelu(self, x, out=None):
        if out is None:
            out = np.array(x, dtype=x.dtype, copy=True)
        elif out is not x:
            np.copyto(out, x)
        return self.gelu_inplace(out)

    def sigmoid(self, x, out=None):
        if out is None:
            out = np.array(x, dtype=x.dtype, copy=True)
        elif out is not x:
            np.copyto(out, x)
        return _sigmoid_inplace(out)

    def tanh(self, x, out=None):
        return np.tanh(x, out=out)

    # -- reductions / normalisation --------------------------------------

    def softmax(self, x, axis):
        shifted = x - np.max(x, axis=axis, keepdims=True)
        np.exp(shifted, out=shifted)
        shifted /= np.sum(shifted, axis=axis, keepdims=True)
        return shifted

    def layernorm(self, x, weight, bias, epsilon):
        mean = np.mean(x, axis=-1, keepdims=True)
        centered = x - mean
        var = np.mean(centered * centered, axis=-1, keepdims=True)
        normed = centered / np.sqrt(var + epsilon)
        normed *= weight
        if bias is not None:
            normed += bias
        return normed

    # -- shape / data movement -------------------------------------------

    def reshape(self, x, shape):
        return np.reshape(x, shape)

    def transpose(self, x, perm):
        return np.transpose(x, perm)

    def gather(self, table, indices, axis):
        return np.take(table, indices, axis=axis)

    def cast(self, x, dtype):
        return x.astype(dtype, copy=False)

    def scale(self, x, factor, out=None):
        return np.multiply(x, np.asarray(factor, dtype=x.dtype), out=out)
