"""Operator registry.

Every operator declares three things:

* ``compute``      -- how to produce the output, via the backend interface.
* ``infer_shape``  -- output shape from concrete input shapes, used by the
                      memory planner before the kernel runs.
* ``infer_dtype``  -- output dtype, used by shape inference and the optimizer.

``supports_out`` marks kernels that can write into a caller-supplied buffer;
those are the ones the arena allocator can recycle memory for.

All registered operators are single-output, which keeps the planner's liveness
analysis simple. Multi-output ops would need ``infer_shape`` to return a list.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from .backends.base import Backend
from .tensor import ConcreteShape, broadcast_shapes, dtype_from_name

ComputeFn = Callable[..., np.ndarray]
ShapeFn = Callable[[list[ConcreteShape], dict], ConcreteShape]
DTypeFn = Callable[[list[np.dtype], dict], np.dtype]


@dataclass(frozen=True)
class OpSpec:
    name: str
    compute: ComputeFn
    infer_shape: ShapeFn
    infer_dtype: DTypeFn
    min_inputs: int
    max_inputs: int
    supports_out: bool = False

    def check_arity(self, n_inputs: int, node_name: str) -> None:
        if not (self.min_inputs <= n_inputs <= self.max_inputs):
            expected = (
                str(self.min_inputs)
                if self.min_inputs == self.max_inputs
                else f"{self.min_inputs}-{self.max_inputs}"
            )
            raise ValueError(
                f"node {node_name!r} ({self.name}) expects {expected} inputs, "
                f"got {n_inputs}"
            )


REGISTRY: dict[str, OpSpec] = {}


def register(
    name: str,
    *,
    infer_shape: ShapeFn,
    infer_dtype: Optional[DTypeFn] = None,
    min_inputs: int = 1,
    max_inputs: Optional[int] = None,
    supports_out: bool = False,
):
    """Decorator registering a compute function as an operator."""

    def wrap(fn: ComputeFn) -> ComputeFn:
        if name in REGISTRY:
            raise ValueError(f"operator {name!r} is already registered")
        REGISTRY[name] = OpSpec(
            name=name,
            compute=fn,
            infer_shape=infer_shape,
            infer_dtype=infer_dtype or _first_dtype,
            min_inputs=min_inputs,
            max_inputs=max_inputs if max_inputs is not None else min_inputs,
            supports_out=supports_out,
        )
        return fn

    return wrap


def get(op_type: str) -> OpSpec:
    try:
        return REGISTRY[op_type]
    except KeyError:
        raise ValueError(
            f"unknown operator {op_type!r}; registered operators: "
            f"{', '.join(sorted(REGISTRY))}"
        ) from None


# -- shared inference helpers ---------------------------------------------


def _first_dtype(dtypes: list[np.dtype], attrs: dict) -> np.dtype:
    return dtypes[0]


def _float32(dtypes: list[np.dtype], attrs: dict) -> np.dtype:
    return np.dtype(np.float32)


def _same_shape(shapes: list[ConcreteShape], attrs: dict) -> ConcreteShape:
    return shapes[0]


def _broadcast(shapes: list[ConcreteShape], attrs: dict) -> ConcreteShape:
    return broadcast_shapes(shapes[0], shapes[1])


def _matmul_shape(shapes: list[ConcreteShape], attrs: dict) -> ConcreteShape:
    a, b = shapes[0], shapes[1]
    if len(a) < 2 or len(b) < 2:
        raise ValueError(f"MatMul needs rank>=2 operands, got {a} and {b}")
    if a[-1] != b[-2]:
        raise ValueError(
            f"MatMul inner dimensions disagree: {a} @ {b} "
            f"({a[-1]} != {b[-2]})"
        )
    batch = broadcast_shapes(a[:-2], b[:-2])
    return batch + (a[-2], b[-1])


# -- linear algebra --------------------------------------------------------


@register("MatMul", infer_shape=_matmul_shape, min_inputs=2, supports_out=True)
def _matmul(backend: Backend, inputs, attrs, out=None):
    return backend.matmul(inputs[0], inputs[1], out=out)


@register("Gemm", infer_shape=_matmul_shape, min_inputs=2, max_inputs=3,
          supports_out=True)
def _gemm(backend: Backend, inputs, attrs, out=None):
    bias = inputs[2] if len(inputs) > 2 else None
    return backend.gemm(
        inputs[0], inputs[1], bias, attrs.get("activation"), out=out
    )


def _qgemm_shape(shapes: list[ConcreteShape], attrs: dict) -> ConcreteShape:
    # inputs: x, w_q, w_scale, [bias]
    return _matmul_shape([shapes[0], shapes[1]], attrs)


@register("QGemmInt8", infer_shape=_qgemm_shape, infer_dtype=_float32,
          min_inputs=3, max_inputs=4)
def _qgemm(backend: Backend, inputs, attrs, out=None):
    bias = inputs[3] if len(inputs) > 3 else None
    return backend.qgemm_int8(
        inputs[0], inputs[1], inputs[2], bias, attrs.get("activation")
    )


# -- elementwise -----------------------------------------------------------


@register("Add", infer_shape=_broadcast, min_inputs=2, supports_out=True)
def _add(backend: Backend, inputs, attrs, out=None):
    return backend.add(inputs[0], inputs[1], out=out)


@register("Mul", infer_shape=_broadcast, min_inputs=2, supports_out=True)
def _mul(backend: Backend, inputs, attrs, out=None):
    return backend.mul(inputs[0], inputs[1], out=out)


@register("Relu", infer_shape=_same_shape, supports_out=True)
def _relu(backend: Backend, inputs, attrs, out=None):
    return backend.relu(inputs[0], out=out)


@register("Gelu", infer_shape=_same_shape, supports_out=True)
def _gelu(backend: Backend, inputs, attrs, out=None):
    return backend.gelu(inputs[0], out=out)


@register("Sigmoid", infer_shape=_same_shape, supports_out=True)
def _sigmoid(backend: Backend, inputs, attrs, out=None):
    return backend.sigmoid(inputs[0], out=out)


@register("Tanh", infer_shape=_same_shape, supports_out=True)
def _tanh(backend: Backend, inputs, attrs, out=None):
    return backend.tanh(inputs[0], out=out)


@register("Scale", infer_shape=_same_shape, supports_out=True)
def _scale(backend: Backend, inputs, attrs, out=None):
    return backend.scale(inputs[0], float(attrs["factor"]), out=out)


# -- normalisation ---------------------------------------------------------


@register("Softmax", infer_shape=_same_shape)
def _softmax(backend: Backend, inputs, attrs, out=None):
    return backend.softmax(inputs[0], int(attrs.get("axis", -1)))


@register("LayerNorm", infer_shape=_same_shape, min_inputs=2, max_inputs=3)
def _layernorm(backend: Backend, inputs, attrs, out=None):
    bias = inputs[2] if len(inputs) > 2 else None
    return backend.layernorm(
        inputs[0], inputs[1], bias, float(attrs.get("epsilon", 1e-5))
    )


# -- shape / data movement -------------------------------------------------


def _reshape_shape(shapes: list[ConcreteShape], attrs: dict) -> ConcreteShape:
    target = [int(d) for d in attrs["shape"]]
    total = int(np.prod(shapes[0], dtype=np.int64))
    if target.count(-1) > 1:
        raise ValueError(f"Reshape target {target} has more than one -1")
    if -1 in target:
        known = int(np.prod([d for d in target if d != -1], dtype=np.int64))
        if known == 0 or total % known != 0:
            raise ValueError(
                f"cannot reshape {shapes[0]} (={total} elements) into {target}"
            )
        target[target.index(-1)] = total // known
    if int(np.prod(target, dtype=np.int64)) != total:
        raise ValueError(f"cannot reshape {shapes[0]} into {target}")
    return tuple(target)


@register("Reshape", infer_shape=_reshape_shape)
def _reshape(backend: Backend, inputs, attrs, out=None):
    return backend.reshape(
        inputs[0], _reshape_shape([inputs[0].shape], attrs)
    )


def _transpose_shape(shapes: list[ConcreteShape], attrs: dict) -> ConcreteShape:
    shape = shapes[0]
    perm = attrs.get("perm")
    perm = tuple(range(len(shape))[::-1]) if perm is None else tuple(int(p) for p in perm)
    if sorted(perm) != list(range(len(shape))):
        raise ValueError(f"invalid perm {perm} for shape {shape}")
    return tuple(shape[p] for p in perm)


@register("Transpose", infer_shape=_transpose_shape)
def _transpose(backend: Backend, inputs, attrs, out=None):
    shape = inputs[0].shape
    perm = attrs.get("perm")
    perm = tuple(range(len(shape))[::-1]) if perm is None else tuple(int(p) for p in perm)
    return backend.transpose(inputs[0], perm)


def _gather_shape(shapes: list[ConcreteShape], attrs: dict) -> ConcreteShape:
    table, indices = shapes[0], shapes[1]
    axis = int(attrs.get("axis", 0)) % max(len(table), 1)
    return table[:axis] + indices + table[axis + 1:]


def _gather_dtype(dtypes: list[np.dtype], attrs: dict) -> np.dtype:
    return dtypes[0]


@register("Gather", infer_shape=_gather_shape, infer_dtype=_gather_dtype,
          min_inputs=2)
def _gather(backend: Backend, inputs, attrs, out=None):
    return backend.gather(inputs[0], inputs[1], int(attrs.get("axis", 0)))


def _cast_dtype(dtypes: list[np.dtype], attrs: dict) -> np.dtype:
    return dtype_from_name(attrs["to"])


@register("Cast", infer_shape=_same_shape, infer_dtype=_cast_dtype)
def _cast(backend: Backend, inputs, attrs, out=None):
    return backend.cast(inputs[0], dtype_from_name(attrs["to"]))
