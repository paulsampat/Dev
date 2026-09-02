"""Tensor metadata and dtype handling.

Runtime buffers are plain ``numpy.ndarray`` objects -- numpy is the engine's
BLAS. What lives here is the *static* description of a tensor (dtype + shape,
possibly with dynamic dimensions) that the graph, the shape inference pass and
the memory planner reason about before any data exists.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

# Dimensions that are only known at run time (typically batch) are ``None``.
Dim = Optional[int]
Shape = tuple[Dim, ...]
ConcreteShape = tuple[int, ...]

_DTYPES: dict[str, np.dtype] = {
    "float32": np.dtype(np.float32),
    "float16": np.dtype(np.float16),
    "int64": np.dtype(np.int64),
    "int32": np.dtype(np.int32),
    "int8": np.dtype(np.int8),
    "bool": np.dtype(np.bool_),
}


def dtype_from_name(name: str) -> np.dtype:
    try:
        return _DTYPES[name]
    except KeyError:
        raise ValueError(
            f"unsupported dtype {name!r}; known dtypes: {sorted(_DTYPES)}"
        ) from None


def dtype_to_name(dtype: np.dtype) -> str:
    dtype = np.dtype(dtype)
    for name, known in _DTYPES.items():
        if dtype == known:
            return name
    raise ValueError(f"unsupported dtype {dtype!r}")


@dataclass(frozen=True)
class TensorSpec:
    """Static description of a tensor value flowing through the graph."""

    name: str
    dtype: np.dtype
    shape: Shape

    @property
    def rank(self) -> int:
        return len(self.shape)

    @property
    def is_dynamic(self) -> bool:
        return any(d is None for d in self.shape)

    def nbytes_for(self, shape: ConcreteShape) -> int:
        return int(np.prod(shape, dtype=np.int64)) * self.dtype.itemsize

    def matches(self, array: np.ndarray) -> Optional[str]:
        """Return an error message if ``array`` does not satisfy this spec."""
        if array.dtype != self.dtype:
            return (
                f"{self.name}: expected dtype {dtype_to_name(self.dtype)}, "
                f"got {array.dtype}"
            )
        if array.ndim != self.rank:
            return (
                f"{self.name}: expected rank {self.rank} tensor with shape "
                f"{format_shape(self.shape)}, got shape {array.shape}"
            )
        for axis, (expected, actual) in enumerate(zip(self.shape, array.shape)):
            if expected is not None and expected != actual:
                return (
                    f"{self.name}: expected shape {format_shape(self.shape)}, "
                    f"got {array.shape} (axis {axis})"
                )
        return None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "dtype": dtype_to_name(self.dtype),
            "shape": [None if d is None else int(d) for d in self.shape],
        }

    @staticmethod
    def from_dict(data: dict) -> "TensorSpec":
        return TensorSpec(
            name=data["name"],
            dtype=dtype_from_name(data["dtype"]),
            shape=tuple(None if d is None else int(d) for d in data["shape"]),
        )


def format_shape(shape: Sequence[Dim]) -> str:
    return "(" + ", ".join("?" if d is None else str(d) for d in shape) + ")"


def broadcast_shapes(a: ConcreteShape, b: ConcreteShape) -> ConcreteShape:
    """Numpy broadcasting rules, used by shape inference."""
    rank = max(len(a), len(b))
    # Broadcasting aligns from the trailing dimension, so missing leading
    # dimensions are padded with 1 on the left.
    left = (1,) * (rank - len(a)) + tuple(a)
    right = (1,) * (rank - len(b)) + tuple(b)

    out: list[int] = []
    for x, y in zip(left, right):
        if x == y or y == 1:
            out.append(x)
        elif x == 1:
            out.append(y)
        else:
            raise ValueError(f"shapes {a} and {b} are not broadcastable")
    return tuple(out)
