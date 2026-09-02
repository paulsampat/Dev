"""Backend interface.

Operators in ``engine.ops`` are written against this interface and never touch
numpy directly, so adding a GPU or accelerator backend means implementing these
kernels -- the graph, optimizer, planner and executor stay unchanged.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional


class Backend(ABC):
    """The set of primitive kernels an execution target must provide."""

    name: str = "abstract"

    # -- allocation ------------------------------------------------------

    @abstractmethod
    def empty(self, shape: tuple[int, ...], dtype: Any) -> Any:
        """Uninitialised buffer of the given shape/dtype."""

    # -- linear algebra --------------------------------------------------

    @abstractmethod
    def matmul(self, a: Any, b: Any, out: Optional[Any] = None) -> Any: ...

    @abstractmethod
    def gemm(
        self,
        x: Any,
        w: Any,
        bias: Optional[Any] = None,
        activation: Optional[str] = None,
        out: Optional[Any] = None,
    ) -> Any:
        """``activation(x @ w + bias)`` as a single kernel."""

    @abstractmethod
    def qgemm_int8(
        self,
        x: Any,
        w_q: Any,
        w_scale: Any,
        bias: Optional[Any] = None,
        activation: Optional[str] = None,
    ) -> Any:
        """Dynamically quantised ``activation(x @ w + bias)``."""

    # -- elementwise -----------------------------------------------------

    @abstractmethod
    def add(self, a: Any, b: Any, out: Optional[Any] = None) -> Any: ...

    @abstractmethod
    def mul(self, a: Any, b: Any, out: Optional[Any] = None) -> Any: ...

    @abstractmethod
    def relu(self, x: Any, out: Optional[Any] = None) -> Any: ...

    @abstractmethod
    def gelu(self, x: Any, out: Optional[Any] = None) -> Any: ...

    @abstractmethod
    def sigmoid(self, x: Any, out: Optional[Any] = None) -> Any: ...

    @abstractmethod
    def tanh(self, x: Any, out: Optional[Any] = None) -> Any: ...

    # -- reductions / normalisation --------------------------------------

    @abstractmethod
    def softmax(self, x: Any, axis: int) -> Any: ...

    @abstractmethod
    def layernorm(
        self, x: Any, weight: Any, bias: Any, epsilon: float
    ) -> Any: ...

    # -- shape / data movement -------------------------------------------

    @abstractmethod
    def reshape(self, x: Any, shape: tuple[int, ...]) -> Any: ...

    @abstractmethod
    def transpose(self, x: Any, perm: tuple[int, ...]) -> Any: ...

    @abstractmethod
    def gather(self, table: Any, indices: Any, axis: int) -> Any: ...

    @abstractmethod
    def cast(self, x: Any, dtype: Any) -> Any: ...

    @abstractmethod
    def scale(self, x: Any, factor: float, out: Optional[Any] = None) -> Any: ...

    # -- host transfer ---------------------------------------------------

    def to_device(self, array: Any) -> Any:
        """Move a host array onto the device. No-op for CPU backends."""
        return array

    def to_host(self, array: Any) -> Any:
        """Move a device array back to the host. No-op for CPU backends."""
        return array
