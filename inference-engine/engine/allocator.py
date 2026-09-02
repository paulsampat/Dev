"""Arena allocator for intermediate tensors.

Allocating a fresh buffer for every intermediate on every call makes the
allocator, not the math, the bottleneck for small models. The executor instead
computes each value's last use, hands buffers back to this arena as soon as
they die, and reuses them for later values.

Blocks are raw ``uint8`` allocations that get viewed as the requested dtype and
shape, so a block allocated for a ``float32`` activation can later back an
``int8`` one of the same byte size. Because shapes are only known at run time
(batch size varies), the pool is keyed by byte size rather than by shape, and
the smallest block that fits is reused.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass

import numpy as np


@dataclass
class ArenaStats:
    allocations: int = 0          # blocks actually malloc'd
    reuses: int = 0               # requests served from the pool
    bytes_allocated: int = 0      # total bytes held by the arena
    bytes_requested: int = 0      # total bytes asked for across requests

    @property
    def reuse_rate(self) -> float:
        total = self.allocations + self.reuses
        return self.reuses / total if total else 0.0

    def __str__(self) -> str:
        return (
            f"{self.allocations} allocations, {self.reuses} reuses "
            f"({self.reuse_rate:.0%} served from pool), "
            f"{self.bytes_allocated / 1e6:.2f} MB held vs "
            f"{self.bytes_requested / 1e6:.2f} MB requested"
        )


@dataclass
class Block:
    """A reusable raw allocation."""

    buffer: np.ndarray            # 1-D uint8
    in_use: bool = False

    @property
    def nbytes(self) -> int:
        return self.buffer.size

    def view(self, shape: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
        nbytes = int(np.prod(shape, dtype=np.int64)) * np.dtype(dtype).itemsize
        return self.buffer[:nbytes].view(dtype).reshape(shape)


class Arena:
    """Size-bucketed pool of reusable buffers."""

    def __init__(self, growth_slack: float = 1.0):
        # ``growth_slack`` over-allocates new blocks so that a slightly larger
        # request later (a bigger batch) still hits the pool instead of
        # allocating again. 1.0 means allocate exactly what was asked for.
        self._growth_slack = growth_slack
        self._free_sizes: list[int] = []      # sorted, parallel to _free_blocks
        self._free_blocks: list[Block] = []
        self._live: list[Block] = []
        # Identities of every buffer this arena has ever handed out, so callers
        # can ask whether an array is backed by poolable memory.
        self._owned: set[int] = set()
        self.stats = ArenaStats()

    def owns(self, obj) -> bool:
        """True if ``obj`` is a buffer belonging to this arena."""
        return id(obj) in self._owned

    def acquire(self, shape: tuple[int, ...], dtype) -> tuple[np.ndarray, Block]:
        dtype = np.dtype(dtype)
        nbytes = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
        self.stats.bytes_requested += nbytes

        index = bisect.bisect_left(self._free_sizes, nbytes)
        if index < len(self._free_blocks):
            block = self._free_blocks.pop(index)
            self._free_sizes.pop(index)
            self.stats.reuses += 1
        else:
            capacity = max(nbytes, int(nbytes * self._growth_slack))
            block = Block(np.empty(capacity, dtype=np.uint8))
            self._owned.add(id(block.buffer))
            self.stats.allocations += 1
            self.stats.bytes_allocated += capacity

        block.in_use = True
        self._live.append(block)
        return block.view(shape, dtype), block

    def release(self, block: Block) -> None:
        if not block.in_use:
            return
        block.in_use = False
        try:
            self._live.remove(block)
        except ValueError:
            pass
        index = bisect.bisect_left(self._free_sizes, block.nbytes)
        self._free_sizes.insert(index, block.nbytes)
        self._free_blocks.insert(index, block)

    def release_all(self) -> None:
        for block in list(self._live):
            self.release(block)

    def reset(self) -> None:
        """Drop every pooled block, returning the memory to the allocator."""
        self._free_sizes.clear()
        self._free_blocks.clear()
        self._live.clear()
        self._owned.clear()
        self.stats = ArenaStats()

    @property
    def held_bytes(self) -> int:
        return self.stats.bytes_allocated
