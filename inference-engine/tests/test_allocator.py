import numpy as np

from engine.allocator import Arena


def test_a_released_block_is_reused():
    arena = Arena()
    _, block = arena.acquire((16, 16), np.float32)
    arena.release(block)
    _, again = arena.acquire((16, 16), np.float32)

    assert again is block
    assert arena.stats.allocations == 1
    assert arena.stats.reuses == 1


def test_blocks_in_use_are_never_handed_out_twice():
    arena = Arena()
    first, _ = arena.acquire((8, 8), np.float32)
    second, _ = arena.acquire((8, 8), np.float32)

    assert not np.shares_memory(first, second)
    assert arena.stats.allocations == 2


def test_a_block_can_be_reused_at_a_different_dtype_and_shape():
    arena = Arena()
    _, block = arena.acquire((64,), np.float32)  # 256 bytes
    arena.release(block)
    view, again = arena.acquire((4, 16), np.int8)  # 64 bytes, fits

    assert again is block
    assert view.shape == (4, 16)
    assert view.dtype == np.int8


def test_the_smallest_sufficient_block_is_chosen():
    arena = Arena()
    _, small = arena.acquire((10,), np.float32)   # 40 bytes
    _, large = arena.acquire((1000,), np.float32)  # 4000 bytes
    arena.release(large)
    arena.release(small)

    _, chosen = arena.acquire((8,), np.float32)   # 32 bytes
    assert chosen is small, "arena should not waste a large block on a small ask"


def test_a_request_larger_than_every_free_block_allocates():
    arena = Arena()
    _, block = arena.acquire((10,), np.float32)
    arena.release(block)
    _, bigger = arena.acquire((1000,), np.float32)

    assert bigger is not block
    assert arena.stats.allocations == 2


def test_views_are_writable_and_correctly_shaped():
    arena = Arena()
    view, _ = arena.acquire((3, 4), np.float32)
    view[:] = 5.0

    assert view.shape == (3, 4)
    assert view.dtype == np.float32
    assert np.all(view == 5.0)


def test_growth_slack_lets_a_slightly_bigger_request_hit_the_pool():
    arena = Arena(growth_slack=2.0)
    _, block = arena.acquire((100,), np.float32)
    arena.release(block)
    _, again = arena.acquire((150,), np.float32)

    assert again is block
    assert arena.stats.reuses == 1


def test_release_all_returns_every_live_block():
    arena = Arena()
    for _ in range(3):
        arena.acquire((32,), np.float32)
    arena.release_all()

    for _ in range(3):
        arena.acquire((32,), np.float32)
    assert arena.stats.allocations == 3
    assert arena.stats.reuses == 3


def test_double_release_is_a_no_op():
    arena = Arena()
    _, block = arena.acquire((8,), np.float32)
    arena.release(block)
    arena.release(block)

    _, a = arena.acquire((8,), np.float32)
    _, b = arena.acquire((8,), np.float32)
    assert a is not b, "a doubly-released block must not be handed out twice"


def test_owns_identifies_arena_memory():
    arena = Arena()
    view, block = arena.acquire((8,), np.float32)

    assert arena.owns(block.buffer)
    assert arena.owns(view.base if view.base is not None else view)
    assert not arena.owns(np.zeros(8, dtype=np.float32))


def test_reset_drops_everything():
    arena = Arena()
    arena.acquire((8,), np.float32)
    arena.reset()

    assert arena.stats.allocations == 0
    assert arena.held_bytes == 0


def test_stats_report_reuse_rate():
    arena = Arena()
    _, block = arena.acquire((8,), np.float32)
    arena.release(block)
    arena.acquire((8,), np.float32)

    assert arena.stats.reuse_rate == 0.5
    assert "reuses" in str(arena.stats)
