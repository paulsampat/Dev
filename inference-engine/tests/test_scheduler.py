import threading

import numpy as np
import pytest

from engine.api import InferenceSession
from engine.scheduler import BatchScheduler
from tools.models import build_mlp

RNG = np.random.default_rng(13)


@pytest.fixture
def session():
    demo = build_mlp(in_features=16, hidden=(32,), out_features=4)
    with InferenceSession(demo.graph) as handle:
        handle.demo = demo
        yield handle


def row(features=16):
    return RNG.standard_normal((1, features)).astype(np.float32)


def test_a_single_request_matches_a_direct_call(session):
    feeds = {"x": row()}
    expected = session.run(feeds)[session.outputs[0].name]

    with BatchScheduler(session, max_wait_ms=1) as scheduler:
        result = scheduler.run(feeds, timeout=5)[session.outputs[0].name]
    assert np.allclose(result, expected, atol=1e-5)


def test_concurrent_requests_each_get_their_own_answer(session):
    name = session.outputs[0].name
    requests = [{"x": row()} for _ in range(12)]
    expected = [session.run(f)[name] for f in requests]

    with BatchScheduler(session, max_batch_size=8, max_wait_ms=20) as scheduler:
        futures = [scheduler.submit(f) for f in requests]
        results = [f.result(timeout=10)[name] for f in futures]

    for got, want in zip(results, expected):
        assert np.allclose(got, want, atol=1e-5)


def test_requests_arriving_together_are_actually_batched(session):
    with BatchScheduler(session, max_batch_size=8, max_wait_ms=50) as scheduler:
        futures = [scheduler.submit({"x": row()}) for _ in range(8)]
        for future in futures:
            future.result(timeout=10)

    assert scheduler.stats.requests == 8
    assert scheduler.stats.batches < 8, "requests were not batched at all"
    assert scheduler.stats.mean_batch_size > 1


def test_a_batch_never_exceeds_the_configured_maximum(session):
    with BatchScheduler(session, max_batch_size=3, max_wait_ms=30) as scheduler:
        futures = [scheduler.submit({"x": row()}) for _ in range(9)]
        for future in futures:
            future.result(timeout=10)

    assert scheduler.stats.batches >= 3  # 9 rows cannot fit in fewer than 3
    assert scheduler.stats.rows == 9


def test_multi_row_requests_are_split_back_correctly(session):
    name = session.outputs[0].name
    feeds = [{"x": RNG.standard_normal((n, 16)).astype(np.float32)} for n in (1, 2, 3)]
    expected = [session.run(f)[name] for f in feeds]

    with BatchScheduler(session, max_batch_size=8, max_wait_ms=30) as scheduler:
        futures = [scheduler.submit(f) for f in feeds]
        results = [f.result(timeout=10)[name] for f in futures]

    for got, want in zip(results, expected):
        assert got.shape == want.shape
        assert np.allclose(got, want, atol=1e-5)


def test_a_bad_request_surfaces_its_error_to_the_caller(session):
    with BatchScheduler(session, max_wait_ms=1) as scheduler:
        future = scheduler.submit({"x": RNG.standard_normal((1, 999)).astype(np.float32)})
        with pytest.raises(ValueError, match="axis 1"):
            future.result(timeout=5)


def test_inconsistent_batch_dimensions_are_rejected_at_submit(session):
    with BatchScheduler(session, max_wait_ms=1) as scheduler:
        with pytest.raises(ValueError, match="disagree on batch size"):
            scheduler.submit({"x": row(), "y": RNG.standard_normal((2, 4))})


def test_submitting_before_start_is_an_error(session):
    scheduler = BatchScheduler(session)
    with pytest.raises(RuntimeError, match="not running"):
        scheduler.submit({"x": row()})


def test_stop_is_idempotent(session):
    scheduler = BatchScheduler(session).start()
    scheduler.stop()
    scheduler.stop()


def test_invalid_configuration_is_rejected(session):
    with pytest.raises(ValueError, match="max_batch_size"):
        BatchScheduler(session, max_batch_size=0)
    with pytest.raises(ValueError, match="batch_axis"):
        BatchScheduler(session, batch_axis=1)


def test_requests_from_many_threads_are_all_served(session):
    name = session.outputs[0].name
    results = {}
    errors = []

    def worker(index):
        try:
            feeds = {"x": row()}
            expected = session.run(feeds)[name]
            got = scheduler.run(feeds, timeout=10)[name]
            results[index] = np.allclose(got, expected, atol=1e-5)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    with BatchScheduler(session, max_batch_size=4, max_wait_ms=10) as scheduler:
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(16)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=20)

    assert not errors
    assert len(results) == 16 and all(results.values())


def test_stats_are_reportable(session):
    with BatchScheduler(session, max_wait_ms=1) as scheduler:
        scheduler.run({"x": row()}, timeout=5)
    assert "requests" in str(scheduler.stats)
