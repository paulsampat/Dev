import json

import numpy as np
import pytest

from engine.format import ALIGNMENT, MAGIC, FormatError, ModelFile, write_model
from engine.loader import load_graph, save_graph
from tools.models import build_mlp


@pytest.fixture
def model_path(tmp_path):
    path = str(tmp_path / "m.iem")
    save_graph(path, build_mlp(hidden=(8,), in_features=4, out_features=2).graph)
    return path


def test_round_trip_preserves_weights(tmp_path):
    tensors = {
        "a": np.arange(12, dtype=np.float32).reshape(3, 4),
        "b": np.array([1, 2, 3], dtype=np.int64),
        "c": np.array([[1, -2]], dtype=np.int8),
    }
    path = str(tmp_path / "w.iem")
    write_model(path, {"graph": {}}, tensors)

    with ModelFile(path) as handle:
        for name, expected in tensors.items():
            loaded = handle.tensor(name)
            assert loaded.dtype == expected.dtype
            assert loaded.shape == expected.shape
            assert np.array_equal(loaded, expected)


def test_weights_are_zero_copy_views_into_the_mapping(tmp_path):
    path = str(tmp_path / "w.iem")
    write_model(path, {"graph": {}}, {"a": np.ones((64, 64), dtype=np.float32)})

    with ModelFile(path) as handle:
        array = handle.tensor("a")
        # A copy would own its memory; a view carries a base object.
        assert array.base is not None
        assert not array.flags["OWNDATA"]


def test_tensors_are_aligned(tmp_path):
    path = str(tmp_path / "w.iem")
    write_model(
        path,
        {"graph": {}},
        {"odd": np.ones(3, dtype=np.int8), "next": np.ones(4, dtype=np.float32)},
    )
    with ModelFile(path) as handle:
        for entry in handle.header["initializers"].values():
            assert entry["offset"] % ALIGNMENT == 0


def test_graph_round_trip_keeps_structure(model_path):
    graph, handle = load_graph(model_path)
    try:
        assert graph.name == "mlp"
        assert [s.name for s in graph.inputs] == ["x"]
        assert graph.inputs[0].shape == (None, 4)
        assert len(graph.nodes) == 5  # 2 layers (matmul/add/relu) + head
        graph.validate()
    finally:
        handle.close()


def test_bad_magic_is_reported(tmp_path):
    path = tmp_path / "bad.iem"
    path.write_bytes(b"NOTAMODEL" + b"\0" * 32)
    with pytest.raises(FormatError, match="bad magic"):
        ModelFile(str(path))


def test_empty_file_is_reported(tmp_path):
    path = tmp_path / "empty.iem"
    path.write_bytes(b"")
    with pytest.raises(FormatError, match="empty"):
        ModelFile(str(path))


def test_truncated_file_is_reported(tmp_path):
    path = tmp_path / "short.iem"
    path.write_bytes(MAGIC + (999_999).to_bytes(8, "little") + b"{}")
    with pytest.raises(FormatError, match="header claims"):
        ModelFile(str(path))


def test_unsupported_version_is_reported(tmp_path):
    header = json.dumps({"format_version": 99, "initializers": {}}).encode()
    path = tmp_path / "future.iem"
    path.write_bytes(MAGIC + len(header).to_bytes(8, "little") + header)
    with pytest.raises(FormatError, match="format version 99"):
        ModelFile(str(path))


def test_invalid_json_header_is_reported(tmp_path):
    header = b"{not json"
    path = tmp_path / "bad.iem"
    path.write_bytes(MAGIC + len(header).to_bytes(8, "little") + header)
    with pytest.raises(FormatError, match="not valid JSON"):
        ModelFile(str(path))


def test_missing_graph_section_is_reported(tmp_path):
    path = str(tmp_path / "nograph.iem")
    write_model(path, {}, {})
    with pytest.raises(FormatError, match="no 'graph' section"):
        load_graph(path)


def test_write_is_atomic_leaving_no_temp_file(tmp_path):
    path = str(tmp_path / "m.iem")
    write_model(path, {"graph": {}}, {"a": np.ones(4, dtype=np.float32)})
    assert not (tmp_path / "m.iem.tmp").exists()
