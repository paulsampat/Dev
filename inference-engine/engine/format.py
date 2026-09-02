"""The ``.iem`` model file format: JSON header + raw weight blob.

Layout::

    magic       8 bytes   b"IEMODEL\\0"
    header_len  8 bytes   little-endian uint64
    header      JSON, utf-8, header_len bytes
    padding     to a 64-byte boundary
    blob        raw tensor bytes, each tensor 64-byte aligned

The header describes the graph structure and, for each initializer, its dtype,
shape and (offset, nbytes) into the blob. Keeping weights as one contiguous
blob is what makes zero-copy loading possible: the file is mmap'd once and each
weight becomes a numpy view into that mapping rather than a fresh allocation,
so a 400 MB model costs no heap and starts instantly.

This mirrors how safetensors and GGUF are laid out, for the same reasons.
"""

from __future__ import annotations

import json
import mmap
import os
from typing import BinaryIO

import numpy as np

MAGIC = b"IEMODEL\0"
FORMAT_VERSION = 1
ALIGNMENT = 64


class FormatError(ValueError):
    """Raised when a model file is malformed or of an unsupported version."""


def _pad_to_alignment(fh: BinaryIO) -> None:
    remainder = fh.tell() % ALIGNMENT
    if remainder:
        fh.write(b"\0" * (ALIGNMENT - remainder))


def write_model(path: str, header: dict, tensors: dict[str, np.ndarray]) -> None:
    """Write ``header`` plus the weight blob, filling in tensor offsets.

    ``header['initializers']`` is replaced with the offset table produced here,
    so callers pass the graph structure and this function owns the layout.
    """
    offsets: dict[str, dict] = {}
    blob_parts: list[bytes] = []
    cursor = 0

    for name, array in tensors.items():
        array = np.ascontiguousarray(array)
        padding = (-cursor) % ALIGNMENT
        if padding:
            blob_parts.append(b"\0" * padding)
            cursor += padding
        data = array.tobytes()
        offsets[name] = {
            "dtype": array.dtype.name,
            "shape": [int(d) for d in array.shape],
            "offset": cursor,
            "nbytes": len(data),
        }
        blob_parts.append(data)
        cursor += len(data)

    full_header = dict(header)
    full_header["format_version"] = FORMAT_VERSION
    full_header["initializers"] = offsets
    encoded = json.dumps(full_header).encode("utf-8")

    tmp_path = f"{path}.tmp"
    with open(tmp_path, "wb") as fh:
        fh.write(MAGIC)
        fh.write(len(encoded).to_bytes(8, "little"))
        fh.write(encoded)
        _pad_to_alignment(fh)
        for part in blob_parts:
            fh.write(part)
    os.replace(tmp_path, path)


class ModelFile:
    """Memory-mapped read side of a ``.iem`` file.

    Weight arrays are views into the mapping and stay valid until ``close()``.
    """

    def __init__(self, path: str):
        self.path = path
        self._file = open(path, "rb")
        try:
            self._map = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        except ValueError as exc:  # empty file
            self._file.close()
            raise FormatError(f"{path}: file is empty") from exc

        try:
            self.header, self._blob_start = self._read_header()
        except Exception:
            self.close()
            raise

    def _read_header(self) -> tuple[dict, int]:
        if len(self._map) < 16:
            raise FormatError(f"{self.path}: file is too short to be a model")
        if self._map[:8] != MAGIC:
            raise FormatError(
                f"{self.path}: bad magic {bytes(self._map[:8])!r}, "
                f"expected {MAGIC!r}"
            )
        header_len = int.from_bytes(self._map[8:16], "little")
        end = 16 + header_len
        if end > len(self._map):
            raise FormatError(
                f"{self.path}: header claims {header_len} bytes but the file "
                f"holds {len(self._map) - 16}"
            )
        try:
            header = json.loads(bytes(self._map[16:end]).decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise FormatError(f"{self.path}: header is not valid JSON: {exc}") from exc

        version = header.get("format_version")
        if version != FORMAT_VERSION:
            raise FormatError(
                f"{self.path}: format version {version} is not supported "
                f"(this build reads version {FORMAT_VERSION})"
            )
        return header, end + ((-end) % ALIGNMENT)

    def tensor(self, name: str) -> np.ndarray:
        """Zero-copy view of one initializer."""
        entry = self.header["initializers"][name]
        start = self._blob_start + int(entry["offset"])
        stop = start + int(entry["nbytes"])
        if stop > len(self._map):
            raise FormatError(
                f"{self.path}: initializer {name!r} runs past the end of the file"
            )
        dtype = np.dtype(entry["dtype"])
        nbytes = int(entry["nbytes"])
        if nbytes % dtype.itemsize:
            raise FormatError(
                f"{self.path}: initializer {name!r} spans {nbytes} bytes, which "
                f"is not a whole number of {dtype} elements"
            )
        # An explicit count is required: the tail of the mapping past this
        # tensor is not necessarily a whole number of elements.
        array = np.frombuffer(
            self._map, dtype=dtype, count=nbytes // dtype.itemsize, offset=start
        )
        return array.reshape(tuple(entry["shape"]))

    def tensors(self) -> dict[str, np.ndarray]:
        return {name: self.tensor(name) for name in self.header["initializers"]}

    def close(self) -> None:
        mapping = getattr(self, "_map", None)
        if mapping is not None and not mapping.closed:
            try:
                mapping.close()
            except BufferError:
                # Weight views into this mapping are still alive. Unmapping now
                # would leave them dangling, so leave it to the garbage
                # collector: the mapping is released once the last view dies.
                pass
        handle = getattr(self, "_file", None)
        if handle is not None and not handle.closed:
            handle.close()

    def __enter__(self) -> "ModelFile":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()
