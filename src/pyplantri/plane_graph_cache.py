# src/pyplantri/plane_graph_cache.py
from __future__ import annotations

import gzip
import hashlib
import io
import json
import mmap
import operator
import os
import struct
import tempfile
from collections import OrderedDict
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import ExitStack, nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, BinaryIO, Literal, cast, get_args, overload

from .plantri import PlaneDual, PlantriEnumeration, SimpleQuadrangulation

_CACHE_MAGIC = b"PYPLANTRI"
_CACHE_FOOTER_STRUCT = struct.Struct(">9sQ32s")
# Each manifest and decoded chunk is limited to 64 MiB.
_MAX_BLOCK_SIZE = 64 * 1024**2
_MAX_GRAPH_COUNT = (1 << 64) - 1
# The uint8 twin format addresses 256 darts, or 64 quartic vertices.
_MAX_DUAL_VERTEX_COUNT = 64
_INDEX_ENCODING = "stored-index-plus-one-le"
_IMPLICIT_RECORD_ENCODING = "twin-u8-implicit-id"
_MAX_RECORDS_PER_CHUNK = _MAX_BLOCK_SIZE // (4 * _MAX_DUAL_VERTEX_COUNT + 8)

CacheGraphClass = Literal["quartic_multigraph", "simple_quartic"]
CacheIndexMode = Literal["implicit", "explicit"]


@dataclass(frozen=True, slots=True)
class CacheMetadata:
    """Structural metadata for one cache of dual records."""

    dual_vertex_count: int
    graph_count: int
    graph_class: CacheGraphClass


class DualPlaneGraphCache(Sequence[PlaneDual]):
    """Path-backed lazy cache sequence in physical storage order; reopen after replacing the file."""

    __slots__ = (
        "_filepath", "_metadata", "_storage_order", "_compression", "_chunks", "_graph_id_index", "_graph_id_width",
        "_record_size", "_cached_chunks", "_cached_chunk_limit", "_graph_id_struct", "_chunk_size",
    )

    def __init__(
        self, filepath: Path, metadata: CacheMetadata, *, storage_order: str,
        chunk_size: int, compression: Literal["gzip", "none"],
        graph_id_index: tuple[int, bytes] | None, chunks: tuple[tuple[int, int, bytes], ...], cached_chunks: int = 1,
    ) -> None:
        """Initialize from factory-validated layout and cache capacity."""
        self._filepath = filepath
        self._metadata = metadata
        self._storage_order = storage_order
        self._compression = compression
        self._chunks = chunks
        self._graph_id_index = graph_id_index
        self._chunk_size = chunk_size
        # The +1 index sentinel needs eight bytes once the graph count reaches 2**32.
        self._graph_id_width: Literal[0, 4, 8] = 0 if graph_id_index is None else 4 if metadata.graph_count < 2**32 else 8
        self._record_size = 4 * metadata.dual_vertex_count + self._graph_id_width
        self._graph_id_struct = struct.Struct("<I" if self._graph_id_width == 4 else "<Q")
        self._cached_chunks: OrderedDict[int, tuple[bytes, list[PlaneDual | None]]] = OrderedDict()
        self._cached_chunk_limit = cached_chunks

    @property
    def metadata(self) -> CacheMetadata:
        """Return validated public cache metadata."""
        return self._metadata

    @property
    def storage_order(self) -> str:
        """Return the declared record-order name."""
        return self._storage_order

    def __len__(self) -> int:
        """Return the manifest-declared record count."""
        return self._metadata.graph_count

    @overload
    def __getitem__(self, index: int) -> PlaneDual: ...

    @overload
    def __getitem__(self, index: slice) -> list[PlaneDual]: ...

    def __getitem__(self, index: int | slice) -> PlaneDual | list[PlaneDual]:
        """Return one physical record or materialized slice."""
        if isinstance(index, slice):
            return list(self.__iter__(range(len(self))[index]))
        position = range(len(self))[index]
        return next(self.__iter__((position,)))

    def __iter__(
        self, positions: Iterable[int] | None = None, stream: BinaryIO | None = None, *, fresh: bool = False,
    ) -> Iterator[PlaneDual]:
        """Yield checked records, closing internally opened files before each yield."""
        graph_count = self._metadata.graph_count
        if positions is None:
            positions = range(graph_count)
            stream = None
        loaded_chunk: int | None = None
        raw = b""
        graphs: list[PlaneDual | None] = []
        for position in positions:
            chunk_index, local_index = divmod(position, self._chunk_size)
            if chunk_index != loaded_chunk:
                loaded_chunk = chunk_index
                cached = None if fresh else self._cached_chunks.get(chunk_index)
                if cached is not None:
                    self._cached_chunks.move_to_end(chunk_index)
                    raw, graphs = cached
                else:
                    offset, size, digest = self._chunks[chunk_index]
                    with self._filepath.open("rb") if stream is None else nullcontext(stream) as source:
                        source.seek(offset)
                        raw = payload = source.read(size)
                    if len(payload) != size or hashlib.sha256(payload).digest() != digest:
                        raise ValueError(f'cache: invalid chunk {chunk_index} payload ({self._filepath})')
                    chunk_count = min(self._chunk_size, graph_count - chunk_index * self._chunk_size)
                    expected_size = chunk_count * self._record_size
                    if self._compression == "gzip":
                        with gzip.GzipFile(fileobj=io.BytesIO(payload), mode="rb") as decompressor:
                            # One extra byte detects decompression bombs and overlong records.
                            raw = decompressor.read(expected_size + 1)
                        if len(raw) != expected_size:
                            raise ValueError(f'cache: chunk {chunk_index} raw size {len(raw)}!={expected_size} ({self._filepath})')
                    graphs = [None] * chunk_count
                    if not fresh and self._cached_chunk_limit:
                        self._cached_chunks[chunk_index] = raw, graphs
                        if len(self._cached_chunks) > self._cached_chunk_limit:
                            self._cached_chunks.popitem(last=False)
            if (graph := graphs[local_index]) is None:
                width, offset = self._graph_id_width, local_index * self._record_size
                graph_id = self._graph_id_struct.unpack_from(raw, offset)[0] if width else position
                graph = PlaneDual(raw[offset + width : offset + self._record_size], graph_id)
                if graph_id >= graph_count or (
                    self.metadata.graph_class == "simple_quartic"
                    and any(value != 1 for value in graph.edge_multiplicity.values())
                ):
                    raise ValueError(f'cache: invalid {self.metadata.graph_class} record {position}, graph_id={graph_id} ({self._filepath})')
                graphs[local_index] = graph
            yield graph

    def get_by_graph_id(self, graph_id: int) -> PlaneDual:
        """Resolve one dense source Graph ID to its physical record."""
        graph_id = operator.index(graph_id)
        if not 0 <= graph_id < len(self):
            raise KeyError(graph_id)
        descriptor = self._graph_id_index
        if descriptor is None:
            return self[graph_id]
        offset, _ = descriptor
        # Keep the inverse-index entry and an uncached chunk on the same short-lived handle.
        with self._filepath.open("rb") as stream:
            stream.seek(offset + graph_id * self._graph_id_width)
            encoded = self._graph_id_struct.unpack(stream.read(self._graph_id_width))[0]
            if not 1 <= encoded <= len(self):
                raise ValueError(f'cache: invalid stored-index sentinel {encoded} for graph_id={graph_id} ({self._filepath})')
            graph = next(self.__iter__((encoded - 1,), stream))
        if graph.graph_id != graph_id:
            raise ValueError(f'cache: graph_id_index maps {graph_id} to graph {graph.graph_id} ({self._filepath})')
        return graph

    def get_primal_by_graph_id(self, graph_id: int) -> SimpleQuadrangulation:
        """Return the candidate-primal view for one source Graph ID."""
        return SimpleQuadrangulation(self.get_by_graph_id(graph_id))

    def verify_integrity(self) -> None:
        """Freshly verify every chunk, raw record, and inverse index."""
        descriptor = self._graph_id_index
        with ExitStack() as resources:
            index_view: memoryview | None = None
            if descriptor is not None:
                offset, digest = descriptor
                size = self._metadata.graph_count * self._graph_id_width
                if size:
                    index_stream = resources.enter_context(self._filepath.open("rb"))
                    delta = offset % mmap.ALLOCATIONGRANULARITY
                    mapped_index = resources.enter_context(
                        mmap.mmap(index_stream.fileno(), delta + size, access=mmap.ACCESS_READ, offset=offset - delta)
                    )
                    index_view = resources.enter_context(memoryview(mapped_index)[delta : delta + size])
                if hashlib.sha256(index_view or b"").digest() != digest:
                    raise ValueError(f'cache: graph_id_index hash mismatch ({self._filepath})')

            for stored_index, graph in enumerate(self.__iter__(fresh=True)):
                if index_view is not None:
                    encoded = self._graph_id_struct.unpack_from(index_view, graph.graph_id * self._graph_id_width)[0]
                    if encoded != stored_index + 1:
                        raise ValueError(f'cache: graph_id_index mismatch for graph_id={graph.graph_id} ({self._filepath})')


def open_graph_cache(filepath: str | Path, *, cached_chunks: int = 1) -> DualPlaneGraphCache:
    """Open the raw-record cache without reading graph chunks."""
    cached_chunks = operator.index(cached_chunks)
    if cached_chunks < 0:
        raise ValueError(f'cache: invalid cached_chunks={cached_chunks!r} expected=nonnegative int')
    path = Path(filepath).absolute()

    # The footer commits the complete manifest without touching any chunk.
    footer_size = _CACHE_FOOTER_STRUCT.size
    with path.open("rb") as stream:
        file_size = stream.seek(0, io.SEEK_END)
        if file_size < footer_size:
            raise ValueError(f'cache: file shorter than footer ({path})')
        footer_offset = file_size - footer_size
        stream.seek(footer_offset)
        magic, manifest_size, expected_digest = _CACHE_FOOTER_STRUCT.unpack(stream.read(footer_size))
        if magic != _CACHE_MAGIC or manifest_size > min(_MAX_BLOCK_SIZE, footer_offset):
            raise ValueError(f'cache: invalid header: magic={magic!r}, manifest_size={manifest_size} ({path})')
        manifest_offset = footer_offset - manifest_size
        stream.seek(manifest_offset)
        manifest_bytes = stream.read(manifest_size)
        if len(manifest_bytes) != manifest_size or hashlib.sha256(manifest_bytes).digest() != expected_digest:
            raise ValueError(f'cache: invalid manifest payload ({path})')

    # Parse the manifest and prove its exact file partition.
    manifest_data = json.loads(manifest_bytes)
    metadata_data = manifest_data["metadata"]
    metadata = CacheMetadata(
        metadata_data["dual_vertex_count"], metadata_data["graph_count"], metadata_data["graph_class"],
    )
    storage_order = metadata_data.get("storage_order_name", "source")
    for field, low, high in (
        ("dual_vertex_count", 3, _MAX_DUAL_VERTEX_COUNT),
        ("graph_count", 0, _MAX_GRAPH_COUNT),
    ):
        if type(value := getattr(metadata, field)) is not int or not low <= value <= high:
            raise ValueError(f'cache: invalid metadata.{field}={value!r} expected=int[{low},{high}] ({path})')
    if metadata.graph_class not in get_args(CacheGraphClass):
        raise ValueError(f'cache: invalid metadata.graph_class={metadata.graph_class!r} ({path})')
    if not isinstance(storage_order, str):
        raise ValueError(f'cache: invalid storage_order={storage_order!r} ({path})')
    chunk_size, compression = manifest_data["chunk_size"], manifest_data["compression"]
    if type(chunk_size) is not int or not 1 <= chunk_size <= _MAX_RECORDS_PER_CHUNK or compression not in ("gzip", "none"):
        raise ValueError(f'cache: invalid chunk_size={chunk_size!r}, compression={compression!r} ({path})')

    # Optional inverse index: width 0 means implicit Graph IDs (the stored position).
    index_data = manifest_data["graph_id_index"]
    index_width, index_digest, expected_record_encoding = 0, None, _IMPLICIT_RECORD_ENCODING
    if index_data is not None:
        index_width = index_data["width"]
        # The +1 sentinel needs eight bytes exactly when N reaches 2^32.
        expected_width = 4 if metadata.graph_count < 2**32 else 8
        if type(index_width) is not int or index_width != expected_width or index_data["encoding"] != _INDEX_ENCODING:
            raise ValueError(f'cache: invalid graph_id_index ({path})')
        index_digest = bytes.fromhex(index_data["sha256"])
        expected_record_encoding = f"graph-id-u{8 * index_width}-le+twin-u8"
    record_encoding, record_size = manifest_data["record_encoding"], manifest_data["record_size"]
    expected_record_size = 4 * metadata.dual_vertex_count + index_width
    if record_encoding != expected_record_encoding or type(record_size) is not int or record_size != expected_record_size:
        raise ValueError(f'cache: invalid record_size={record_size!r}, record_encoding={record_encoding!r} ({path})')

    chunk_descriptors = manifest_data["chunks"]
    expected_chunk_count = (metadata.graph_count + chunk_size - 1) // chunk_size
    if type(chunk_descriptors) is not list or len(chunk_descriptors) != expected_chunk_count:
        raise ValueError(f'cache: invalid chunk count (expected {expected_chunk_count}) ({path})')

    # Chunks, then the optional inverse index, must tile the payload exactly up to the manifest.
    chunks: list[tuple[int, int, bytes]] = []
    next_offset = 0
    for chunk_index, descriptor in enumerate(chunk_descriptors):
        size, digest_hex = descriptor
        raw_size = min(chunk_size, metadata.graph_count - chunk_index * chunk_size) * record_size
        minimum, maximum = (raw_size, raw_size) if compression == "none" else (1, _MAX_BLOCK_SIZE + 1024**2)
        if type(size) is not int or not minimum <= size <= maximum:
            raise ValueError(f'cache: invalid chunk[{chunk_index}].size={size!r} expected=int[{minimum},{maximum}] ({path})')
        chunks.append((next_offset, size, bytes.fromhex(digest_hex)))
        next_offset += size

    graph_id_index = None if index_digest is None else (next_offset, index_digest)
    payload_end = next_offset + metadata.graph_count * index_width
    if payload_end != manifest_offset:
        raise ValueError(f'cache: payload_end {payload_end}!={manifest_offset} ({path})')
    return DualPlaneGraphCache(
        path, metadata, storage_order=storage_order, chunk_size=chunk_size, compression=compression,
        graph_id_index=graph_id_index, chunks=tuple(chunks), cached_chunks=cached_chunks,
    )


def write_graph_cache(
    graphs: Sequence[PlaneDual] | PlantriEnumeration,
    filepath: str | Path,
    *,
    dual_vertex_count: int | None = None,
    graph_class: CacheGraphClass,
    compression: Literal["gzip", "none"] = "gzip",
    compress_level: int = 1,
    chunk_size: int = 512,
    storage_order_name: str = "source",
    index_mode: CacheIndexMode = "implicit",
    durable: bool = False,
) -> Path:
    """Write dual records in the supplied order; empty input requires dual_vertex_count."""
    if isinstance(graphs, PlantriEnumeration):
        graphs = graphs.graphs

    # Validate cache-format options before filesystem side effects.
    for field, value, valid in (
        ("compression", compression, type(compression) is str and compression in ("gzip", "none")),
        ("chunk_size", chunk_size, type(chunk_size) is int and 1 <= chunk_size <= _MAX_RECORDS_PER_CHUNK),
        ("storage_order_name", storage_order_name, isinstance(storage_order_name, str)),
        ("index_mode", index_mode, type(index_mode) is str and index_mode in ("implicit", "explicit")),
        ("graph_class", graph_class, type(graph_class) is str and graph_class in get_args(CacheGraphClass)),
        ("dual_vertex_count", dual_vertex_count, dual_vertex_count is None or (type(dual_vertex_count) is int and 3 <= dual_vertex_count <= _MAX_DUAL_VERTEX_COUNT)),
    ):
        if not valid:
            raise ValueError(f'cache: invalid {field}={value!r}')
    resolved_path = Path(filepath)
    graph_count = len(graphs)

    # Resolve the header once; each graph constructor already owns its topology and n bounds.
    if dual_vertex_count is None:
        if not graph_count:
            raise ValueError('cache: dual_vertex_count is required for empty input')
        dual_vertex_count = graphs[0].num_vertices
    twin_size = 4 * dual_vertex_count

    # stored_index + 1 reserves zero for duplicate detection in the mmap.
    width: Literal[0, 4, 8] = (4 if graph_count < 2**32 else 8) if index_mode == "explicit" else 0
    resources = ExitStack()
    temporary_path: Path | None = None
    directory_fd: int | None = None
    finalizing = False
    try:
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        # Use a same-directory temporary file for atomic replace; its wrapper is not BinaryIO.
        destination_stream = cast(BinaryIO, resources.enter_context(tempfile.NamedTemporaryFile(mode="wb", dir=resolved_path.parent, suffix=".tmp", delete=False)))
        temporary_path = Path(destination_stream.name)
        index_file: BinaryIO | None = None
        writable_index: mmap.mmap | None = None
        if width:
            # Disk-backed random writes avoid an O(N) in-memory inverse index.
            index_file = cast(BinaryIO, resources.enter_context(tempfile.TemporaryFile(dir=resolved_path.parent)))
            index_file.truncate(graph_count * width)
            if graph_count:
                writable_index = resources.enter_context(mmap.mmap(index_file.fileno(), 0, access=mmap.ACCESS_WRITE))

        # Records: chunked, hashed in storage order, with Graph IDs recorded in the inverse index when explicit.
        chunks: list[tuple[int, str]] = []
        index_struct = struct.Struct("<I" if width == 4 else "<Q")
        for chunk_start in range(0, graph_count, chunk_size):
            raw_chunk = bytearray()
            # Declared-length indexing preserves failure on a prematurely exhausted sequence.
            for stored_index in range(chunk_start, min(chunk_start + chunk_size, graph_count)):
                graph = graphs[stored_index]
                twin, graph_id = graph.twin, graph.graph_id
                if (
                    len(twin) != twin_size
                    or not 0 <= graph_id < graph_count
                    or (graph_class == "simple_quartic" and any(value != 1 for value in graph.edge_multiplicity.values()))
                    or (writable_index is not None and index_struct.unpack_from(writable_index, graph_id * width)[0])
                    or (writable_index is None and graph_id != stored_index)
                ):
                    raise ValueError(f'cache: invalid {graph_class} record {stored_index}, graph_id={graph_id} ({resolved_path})')
                if writable_index is not None:
                    index_struct.pack_into(writable_index, graph_id * width, stored_index + 1)
                    raw_chunk.extend(graph_id.to_bytes(width, "little"))
                raw_chunk.extend(twin)
            # Compress the bounded chunk in one call, without a timestamp or filename.
            payload = gzip.compress(raw_chunk, compresslevel=compress_level, mtime=0) if compression == "gzip" else raw_chunk
            destination_stream.write(payload)
            chunks.append((len(payload), hashlib.sha256(payload).hexdigest()))

        # Inverse index, manifest, and footer.
        graph_id_index: dict[str, object] | None = None
        if index_file is not None:
            if writable_index is not None:
                writable_index.flush()
            # Append and hash the completed index without materializing it.
            index_file.seek(0)
            index_hasher = hashlib.sha256()
            while block := index_file.read(1024 * 1024):
                destination_stream.write(block)
                index_hasher.update(block)
            graph_id_index = {"width": width, "encoding": _INDEX_ENCODING, "sha256": index_hasher.hexdigest()}

        metadata = CacheMetadata(dual_vertex_count, graph_count, graph_class)
        manifest_bytes = json.dumps(
            {
                "metadata": asdict(metadata) | {"storage_order_name": storage_order_name},
                "chunk_size": chunk_size,
                "compression": compression,
                "record_encoding": f"graph-id-u{8 * width}-le+twin-u8" if width else _IMPLICIT_RECORD_ENCODING,
                "record_size": twin_size + width,
                "graph_id_index": graph_id_index,
                "chunks": chunks,
            },
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        if len(manifest_bytes) > _MAX_BLOCK_SIZE:
            raise ValueError(f'cache: manifest size={len(manifest_bytes)} expected<={_MAX_BLOCK_SIZE} ({resolved_path})')
        destination_stream.write(manifest_bytes)
        destination_stream.write(_CACHE_FOOTER_STRUCT.pack(_CACHE_MAGIC, len(manifest_bytes), hashlib.sha256(manifest_bytes).digest()))
        if durable:
            destination_stream.flush()
            os.fsync(destination_stream.fileno())

        # Close every file mapping and stream before replace.
        resources.close()
        temporary_path.replace(resolved_path)
        if durable and os.name == "posix":
            # POSIX directory fsync persists the rename itself after a crash.
            finalizing = True
            directory_fd = os.open(resolved_path.parent, os.O_RDONLY)
            os.fsync(directory_fd)
            closing_fd, directory_fd = directory_fd, None
            os.close(closing_fd)
    except BaseException as error:
        # Cleanup must never replace the primary failure: record its own failure as a note.
        cleanups: list[tuple[str, Callable[..., object], tuple[Any, ...]]] = []
        if directory_fd is not None:
            cleanups.append(("failed to close directory fd", os.close, (directory_fd,)))
        if finalizing:
            # Not a cleanup: this note must follow the directory-fd note and precede the resource notes.
            cleanups.append(("", error.add_note, ("cache: replace committed; durable finalization failed",)))
        cleanups.append(("failed to close temporary resources", resources.close, ()))
        if temporary_path is not None:
            cleanups.append(("failed to remove temporary file", temporary_path.unlink, (True,)))
        for message, cleanup, arguments in cleanups:
            try:
                cleanup(*arguments)
            except BaseException as cleanup_error:
                detail = " ".join(str(cleanup_error).split()) or type(cleanup_error).__name__
                error.add_note(f"cache: {message}: {detail[:240]}")
        raise

    return resolved_path
