# plantri.py
from __future__ import annotations

import os
import stat
import subprocess
import tempfile
import time
from collections.abc import Generator, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from enum import Enum
from functools import partial
from importlib.resources import as_file, files
from io import BufferedReader
from pathlib import Path
from threading import TIMEOUT_MAX
from types import MappingProxyType, TracebackType
from typing import Any, BinaryIO, cast

_PLANTRI_MAXN = 64  # plantri.c MAXN

# plantri_sqs.c requires 5 <= maxnv < MAXN primal vertices; |V(G*)| = |V(G)| - 2.
MIN_DUAL_VERTEX_COUNT = 3
MAX_DUAL_VERTEX_COUNT = (_PLANTRI_MAXN - 1) - 2


@dataclass(frozen=True, slots=True)
class PlaneDual:
    """Candidate plane dual ``G*`` of a simple spherical quadrangulation ``G``."""

    twin: bytes
    graph_id: int = field(default=0, compare=False)
    _right_faces: bytes | None = field(default=None, init=False, repr=False, compare=False)
    _faces: tuple[tuple[int, ...], ...] | None = field(default=None, init=False, repr=False, compare=False)
    _edge_multiplicity: Mapping[tuple[int, int], int] | None = field(default=None, init=False, repr=False, compare=False)
    _face_size_sequence: tuple[int, ...] | bytes | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Validate persistent fields and common-family topology membership."""
        twin = self.twin
        if type(twin) is not bytes or type(self.graph_id) is not int:
            raise TypeError("twin must be bytes and graph_id must be int")

        dart_count = len(twin)
        vertex_count = dart_count // 4
        valid = self.graph_id >= 0 and dart_count % 4 == 0 and 4 * MIN_DUAL_VERTEX_COUNT <= dart_count <= 256
        if valid:
            valid = all(
                opposite < dart_count and opposite != dart and twin[opposite] == dart
                for dart, opposite in enumerate(twin)
            )
        if valid:
            reached = {0}
            pending = [0]
            while pending:
                vertex = pending.pop()
                for opposite in twin[4 * vertex : 4 * vertex + 4]:
                    neighbor = opposite // 4
                    if neighbor not in reached:
                        reached.add(neighbor)
                        pending.append(neighbor)

            # Dual right-face orbits are the vertices of the paired primal map.
            face_by_dart = [-1] * dart_count
            face_count = 0
            for start_dart in range(dart_count):
                if face_by_dart[start_dart] < 0:
                    dart = start_dart
                    while face_by_dart[dart] < 0:
                        face_by_dart[dart] = face_count
                        opposite = twin[dart]
                        dart = 4 * (opposite // 4) + (opposite % 4 - 1) % 4
                    face_count += 1

            # A connected spherical quartic map has no bridges, so its primal has no loops.
            primal_edges: set[tuple[int, int]] = set()
            for dart, opposite in enumerate(twin):
                if dart < opposite:
                    first, second = face_by_dart[dart], face_by_dart[opposite]
                    primal_edges.add((first, second) if first < second else (second, first))
            valid = len(reached) == vertex_count and face_count == vertex_count + 2 and len(primal_edges) == dart_count // 2
        if not valid:
            raise ValueError("graph_id must be non-negative and twin must encode a spherical dual of a simple quadrangulation")

    @classmethod
    def _from_filter(
        cls,
        twin: bytes,
        graph_id: int,
        face_size_profile: bytes,
        _new=object.__new__,
        _set=object.__setattr__,
    ) -> PlaneDual:
        """Construct from one trusted record emitted by the bundled C FILTER."""
        dual = _new(cls)
        _set(dual, "twin", twin)
        _set(dual, "graph_id", graph_id)
        _set(dual, "_right_faces", None)
        _set(dual, "_faces", None)
        _set(dual, "_edge_multiplicity", None)
        _set(dual, "_face_size_sequence", face_size_profile)
        return dual

    def __reduce__(self) -> tuple[type[PlaneDual], tuple[bytes, int]]:
        """Serialize only the persistent twin and namespace-local Graph ID."""
        return type(self), (self.twin, self.graph_id)

    @property
    def num_vertices(self) -> int:
        """Return the number of candidate-dual vertices."""
        return len(self.twin) // 4

    @property
    def embedding(self) -> tuple[tuple[int, ...], ...]:
        """Return vertex-indexed exterior-view-CW candidate-dual rotations."""
        twin = self.twin
        return tuple(
            tuple(opposite // 4 for opposite in twin[base : base + 4])
            for base in range(0, len(twin), 4)
        )

    @property
    def right_faces(self) -> bytes:
        """Return the right face of every dart, labelled in ``faces`` order."""
        if self._right_faces is None:
            self.faces  # The single orbit walk records the dart labels as well.
        return cast(bytes, self._right_faces)

    @property
    def faces(self) -> tuple[tuple[int, ...], ...]:
        """Return candidate-dual right-face vertex cycles in first-dart order."""
        faces = self._faces
        if faces is None:
            labels = [-1] * len(self.twin)
            cycles: list[tuple[int, ...]] = []
            for start_dart in range(len(self.twin)):
                if labels[start_dart] < 0:
                    cycle: list[int] = []
                    dart = start_dart
                    while labels[dart] < 0:
                        labels[dart] = len(cycles)
                        cycle.append(self.vertex(dart))
                        dart = self.right_face_next(dart)
                    cycles.append(tuple(cycle))
            faces = tuple(cycles)
            object.__setattr__(self, "_right_faces", bytes(labels))
            object.__setattr__(self, "_faces", faces)
        return faces

    @property
    def num_faces(self) -> int:
        """Return the number of candidate-dual faces."""
        return self.num_vertices + 2

    @property
    def edge_multiplicity(self) -> Mapping[tuple[int, int], int]:
        """Return multiplicities of normalized candidate-dual support edges."""
        multiplicities = self._edge_multiplicity
        if multiplicities is None:
            counts: dict[tuple[int, int], int] = {}
            for dart, twin_dart in enumerate(self.twin):
                # The lower dart counts each twin pair once and orders its endpoints.
                if dart > twin_dart:
                    continue
                edge = (dart // 4, twin_dart // 4)
                counts[edge] = counts.get(edge, 0) + 1
            multiplicities = MappingProxyType(dict(sorted(counts.items())))
            object.__setattr__(self, "_edge_multiplicity", multiplicities)
        return multiplicities

    @property
    def support_edges(self) -> tuple[tuple[int, int], ...]:
        """Return normalized candidate-dual support edges in lexicographic order."""
        return tuple(self.edge_multiplicity)

    @property
    def double_edges(self) -> frozenset[tuple[int, int]]:
        """Return dual support edges of multiplicity two."""
        return frozenset(
            edge for edge, multiplicity in self.edge_multiplicity.items()
            if multiplicity == 2
        )

    @property
    def face_size_sequence(self) -> tuple[int, ...]:
        """Return candidate-dual face sizes in nonincreasing order."""
        sequence = self._face_size_sequence
        if type(sequence) is not tuple:
            sequence = (
                tuple(sorted(map(len, self.faces), reverse=True))
                if sequence is None
                else tuple(sequence)
            )
            object.__setattr__(self, "_face_size_sequence", sequence)
        return sequence

    @staticmethod
    def vertex(dart: int) -> int:
        """Return the source vertex of a valid dart."""
        return dart // 4

    @staticmethod
    def next_at_vertex(dart: int) -> int:
        """Return the next valid dart in its exterior-view-CW vertex rotation."""
        return 4 * (dart // 4) + (dart + 1) % 4

    @staticmethod
    def prev_at_vertex(dart: int) -> int:
        """Return the previous valid dart in its exterior-view-CW vertex rotation."""
        return 4 * (dart // 4) + (dart - 1) % 4

    def right_face_next(self, dart: int) -> int:
        """Return the next valid dart along the face on a dart's right."""
        return self.prev_at_vertex(self.twin[dart])

    def right_face(self, dart: int) -> int:
        """Return the face on the right of a valid dart."""
        return self.right_faces[dart]

    def neighbor(self, dart: int) -> int:
        """Return the target vertex of a valid dart."""
        return self.vertex(self.twin[dart])


class PrimalMinimumDegree(Enum):
    """Minimum-degree policy for plantri's primal quadrangulation ``G``.

    - ``AT_LEAST_2``: explicit ``-q -c2 -m2``.
    - ``AT_LEAST_3``: ``-q -c2``; default minimum degree 3; ``-m3`` omitted.
    """

    AT_LEAST_2 = 2
    AT_LEAST_3 = 3

    @property
    def _plantri_switches(self) -> tuple[str, ...]:
        return ("-q", "-c2", "-m2") if self is self.AT_LEAST_2 else ("-q", "-c2")

    @property
    def _minimum_nonempty_dual_vertex_count(self) -> int:
        return MIN_DUAL_VERTEX_COUNT if self is self.AT_LEAST_2 else 6


@dataclass(frozen=True, slots=True)
class SimpleQuadrangulation:
    """Candidate primal ``G`` viewed over the darts of a validated ``PlaneDual``."""

    dual: PlaneDual = field(repr=False)
    _embedding: tuple[tuple[int, ...], ...] | None = field(default=None, init=False, repr=False, compare=False)
    _faces: tuple[tuple[int, ...], ...] | None = field(default=None, init=False, repr=False, compare=False)

    @property
    def twin(self) -> bytes:
        """Return the dart involution shared with ``G*``."""
        return self.dual.twin

    @property
    def graph_id(self) -> int:
        """Return the Graph ID shared with ``G*``."""
        return self.dual.graph_id

    @property
    def num_vertices(self) -> int:
        """Return the number of candidate-primal vertices, ``|V(G*)| + 2``."""
        return self.dual.num_vertices + 2

    @property
    def num_edges(self) -> int:
        """Return the number of candidate-primal edges, one per dual edge."""
        return len(self.dual.twin) // 2

    @property
    def num_faces(self) -> int:
        """Return the number of candidate-primal faces, one per dual vertex."""
        return self.dual.num_vertices

    @property
    def embedding(self) -> tuple[tuple[int, ...], ...]:
        """Return vertex-indexed exterior-view-CW candidate-primal rotations."""
        embedding = self._embedding
        if embedding is None:
            # Each incident 4-cycle contributes the neighbour preceding the vertex on it;
            # the vertex's dual face lists those 4-cycles in rotation order.
            faces = self.faces
            embedding = tuple(
                tuple(faces[face][faces[face].index(vertex) - 1] for face in incident_faces)
                for vertex, incident_faces in enumerate(self.dual.faces)
            )
            object.__setattr__(self, "_embedding", embedding)
        return embedding

    @property
    def faces(self) -> tuple[tuple[int, ...], ...]:
        """Return candidate-primal 4-cycles indexed by candidate-dual vertices."""
        faces = self._faces
        if faces is None:
            twin, labels = self.twin, self.dual.right_faces
            faces = tuple(
                tuple(labels[opposite] for opposite in twin[base : base + 4])
                for base in range(0, len(twin), 4)
            )
            object.__setattr__(self, "_faces", faces)
        return faces

    @property
    def edges(self) -> tuple[tuple[int, int], ...]:
        """Return candidate-primal edges as endpoint-sorted pairs in lexicographic order."""
        right_faces = self.dual.right_faces
        return tuple(sorted(
            (min(right_faces[dart], right_faces[opposite]), max(right_faces[dart], right_faces[opposite]))
            for dart, opposite in enumerate(self.twin)
            if dart < opposite
        ))

    @property
    def degrees(self) -> tuple[int, ...]:
        """Return vertex-indexed candidate-primal degrees."""
        return tuple(map(len, self.dual.faces))

    @property
    def degree_sequence(self) -> tuple[int, ...]:
        """Return candidate-primal degrees in nonincreasing order."""
        return self.dual.face_size_sequence

    @property
    def min_degree(self) -> int:
        """Return the minimum candidate-primal degree; 3 or more means ``G*`` is digon-free."""
        return self.degree_sequence[-1]

    def vertex(self, dart: int) -> int:
        """Return the source vertex of a valid dart."""
        return self.dual.right_faces[dart]

    def next_at_vertex(self, dart: int) -> int:
        """Return the next valid dart in its exterior-view-CW vertex rotation."""
        return self.dual.right_face_next(dart)

    def prev_at_vertex(self, dart: int) -> int:
        """Return the previous valid dart in its exterior-view-CW vertex rotation."""
        return self.twin[self.dual.next_at_vertex(dart)]

    def right_face_next(self, dart: int) -> int:
        """Return the next valid dart along the face on the right of a dart."""
        twin = self.twin
        return twin[self.dual.next_at_vertex(twin[dart])]

    def right_face(self, dart: int) -> int:
        """Return the face on the right of a valid dart."""
        return self.dual.vertex(self.twin[dart])

    def neighbor(self, dart: int) -> int:
        """Return the target vertex of a valid dart."""
        return self.vertex(self.twin[dart])


class PlantriError(Exception):
    """Failure while invoking plantri_sqs or decoding its fixed records."""


@dataclass(frozen=True, slots=True)
class PlantriEnumeration:
    """Completed immutable map batch; reader resources exist only during its run."""

    graphs: tuple[PlaneDual, ...]
    time_to_first_record_s: float
    remaining_s: float

    @property
    def total_s(self) -> float:
        """Return total enumeration time."""
        return self.time_to_first_record_s + self.remaining_s

    @classmethod
    def enumerate(
        cls,
        dual_vertex_count: int,
        *,
        max_count: int,
        primal_minimum_degree: PrimalMinimumDegree = PrimalMinimumDegree.AT_LEAST_2,
        timeout: float | None = None,
    ) -> PlantriEnumeration:
        """Collect a source-order prefix; timeout bounds reading and producer completion."""
        started_at = time.perf_counter()
        if (
            type(dual_vertex_count) is not int or not MIN_DUAL_VERTEX_COUNT <= dual_vertex_count <= MAX_DUAL_VERTEX_COUNT
            or type(max_count) is not int or max_count < 0
            or (timeout is not None and not 0 <= timeout <= TIMEOUT_MAX)
        ):
            raise ValueError(f"invalid enumeration request: n={dual_vertex_count!r}, max_count={max_count!r}, timeout={timeout!r}")
        if not max_count or dual_vertex_count < primal_minimum_degree._minimum_nonempty_dual_vertex_count:
            return cls((), 0.0, time.perf_counter() - started_at)

        with cls._resolved_executable() as executable, tempfile.TemporaryFile() as stderr, ExitStack() as cleanup:
            # No worker starts until submit(); register ownership before then.
            pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pyplantri-stdout")
            command = [str(executable), *primal_minimum_degree._plantri_switches, str(dual_vertex_count + 2)]
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=stderr)
            cleanup.push(partial(cls._finish, process, pool, cast(BinaryIO, stderr)))
            deadline = None if timeout is None else time.monotonic() + timeout
            future = pool.submit(
                cls._read_stdout, cast(BufferedReader, process.stdout),
                dual_vertex_count, max_count, started_at,
            )
            graphs, first_record_s = future.result(timeout=cls._remaining_timeout(deadline))
            remaining = cls._remaining_timeout(deadline)
            if len(graphs) < max_count:
                process.wait(timeout=remaining)

        return cls(graphs, first_record_s, time.perf_counter() - started_at - first_record_s)

    @staticmethod
    def _remaining_timeout(deadline: float | None) -> float | None:
        if deadline is None:
            return None
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("plantri_sqs: deadline expired")
        return remaining

    @staticmethod
    def _read_stdout(
        stream: BufferedReader[Any],
        dual_vertex_count: int,
        max_count: int,
        started_at: float,
    ) -> tuple[tuple[PlaneDual, ...], float]:
        """Read a prefix from blocking buffered stdout; a short read means EOF."""
        twin_size = 4 * dual_vertex_count
        record_size = 5 * dual_vertex_count + 2
        graphs: list[PlaneDual] = []
        first_record_s = 0.0
        for graph_id in range(max_count):
            record = stream.read(record_size)
            if not record:
                break
            if len(record) != record_size:
                raise PlantriError(f"plantri_sqs: truncated fixed record {graph_id} ({len(record)}/{record_size} bytes)")
            graphs.append(PlaneDual._from_filter(record[:twin_size], graph_id, record[twin_size:]))
            if graph_id == 0:
                first_record_s = time.perf_counter() - started_at
        return tuple(graphs), first_record_s

    @staticmethod
    def _finish(
        process: subprocess.Popen[bytes],
        pool: ThreadPoolExecutor,
        stderr: BinaryIO,
        _exc_type: type[BaseException] | None,
        body_error: BaseException | None,
        _traceback: TracebackType | None,
    ) -> bool:
        """Reap the producer, join its reader and close stdout, retaining the primary error."""
        try:
            with ExitStack() as cleanup:
                cleanup.callback(cast(BinaryIO, process.stdout).close)
                cleanup.callback(pool.shutdown, wait=True, cancel_futures=True)
                cleanup.callback(process.wait, timeout=5.0)
                return_code = process.poll()
                if return_code is None:
                    process.kill()
            if return_code not in (None, 0):
                stderr.seek(0)
                # Collapse whitespace and keep at most 4000 characters of stderr.
                detail = " ".join(stderr.read(4001).decode("utf-8", errors="replace").split())
                if len(detail) > 4000:
                    detail = detail[:3997] + "..."
                raise PlantriError(f"plantri_sqs: execution failed (exit {return_code}); {detail or 'no output'}")
        except BaseException as cleanup_error:
            if body_error is None:
                raise
            body_error.add_note(f"plantri_sqs: cleanup also failed: {cleanup_error}")
        return False

    @staticmethod
    @contextmanager
    def _resolved_executable() -> Generator[Path, None, None]:
        exe_name = "plantri_sqs.exe" if os.name == "nt" else "plantri_sqs"
        resource = files("pyplantri").joinpath("bin", exe_name)
        with as_file(resource) as candidate:
            executable = candidate.resolve()
            if os.name == "posix" and not os.access(executable, os.X_OK):
                executable.chmod(executable.stat().st_mode | stat.S_IXUSR)
            yield executable
