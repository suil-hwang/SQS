# src/sqs/primal.py
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from operator import index

import numpy as np

from geometry.arc import has_crossing_arcs
from geometry.vector import normalize_vector

from .evaluation import GeometryClass, SQSQuality


@dataclass(frozen=True, slots=True, eq=False)
class PrimalVertex:
    """A primal vertex realized by the dual face with the same index."""

    position: np.ndarray

    def __post_init__(self) -> None:
        position = np.array(normalize_vector(self.position, name="PrimalVertex position"), copy=True)
        position.setflags(write=False)
        object.__setattr__(self, "position", position)


@dataclass(frozen=True, slots=True)
class PrimalEdge:
    """One primal edge stored by normalized `(min, max)` endpoints."""

    min_vertex_index: int
    max_vertex_index: int
    arc_length: float

    def __post_init__(self) -> None:
        for name in ("min_vertex_index", "max_vertex_index"):
            object.__setattr__(self, name, index(getattr(self, name)))
        object.__setattr__(self, "arc_length", float(self.arc_length))
        if not (0 <= self.min_vertex_index < self.max_vertex_index and 0.0 <= self.arc_length <= np.pi):
            raise ValueError("Invalid PrimalEdge")

    @property
    def endpoint_indices(self) -> tuple[int, int]:
        """Return the normalized endpoint pair."""
        return self.min_vertex_index, self.max_vertex_index


@dataclass(frozen=True, slots=True)
class PrimalFace:
    """An ordered quadrilateral boundary in the primal graph Q."""

    vertex_indices: tuple[int, int, int, int]

    def __post_init__(self) -> None:
        vertex_indices = tuple(map(index, self.vertex_indices))
        if len(vertex_indices) != 4 or len(set(vertex_indices)) != 4 or min(vertex_indices) < 0:
            raise ValueError("Invalid PrimalFace cycle")
        object.__setattr__(self, "vertex_indices", vertex_indices)


@dataclass(frozen=True, slots=True)
class PrimalGraphData:
    """An immutable primal graph snapshot; embedding validity is checked separately."""

    vertices: Sequence[PrimalVertex]
    edges: Sequence[PrimalEdge]
    faces: Sequence[PrimalFace]

    def __post_init__(self) -> None:
        for name in ("vertices", "edges", "faces"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if any(edge.max_vertex_index >= self.num_vertices for edge in self.edges) or any(
            max(face.vertex_indices) >= self.num_vertices for face in self.faces
        ):
            raise ValueError("Primal graph references an out-of-range vertex")

    @property
    def num_vertices(self) -> int:
        """Return the number of primal vertices."""
        return len(self.vertices)

    @property
    def num_edges(self) -> int:
        """Return the number of primal edges."""
        return len(self.edges)

    @property
    def num_faces(self) -> int:
        """Return the number of primal faces."""
        return len(self.faces)

    @property
    def vertex_positions(self) -> np.ndarray:
        """Return an independent (N, 3) array in vertex order."""
        return np.asarray([vertex.position for vertex in self.vertices], dtype=np.float64).reshape(-1, 3)

    def is_crossing_free(self) -> bool:
        """Whether endpoint-disjoint minor arcs have no proper crossing."""
        return not has_crossing_arcs(self.vertex_positions, [edge.endpoint_indices for edge in self.edges])

    def compute_quality_metrics(self) -> SQSQuality | None:
        """Aggregate face quality, without certifying the graph's validity."""
        if not self.faces:
            return None
        positions = self.vertex_positions
        quality = SQSQuality.from_face_vertices(
            (positions[list(face.vertex_indices)] for face in self.faces),
            total_primal_edge_arc_length=sum((edge.arc_length for edge in self.edges), start=0.0),
        )
        return None if quality.geometry_class is GeometryClass.INVALID else quality
