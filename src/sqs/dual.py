# src/sqs/dual.py
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from operator import index

import numpy as np

from geometry.arc import has_crossing_arcs, minor_arc_midpoint
from geometry.constants import EPS
from geometry.spherical_polygon import PointLocation, point_in_spherical_polygon
from geometry.vector import normalize_vector, unit_vector_angle

from .primal import PrimalVertex


@dataclass(frozen=True, slots=True, eq=False)
class DualVertex:
    """A Q* vertex with an owned, read-only unit position."""

    position: np.ndarray

    def __post_init__(self) -> None:
        position = np.array(normalize_vector(self.position, name="DualVertex position"), copy=True)
        position.setflags(write=False)
        object.__setattr__(self, "position", position)


@dataclass(frozen=True, slots=True)
class DualEdge:
    """A Q* support-edge class with a local ID and native Python scalar fields."""
    min_vertex_index: int
    max_vertex_index: int
    multiplicity: int
    arc_length: float
    edge_id: int

    def __post_init__(self) -> None:
        """Normalize typed numeric inputs and enforce local edge bounds."""
        for name in ("min_vertex_index", "max_vertex_index", "multiplicity", "edge_id"):
            object.__setattr__(self, name, index(getattr(self, name)))
        if not (
            0 <= self.min_vertex_index < self.max_vertex_index
            and self.multiplicity in (1, 2)
            and self.edge_id >= 0
            and 0.0 <= self.arc_length <= np.pi
        ):
            raise ValueError("Invalid DualEdge")
        object.__setattr__(self, "arc_length", float(self.arc_length))

    @property
    def endpoint_indices(self) -> tuple[int, int]:
        """Return the normalized endpoint pair."""
        return self.min_vertex_index, self.max_vertex_index


@dataclass(frozen=True, slots=True)
class DualFace:
    """A directed boundary cycle; two vertices represent a digon."""
    vertex_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        """Freeze distinct integer indices without changing their cyclic order."""
        vertex_indices = tuple(map(index, self.vertex_indices))
        if (
            len(vertex_indices) < 2
            or min(vertex_indices) < 0
            or len(set(vertex_indices)) != len(vertex_indices)
        ):
            raise ValueError("Invalid DualFace cycle")
        object.__setattr__(self, "vertex_indices", vertex_indices)

    @property
    def is_digon(self) -> bool:
        """Whether this boundary is a two-edge face."""
        return len(self.vertex_indices) == 2


@dataclass(frozen=True, slots=True)
class DualGraphData:
    """A realized dual graph Q*."""

    vertices: Sequence[DualVertex]
    edges: Sequence[DualEdge]
    faces: Sequence[DualFace]

    def __post_init__(self) -> None:
        """Freeze supplied order and bound all references by the stored vertices."""
        object.__setattr__(self, "vertices", tuple(self.vertices))
        object.__setattr__(self, "edges", tuple(self.edges))
        object.__setattr__(self, "faces", tuple(self.faces))
        if any(edge.max_vertex_index >= self.num_directions for edge in self.edges) or any(
            max(face.vertex_indices) >= self.num_directions for face in self.faces
        ):
            raise ValueError("Dual graph references an out-of-range vertex")

    @property
    def vertex_positions(self) -> np.ndarray:
        """Return a read-only (N, 3) array in dual-vertex order."""
        positions = np.asarray([vertex.position for vertex in self.vertices], dtype=np.float64).reshape(-1, 3)
        positions.setflags(write=False)
        return positions

    @property
    def num_directions(self) -> int:
        """Return the number of dual vertices and assigned directions."""
        return len(self.vertices)

    @property
    def num_edges_with_multiplicity(self) -> int:
        """Return E(Q*) after expanding support-edge multiplicities."""
        return sum(edge.multiplicity for edge in self.edges)

    @property
    def num_single_support_edges(self) -> int:
        """Return the number of multiplicity-one support-edge classes."""
        return sum(edge.multiplicity == 1 for edge in self.edges)

    @property
    def num_double_support_edges(self) -> int:
        """Return the number of multiplicity-two support-edge classes."""
        return sum(edge.multiplicity == 2 for edge in self.edges)

    @property
    def num_faces(self) -> int:
        """Return the number of dual faces."""
        return len(self.faces)

    def is_all_valence_four(self) -> bool:
        """Test degree four using support-edge multiplicities."""
        return all(
            sum(edge.multiplicity for edge in self.edges if vertex in edge.endpoint_indices) == 4
            for vertex in range(self.num_directions)
        )

    def is_crossing_free(self) -> bool:
        """Test proper crossings of endpoint-disjoint minor arcs."""
        return not has_crossing_arcs(self.vertex_positions, [edge.endpoint_indices for edge in self.edges])

    def classify_primal_vertices(
        self,
        primal_vertices: Sequence[PrimalVertex],
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Return invalid and admissible great-circle face indices, each in face order."""
        if not self.faces or len(primal_vertices) != len(self.faces):
            return tuple(range(len(self.faces))), ()

        degenerate = set(self.great_circle_face_indices(self.vertex_positions, self.faces))
        admissible = [self.is_face_admissible(i, vertex.position) for i, vertex in enumerate(primal_vertices)]
        return (
            tuple(i for i, ok in enumerate(admissible) if not ok),
            tuple(i for i, ok in enumerate(admissible) if ok and i in degenerate),
        )

    @staticmethod
    def great_circle_face_indices(
        unit_directions: np.ndarray,
        faces: Sequence[DualFace],
    ) -> tuple[int, ...]:
        """Return coplanar regular-face indices from finite unit (N, 3) rows."""
        directions = np.asarray(unit_directions)
        degenerate = []
        for face_index, face in enumerate(faces):
            if face.is_digon:
                continue
            points = directions[list(face.vertex_indices)]
            singular_values = np.linalg.svd(points, compute_uv=False)
            tolerance = max(EPS, float(np.finfo(np.float64).eps * max(points.shape) * singular_values[0]))
            if singular_values[-1] <= tolerance:
                degenerate.append(face_index)
        return tuple(degenerate)

    def is_face_admissible(
        self,
        face_index: int,
        candidate: np.ndarray,
    ) -> bool:
        """Require a digon's midpoint within EPS, or a regular face's strict right interior."""
        face = self.faces[face_index]
        points = np.asarray([self.vertices[index].position for index in face.vertex_indices])
        candidate = normalize_vector(candidate)
        if face.is_digon:
            midpoint = minor_arc_midpoint(points[0], points[1])
            return bool(unit_vector_angle(candidate, midpoint) <= EPS)

        return point_in_spherical_polygon(
            points,
            candidate,
            interior_side="right",
        ) is PointLocation.INTERIOR
