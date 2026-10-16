# src/sqs/validation.py
from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, fields
from itertools import pairwise

import networkx as nx
import numpy as np

from geometry.arc import has_crossing_arcs, minor_arc_contains
from geometry.spherical_polygon import PointLocation, point_in_spherical_polygon, self_intersects
from geometry.vector import normalize_vectors, unit_vector_angle

from .dual import DualEdge
from .primal import PrimalEdge
from .structure import SQS


@dataclass(frozen=True, slots=True)
class SQSValidationReport:
    """Boolean outcomes of the listed structural and geometric checks."""

    is_dual_embedding_valid: bool
    is_primal_embedding_valid: bool
    has_expected_primal_counts: bool
    is_primal_connected: bool
    is_primal_dual_incidence_consistent: bool
    has_primal_face_direction_containment: bool

    @property
    def is_valid(self) -> bool:
        """Return whether every check explicitly passed."""
        return all(getattr(self, field.name) is True for field in fields(self))


def is_crossing_free(vertices: np.ndarray, edges: Sequence[DualEdge | PrimalEdge]) -> bool:
    """Test proper crossings of endpoint-disjoint minor arcs, without embedding validation."""
    return not has_crossing_arcs(vertices, [edge.endpoint_indices for edge in edges])


def is_dual_embedding_valid(vertices: np.ndarray, edges: Sequence[DualEdge | PrimalEdge]) -> bool:
    """Check minor-arc regularity, nonincident contacts, and crossings of unexpanded dual (or primal) edges."""
    points = np.asarray(vertices, dtype=np.float64)
    # Multiplicity is topological; each stored dual edge defines one geometric arc.
    endpoints = np.asarray([edge.endpoint_indices for edge in edges], dtype=np.intp).reshape(-1, 2)
    starts, ends = points[endpoints.T]
    # A unique, nondegenerate minor arc requires an angular length in (0, pi).
    lengths = unit_vector_angle(starts, ends)
    if not np.all((lengths > 0.0) & (lengths < np.pi)):
        return False

    # A nonincident vertex may not lie on an arc, even at a coincident endpoint.
    indices = np.arange(len(points))
    nonincident = (indices != endpoints[:, 0, None]) & (indices != endpoints[:, 1, None])
    if np.any(nonincident & minor_arc_contains(starts[:, None], ends[:, None], points[None], inclusive=True)):
        return False

    return is_crossing_free(points, edges)


def is_primal_embedding_valid(sqs: SQS) -> bool:
    """Check primal minor-arc regularity, nonincident contacts, and crossings."""
    # Q's edges obey the same geodesic-embedding predicate as Q*'s support edges.
    primal = sqs.primal
    return is_dual_embedding_valid(primal.vertex_positions, primal.edges)


def has_expected_primal_counts(sqs: SQS) -> bool:
    """Test V=n+2 and E=2n; SQS already enforces F=n."""
    n, primal = sqs.dual.num_directions, sqs.primal
    return n >= 3 and (primal.num_vertices, primal.num_edges) == (n + 2, 2 * n)


def is_primal_connected(sqs: SQS) -> bool:
    """Test connectivity of all primal vertices; reject the empty graph."""
    primal = sqs.primal
    graph = nx.Graph((edge.min_vertex_index, edge.max_vertex_index) for edge in primal.edges)
    graph.add_nodes_from(range(primal.num_vertices))
    return primal.num_vertices > 0 and nx.is_connected(graph)


def is_primal_dual_incidence_consistent(sqs: SQS) -> bool:
    """Check oriented combinatorial duality of locally valid face cycles."""
    primal, dual = sqs.primal, sqs.dual
    right_face = {
        arc: face_index
        for face_index, face in enumerate(primal.faces)
        for arc in pairwise((*face.vertex_indices, face.vertex_indices[0]))
    }
    # Require four distinct darts per quad and two per stored edge.
    if len(right_face) != 4 * primal.num_faces or len(right_face) != 2 * primal.num_edges:
        return False

    multiplicities: Counter[tuple[int, int]] = Counter()
    dual_boundaries: list[set[tuple[int, int]]] = [set() for _ in dual.faces]
    for edge in primal.edges:
        u, v = edge.min_vertex_index, edge.max_vertex_index
        # Consume both darts exactly once, rejecting missing or duplicate edges.
        right = right_face.pop((u, v), None)
        left = right_face.pop((v, u), None)
        if right is None or left is None:
            return False
        # The dual arc right -> left has the face for primal vertex v on its right.
        dual_boundaries[v].add((right, left))
        dual_boundaries[u].add((left, right))
        multiplicities[min(right, left), max(right, left)] += 1

    return multiplicities == {edge.endpoint_indices: edge.multiplicity for edge in dual.edges} and dual_boundaries == [
        set(pairwise((*face.vertex_indices, face.vertex_indices[0]))) for face in dual.faces
    ]


def has_primal_face_direction_containment(sqs: SQS, *, validate_boundary: bool = True) -> bool:
    """Test whether direction i lies strictly inside primal face i and strictly outside every other face."""
    positions = sqs.primal.vertex_positions
    directions = sqs.dual.vertex_positions
    for index, face in enumerate(sqs.primal.faces):
        rows = positions[list(face.vertex_indices)]
        if validate_boundary and self_intersects(normalize_vectors(rows)) is not False:
            return False
        # Exclusion from every other face certifies a single sphere cover.
        expected = np.full(len(directions), PointLocation.EXTERIOR)
        expected[index] = PointLocation.INTERIOR
        locations = point_in_spherical_polygon(rows, directions, interior_side="right", validate_assumptions=False)
        if not np.array_equal(locations, expected):
            return False
    return True


def validate(sqs: SQS) -> SQSValidationReport:
    """Run every listed check independently without computing quality metrics."""
    return SQSValidationReport(
        is_dual_embedding_valid=is_dual_embedding_valid(sqs.dual.vertex_positions, sqs.dual.edges),
        is_primal_embedding_valid=is_primal_embedding_valid(sqs),
        has_expected_primal_counts=has_expected_primal_counts(sqs),
        is_primal_connected=is_primal_connected(sqs),
        is_primal_dual_incidence_consistent=is_primal_dual_incidence_consistent(sqs),
        has_primal_face_direction_containment=has_primal_face_direction_containment(sqs),
    )
