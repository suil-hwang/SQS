# src/sqs/structure.py
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from operator import itemgetter

import numpy as np
from igl import quad_edges
from pyplantri.plantri import PlaneDual

from geometry.arc import minor_arc_midpoint
from geometry.constants import EPS
from geometry.spherical_polygon import SphericalPolygonOrientation, centroid
from geometry.vector import angular_distance, normalize_vectors, unit_vector_angle

from .dual import DualEdge, DualFace, DualGraphData, DualVertex
from .primal import PrimalEdge, PrimalFace, PrimalGraphData, PrimalVertex


@dataclass(frozen=True, slots=True)
class SQS:
    """Realized dual Q* and primal Q."""

    dual: DualGraphData
    primal: PrimalGraphData

    def __post_init__(self) -> None:
        """Enforce primal-dual count correspondence V↔F*, E↔E*, F↔V*."""
        if not isinstance(self.dual, DualGraphData):
            raise TypeError("dual must be DualGraphData")
        if not isinstance(self.primal, PrimalGraphData):
            raise TypeError("primal must be PrimalGraphData")
        correspondences = (
            ("F(Q)", self.primal.num_faces, "V(Q*)", self.dual.num_directions),
            ("V(Q)", self.primal.num_vertices, "F(Q*)", self.dual.num_faces),
            ("E(Q)", self.primal.num_edges, "E(Q*)", self.dual.num_edges_with_multiplicity),
        )
        for primal_name, primal_count, dual_name, dual_count in correspondences:
            if primal_count != dual_count:
                raise ValueError(f"primal-dual count mismatch: {primal_name}={primal_count} != {dual_name}={dual_count}")

    @classmethod
    def construct(
        cls,
        direction_vectors: np.ndarray,
        plane_map: PlaneDual,
        assignment: Mapping[int, int] | Sequence[int],
        *,
        reverse_rotation_system: bool = False,
    ) -> SQS:
        """Realize G* as Q* under the direction assignment, then construct Q."""
        # Each graph vertex requires one distinct unit direction.
        size = plane_map.num_vertices
        directions = normalize_vectors(direction_vectors, name="direction_vectors")
        arc_lengths = unit_vector_angle(directions[:, None, :], directions[None, :, :])
        coincident = np.argwhere(np.triu(arc_lengths <= EPS, k=1))
        if coincident.size:
            first, second = map(int, coincident[0])
            raise ValueError(f"coincident directions: {first}, {second}")
        if len(directions) != size:
            raise ValueError(f"plane_map: {size} vertices != {len(directions)} directions")

        # Read the bijection in graph-vertex order, independent of mapping order.
        mapping = dict(assignment) if isinstance(assignment, Mapping) else dict(enumerate(assignment))
        expected = set(range(size))
        if set(mapping) != expected or set(mapping.values()) != expected:
            raise ValueError(f"assignment must be a bijection on [0, {size})")
        indices = tuple(mapping[vertex] for vertex in range(size))

        # Face cycles and support stars must share the same global orientation.
        orientation_step = -1 if reverse_rotation_system else 1
        mapped_edges: list[tuple[int, int, int]] = []
        for (u, v), multiplicity in plane_map.edge_multiplicity.items():
            first, second = sorted((indices[u], indices[v]))
            # Use the midpoint kernel's regularity check near the antipodal limit.
            if arc_lengths[first, second] >= np.pi - EPS:
                minor_arc_midpoint(directions[first], directions[second])
            mapped_edges.append((first, second, multiplicity))

        # Sort by unoriented cycle, then orientation; retain each stored cycle's start.
        face_cycles = []
        for face in plane_map.faces:
            cycle = tuple(indices[vertex] for vertex in face[::orientation_step])
            anchor = cycle.index(min(cycle))
            forward = cycle[anchor:] + cycle[:anchor]
            backward = forward[:1] + forward[:0:-1]
            sort_key = min(forward, backward), forward
            face_cycles.append((sort_key, cycle))
        face_cycles.sort(key=itemgetter(0))

        # Index stars by assigned direction; reverse only after removing duplicate neighbors.
        rotations: list[tuple[int, ...]] = [()] * len(indices)
        for graph_vertex, rotation in enumerate(plane_map.embedding):
            neighbors = dict.fromkeys(rotation)
            mapped_rotation = tuple(indices[neighbor] for neighbor in neighbors)
            rotations[indices[graph_vertex]] = mapped_rotation[::orientation_step]

        # Assign edge IDs in lexicographic endpoint order.
        dual = DualGraphData(
            vertices=tuple(DualVertex(direction) for direction in directions),
            edges=tuple(
                DualEdge(u, v, multiplicity, arc_lengths[u, v], edge_id)
                for edge_id, (u, v, multiplicity) in enumerate(sorted(mapped_edges))
            ),
            faces=tuple(DualFace(cycle) for _, cycle in face_cycles),
        )

        # Plane duality over the exterior-clockwise support rotations of Q*.
        # Dual face f_i* gives v_i at its projected centroid, or minor-arc midpoint for a digon.
        vertices: list[PrimalVertex] = []
        for face_index, face in enumerate(dual.faces):
            points = np.asarray([dual.vertices[index].position for index in face.vertex_indices])
            candidate = (
                minor_arc_midpoint(points[0], points[1])
                if face.is_digon
                else centroid(points, orientation=SphericalPolygonOrientation.CLOCKWISE)
            )
            # Accept regular-face centroids only in the strict right interior.
            if not dual.is_face_admissible(face_index, candidate):
                raise ValueError("Dual face is inadmissible")
            vertices.append(PrimalVertex(candidate))
        positions = np.asarray([vertex.position for vertex in vertices])

        # Index regular faces by directed boundary arcs and digons by unoriented supports.
        digon_face_by_edge = {
            (min(face.vertex_indices), max(face.vertex_indices)): face_index
            for face_index, face in enumerate(dual.faces)
            if face.is_digon
        }
        right_face_by_arc = {
            arc: face_index
            for face_index, face in enumerate(dual.faces)
            if not face.is_digon
            for arc in pairwise((*face.vertex_indices, face.vertex_indices[0]))
        }
        faces: list[PrimalFace] = []
        # Form f_j clockwise around v_j*, inserting each digon after the incoming arc's right face.
        for dual_vertex, support_neighbors in enumerate(rotations):
            face_vertices: list[int] = []
            for neighbor in support_neighbors:
                edge_key = min(dual_vertex, neighbor), max(dual_vertex, neighbor)
                face_vertices.append(right_face_by_arc[neighbor, dual_vertex])
                if edge_key in digon_face_by_edge:
                    face_vertices.append(digon_face_by_edge[edge_key])
            a, b, c, d = face_vertices  # Q* is 4-regular, so each dual star gives four quad corners.
            faces.append(PrimalFace((a, b, c, d)))

        edges: tuple[PrimalEdge, ...] = ()
        if faces:
            # Extract unique quad edges, retaining both edges of each chain through a digon vertex.
            quads = np.asarray([face.vertex_indices for face in faces], dtype=np.int64)
            endpoints = np.sort(quad_edges(quads), axis=1)
            endpoints = endpoints[np.lexsort((endpoints[:, 1], endpoints[:, 0]))]
            # Measure minor-arc lengths from Q's stored vertices; embedding checks are separate.
            lengths = np.asarray(angular_distance(positions[endpoints[:, 0]], positions[endpoints[:, 1]]))
            edges = tuple(
                PrimalEdge(int(first), int(second), float(length))
                for (first, second), length in zip(endpoints, lengths, strict=True)
            )

        return cls(dual=dual, primal=PrimalGraphData(vertices=vertices, edges=edges, faces=faces))
