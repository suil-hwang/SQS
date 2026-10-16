# src/assignments/candidate_arcs.py
from __future__ import annotations

from dataclasses import dataclass
from typing import final
import numpy as np

from geometry.arc import is_point_on_minor_arc, origin_orientation
from geometry.constants import EPS
from geometry.vector import normalize_vectors, unit_vector_angle

AllowedArc = tuple[int, int, int]
CrossingConflict = tuple[int, int]


@final
@dataclass(frozen=True, slots=True, eq=False, repr=False, match_args=False)
class CandidateArcs:
    """Candidate arc geometry for one direction set."""

    unit_directions: np.ndarray
    allowed_arcs: tuple[AllowedArc, ...]
    conflicts: tuple[CrossingConflict, ...]
    isolated_arc_indices: tuple[int, ...]
    allowed_arc_lengths: tuple[float, ...]
    allowed_arc_incidence: tuple[tuple[int, ...], ...]
    clockwise_ranks: np.ndarray

    def __init__(self) -> None:
        """Require validated construction via from_directions()."""
        raise TypeError("Use CandidateArcs.from_directions()")

    @property
    def num_candidate_pairs(self) -> int:
        """Return the number of unordered direction pairs."""
        num_directions = len(self.unit_directions)
        return num_directions * (num_directions - 1) // 2

    @classmethod
    def from_directions(
        cls,
        directions: np.ndarray,
    ) -> CandidateArcs:
        """Validate directions and precompute candidate arc geometry."""
        direction_rows = np.asarray(directions, dtype=np.float64)
        if direction_rows.ndim != 2 or direction_rows.shape[1] != 3:
            raise ValueError(f"directions shape {direction_rows.shape}, expected (n, 3)")
        num_directions = len(direction_rows)
        if num_directions < 3:
            raise ValueError("SQS requires at least three directions")
        # normalize_vectors returns a fresh array, so this is an owned snapshot of the caller's rows.
        unit_directions = np.ascontiguousarray(normalize_vectors(direction_rows, name="direction_vectors"))
        arc_length_matrix = unit_vector_angle(unit_directions[:, None], unit_directions[None])

        # Global candidate-pair indices follow lexicographic unordered direction pairs.
        endpoint_array = np.column_stack(np.triu_indices(num_directions, k=1))
        pair_lengths = arc_length_matrix[tuple(endpoint_array.T)]
        if (coincident := np.flatnonzero(pair_lengths <= EPS)).size:
            first, second = endpoint_array[coincident[0]].tolist()
            raise ValueError(f"coincident directions: {first}, {second}")

        # Conditions (ii)(a)-(b): a pair-locally non-antipodal minor arc with no third direction on the closed arc.
        regular_arc_indices = np.flatnonzero(pair_lengths < np.pi - EPS)
        regular_endpoints = endpoint_array[regular_arc_indices]
        starts, ends = unit_directions[regular_endpoints.T]
        on_arc = np.asarray(is_point_on_minor_arc(starts[:, None], ends[:, None], unit_directions[None], tolerance=EPS))
        on_arc[np.arange(len(regular_endpoints))[:, None], regular_endpoints] = False
        allowed_candidate_pair_indices = regular_arc_indices[~on_arc.any(axis=1)]
        # A third direction on an arc also excludes distinct overlaps on one great circle.
        allowed_endpoints = endpoint_array[allowed_candidate_pair_indices]

        # Condition (ii)(c): endpoint-disjoint proper crossings by arcs_cross's four exact signs, from one triple table.
        triples = np.indices((num_directions,) * 3).reshape(3, -1)
        orientation = origin_orientation(*unit_directions[triples]).reshape((num_directions,) * 3)
        pair_first, pair_second = np.triu_indices(len(allowed_endpoints), k=1)
        (a, b), (c, d) = allowed_endpoints[pair_first].T, allowed_endpoints[pair_second].T
        disjoint = (a != c) & (a != d) & (b != c) & (b != d)
        pair_first, pair_second, a, b, c, d = (indices[disjoint] for indices in (pair_first, pair_second, a, b, c, d))
        reference = orientation[a, c, b]
        crossing = (
            (reference != 0)
            & (reference == orientation[c, b, d])
            & (reference == orientation[b, d, a])
            & (reference == orientation[d, a, c])
        )
        conflicting = allowed_candidate_pair_indices[np.stack((pair_first[crossing], pair_second[crossing]), axis=1)]

        # Per-direction incidence in allowed-arc order.
        allowed_arc_incidence = tuple(
            tuple(allowed_candidate_pair_indices[np.any(allowed_endpoints == direction, axis=1)].tolist())
            for direction in range(num_directions)
        )

        # Condition (iii): exact clockwise ranks from the first allowed neighbor; det(u, p, q) < 0 puts q after p.
        adjacent = np.zeros((num_directions,) * 2, dtype=np.bool_)
        adjacent[tuple(allowed_endpoints.T)] = adjacent[tuple(allowed_endpoints.T[::-1])] = True
        indices = np.arange(num_directions)
        reference = np.argmax(adjacent, axis=1)
        side = orientation[indices[:, None], reference[:, None], indices]
        turn_class = np.select((indices == reference[:, None], side < 0, side == 0), (0, 1, 2), 3)
        precedes = (turn_class[:, :, None] < turn_class[:, None, :]) | (
            (turn_class[:, :, None] == turn_class[:, None, :]) & (orientation < 0)
        )
        clockwise_ranks = np.where(adjacent, np.sum(precedes & adjacent[:, :, None], axis=1), -1).astype(np.int32)
        for array in (unit_directions, clockwise_ranks):
            array.flags.writeable = False
        data = object.__new__(cls)
        for name, value in (
            ("unit_directions", unit_directions),
            ("allowed_arcs", tuple(zip(allowed_candidate_pair_indices.tolist(), *allowed_endpoints.T.tolist(), strict=True))),
            ("conflicts", tuple(map(tuple, conflicting.tolist()))),
            ("isolated_arc_indices", tuple(np.setdiff1d(allowed_candidate_pair_indices, conflicting).tolist())),
            ("allowed_arc_lengths", tuple(pair_lengths[allowed_candidate_pair_indices].tolist())),
            ("allowed_arc_incidence", allowed_arc_incidence),
            ("clockwise_ranks", clockwise_ranks),
        ):
            object.__setattr__(data, name, value)
        return data
