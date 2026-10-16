# src/sqs/evaluation.py
from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import ArrayLike

from geometry.constants import EPS
from geometry.spherical_polygon import interior_angle, self_intersects
from geometry.vector import normalize_vectors, unit_vector_angle


class GeometryClass(Enum):
    """Primal-face angle classes, plus a rejection sentinel."""

    INVALID = "INVALID"
    CONCAVE = "CONCAVE"
    DEGENERATE = "DEGENERATE"
    CONVEX = "CONVEX"

    @property
    def preference_rank(self) -> int:
        """Return the geometry preference; higher is better."""
        return {
            GeometryClass.INVALID: 0,
            GeometryClass.CONCAVE: 1,
            GeometryClass.DEGENERATE: 2,
            GeometryClass.CONVEX: 3,
        }[self]


@dataclass(frozen=True, slots=True)
class SQSQuality:
    """Aggregate metrics and optional geometry classification, not an SQS audit."""

    min_angle: float
    max_angle: float
    max_edge_length_ratio: float
    total_primal_edge_arc_length: float
    geometry_class: GeometryClass | None = None

    @classmethod
    def invalid(cls, total_primal_edge_arc_length: float) -> SQSQuality:
        """Represent rejection without fabricating angle metrics."""
        return cls(
            math.nan, math.nan, math.inf,
            float(total_primal_edge_arc_length), GeometryClass.INVALID,
        )

    @classmethod
    def from_face_vertices(
        cls,
        face_vertices: Iterable[ArrayLike],
        *,
        total_primal_edge_arc_length: float,
        primal_embedding_is_valid: bool | None = None,
    ) -> SQSQuality:
        """Measure spherical quadrilaterals in their stored boundary order."""
        total = float(total_primal_edge_arc_length)
        if not math.isfinite(total) or total < 0.0:
            raise ValueError("total_primal_edge_arc_length must be finite and nonnegative")

        check_geometry = primal_embedding_is_valid is not None
        if check_geometry and not primal_embedding_is_valid:
            return cls.invalid(total)

        min_angle, max_angle, max_ratio = math.inf, -math.inf, 1.0
        for vertices in face_vertices:
            rows = np.asarray(vertices, dtype=np.float64)
            if rows.shape != (4, 3):
                raise ValueError(f"vertices must have shape (4, 3), got {rows.shape}")
            rows = normalize_vectors(rows)

            lengths = unit_vector_angle(rows, np.roll(rows, -1, axis=0))
            if not np.all((lengths > 0.0) & (lengths < math.pi)):
                return cls.invalid(total)

            if check_geometry and self_intersects(rows) is not False:
                return cls.invalid(total)

            # Preserve the right-region convention; never reverse rows.
            angles = interior_angle(rows)
            if not np.all((angles > 0.0) & (angles < math.tau)):
                return cls.invalid(total)

            min_angle = min(min_angle, float(angles.min()))
            max_angle = max(max_angle, float(angles.max()))
            shortest, longest = float(lengths.min()), float(lengths.max())
            max_ratio = max(max_ratio, math.inf if shortest <= EPS else longest / shortest)

        if min_angle == math.inf:
            return cls.invalid(total)

        geometry_class = None
        if check_geometry:
            if max_angle > math.pi + EPS:
                geometry_class = GeometryClass.CONCAVE
            elif max_angle >= math.pi - EPS:
                geometry_class = GeometryClass.DEGENERATE
            else:
                geometry_class = GeometryClass.CONVEX

        return cls(min_angle, max_angle, max_ratio, total, geometry_class)

    @property
    def is_selectable(self) -> bool:
        """Pass geometry screening, not the full SQS audit."""
        return self.geometry_class not in (None, GeometryClass.INVALID)

    @property
    def max_deviation_from_right_angle(self) -> float:
        """Return the maximum angle deviation from pi/2 in radians."""
        return max(abs(self.min_angle - math.pi / 2), abs(self.max_angle - math.pi / 2))
