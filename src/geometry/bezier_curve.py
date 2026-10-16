# src/geometry/bezier_curve.py
from __future__ import annotations

import numpy as np
from geomdl import NURBS

from .constants import EPS, FloatArray, Vec3
from .vector import normalize_vector, normalize_vectors


def arc_to_homogeneous_control_points(p0: Vec3, p1: Vec3, degree: int) -> FloatArray:
    """Return homogeneous Bézier controls for a unit-sphere minor arc."""
    if degree not in (2, 3, 4):
        raise ValueError(f"Unsupported degree: {degree}. Use 2, 3, or 4.")

    unit_tolerance = 5.0 * np.finfo(np.float64).eps
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        a, b = [
            point if point.shape == (3,) and abs(float(np.vecdot(point, point)) - 1.0) <= unit_tolerance
            else normalize_vector(point, name=name)
            for point, name in ((np.asarray(p0, dtype=np.float64), "p0"), (np.asarray(p1, dtype=np.float64), "p1"))
        ]

    bisector = a + b
    cos_half_angle = 0.5 * float(np.linalg.norm(bisector))
    if cos_half_angle < np.sin(0.5 * EPS):
        raise ValueError("Endpoints are antipodal or within EPS of antipodal.")

    # Both bisectors are symmetric; the cross form limits radial-error amplification near antipodes.
    if np.vecdot(a, b) < 0.0:
        difference = a - b
        bisector = np.cross(np.cross(difference, bisector), difference)
    midpoint = bisector / np.linalg.norm(bisector)
    h0, h1, h2 = np.column_stack(((a, midpoint, b), (1.0, cos_half_angle, 1.0)))
    if degree == 2:
        return np.stack((h0, h1, h2))
    # Farin Eq. (13.14): elevate the quadratic controls in homogeneous coordinates.
    if degree == 3:
        return np.stack((h0, (h0 + 2.0 * h1) / 3.0, (2.0 * h1 + h2) / 3.0, h2))
    return np.stack((h0, 0.5 * (h0 + h1), ((h0 + h2) + 4.0 * h1) / 6.0, 0.5 * (h1 + h2), h2))


def arc_to_bezier_curve(p0: Vec3, p1: Vec3, degree: int = 4) -> NURBS.Curve:
    """Construct a rational Bézier curve for a unit-sphere minor arc."""
    curve = NURBS.Curve()
    curve.degree = degree
    curve.ctrlptsw = arc_to_homogeneous_control_points(p0, p1, degree).tolist()
    curve.knotvector = [0.0] * (degree + 1) + [1.0] * (degree + 1)
    return curve


def primal_edges_to_bezier_curves(
    primal_vertex_positions: list[Vec3], primal_edges: list[tuple[int, int, float]],
) -> list[NURBS.Curve]:
    """Return input-ordered rational quartic curves for unit-sphere minor arcs."""
    if not primal_edges:
        return []

    unit_positions = normalize_vectors(primal_vertex_positions, name="primal_vertex_positions")
    return [arc_to_bezier_curve(unit_positions[start], unit_positions[end]) for start, end, _arc_length in primal_edges]
