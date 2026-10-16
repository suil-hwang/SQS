# src/geometry/spherical_polygon.py
from __future__ import annotations

from enum import IntEnum
from itertools import product
from math import fsum
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from .arc import arcs_cross, is_point_on_minor_arc, minor_arc_contains, minor_arc_midpoint, origin_orientation
from .constants import EPS, TWO_PI, FloatArray, IntArray
from .vector import broadcast_vector_rows, normalize_vector, normalize_vectors, orthogonal_unit_vector


class SphericalPolygonOrientation(IntEnum):
    """Exterior-view traversal convention for the selected region."""

    COUNTERCLOCKWISE = 1  # Positive orientation; region lies to the left.
    CLOCKWISE = -1  # SQS face convention; region lies to the right.


class PointLocation(IntEnum):
    """Point classification; INVALID denotes an unresolved location."""

    INTERIOR = 1
    BOUNDARY = 0
    EXTERIOR = -1
    INVALID = -2


def boundary_moment_vector(points: FloatArray, edge_tolerance: float = EPS) -> FloatArray | None:
    """Return the oriented boundary moment vector, or None when unresolved.

    For a simple polygon with a directed minor-arc boundary on S^2:

        M = sum_i hat{n}_i * theta_i = 2 * integral_R x dA

    hat{n}_i = normalize(v_i x v_{i+1}) and theta_i is the edge angle.
    Viewed from outside, R lies left of the directed polygon boundary.
    Reversing the vertex order negates M.
    """
    unit_points = normalize_vectors(points, name="points")
    if len(unit_points) < 3 or not 0.0 <= edge_tolerance < np.inf:
        return None

    next_points = np.roll(unit_points, -1, axis=0)
    poles = np.cross(unit_points - next_points, unit_points + next_points)
    pole_norms = np.linalg.vector_norm(poles, axis=1)
    angles = np.arctan2(0.5 * pole_norms, np.vecdot(unit_points, next_points))

    if np.any((angles <= edge_tolerance) | ((np.pi - angles) <= edge_tolerance)):
        return None

    terms = poles * (angles / pole_norms)[:, None]
    return np.array([fsum(component) for component in terms.T], dtype=np.float64)


def centroid(
    points: FloatArray, edge_tolerance: float = EPS, *,
    orientation: SphericalPolygonOrientation = SphericalPolygonOrientation.CLOCKWISE, min_moment_norm: float = EPS,
) -> FloatArray:
    """Return the normalized surface-area centroid of the selected region."""
    moment = boundary_moment_vector(points, edge_tolerance=edge_tolerance)
    if moment is None:
        raise ValueError("spherical polygon boundary moment is undefined; regular minor arcs are required")
    return normalize_vector(orientation.value * moment, name="boundary moment", min_norm=min_moment_norm)


def interior_angle(unit_vertices: FloatArray, *, edge_tolerance: float = EPS, return_turns: bool = False) -> FloatArray:
    """Return right-region interior angles in [0, 2π), or signed left-positive turns."""
    following = np.roll(unit_vertices, -1, axis=0)
    poles = np.cross(unit_vertices - following, unit_vertices + following)
    # Rescale pole magnitudes before products without unitizing them.
    _, exponents = np.frexp(np.max(np.abs(poles), axis=1, keepdims=True))
    poles = np.ldexp(poles, -exponents)
    pole_norms = np.ldexp(np.linalg.vector_norm(poles, axis=1), exponents[:, 0])
    previous = np.roll(poles, 1, axis=0)
    sine = np.vecdot(unit_vertices, np.cross(previous, poles))
    cosine = np.vecdot(previous, poles)
    regular = pole_norms > 2.0 * np.sin(edge_tolerance)

    if return_turns:
        angles = np.arctan2(sine, cosine)
    else:
        # Compute tiny interiors directly, without adding pi to a rounded turn.
        angles = np.mod(np.arctan2(-sine, -cosine), TWO_PI)
        angles[angles >= TWO_PI] = 0.0

    return np.where(regular & np.roll(regular, 1), angles, np.nan if return_turns else 0.0)


def spherical_polygon_orientation(
    vertices: FloatArray, *, edge_tolerance: float = EPS, turn_tolerance: float = EPS,
) -> SphericalPolygonOrientation | None:
    """Return the smaller-region orientation, or None when unresolved."""
    unit_vertices = normalize_vectors(vertices, name="vertices")
    if len(unit_vertices) < 3 or not (0.0 <= edge_tolerance < 0.5 * np.pi and 0.0 <= turn_tolerance < np.inf):
        return None

    turns = interior_angle(unit_vertices, edge_tolerance=edge_tolerance, return_turns=True)
    total_turn = fsum(turns)
    if abs(total_turn) > turn_tolerance and np.all(np.pi - np.abs(turns) > EPS):
        return SphericalPolygonOrientation(1 if total_turn > 0.0 else -1)
    return None


def is_geodesically_convex(vertices: ArrayLike, *, tolerance: float = EPS) -> bool:
    """Test strict geodesic convexity with a sine-clearance tolerance."""
    if not 0.0 <= tolerance < 1.0:
        return False
    vertices = normalize_vectors(vertices, name="vertices", min_norm=EPS)
    n = len(vertices)
    if n < 3:
        return False
    ends = np.roll(vertices, -1, axis=0)
    edge = np.repeat(np.arange(n), n - 2)
    vertex = (edge + np.tile(np.arange(2, n), n)) % n
    sides = origin_orientation(vertices[edge], ends[edge], vertices[vertex])
    if not (np.all(sides > 0) or np.all(sides < 0)):
        return False
    if tolerance > 0.0:
        # Exact signs determine sides; floating distances impose the clearance.
        poles = np.cross(vertices - ends, vertices + ends)
        norms = np.hypot.reduce(poles, axis=1)
        if np.any(norms <= 2.0 * tolerance):
            return False
        normals = poles / norms[:, None]
        if np.any(np.abs(np.vecdot(normals[edge], vertices[vertex])) <= tolerance):
            return False
    return True


def self_intersects(points: FloatArray) -> bool | None:
    """Detect crossings, nonincident contacts, or overlaps of a minor arc boundary."""
    points = np.asarray_chkfinite(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3).")
    if len(points) > 1 and np.linalg.norm(points[0] - points[-1]) <= EPS:
        points = points[:-1]
    n = len(points)
    if n < 3:
        return None

    ends = np.roll(points, -1, axis=0)
    i, j = np.triu_indices(n, k=1)
    if np.any(np.linalg.norm(points[i] - points[j], axis=1) <= EPS) or np.any(
        np.linalg.norm(points + ends, axis=1) <= 2.0 * np.sin(0.5 * EPS)
    ):
        return None

    edge = np.repeat(np.arange(n), n - 2)
    vertex = (edge + np.tile(np.arange(2, n), n)) % n
    # Only coplanar nonincident vertices can contact a finite minor arc.
    coplanar = origin_orientation(points[edge], ends[edge], points[vertex]) == 0
    if np.any(coplanar) and np.any(minor_arc_contains(
        points[edge[coplanar]], ends[edge[coplanar]], points[vertex[coplanar]], inclusive=True,
    )):
        return True

    nonadjacent = (j > i + 1) & (j - i < n - 1)
    i, j = i[nonadjacent], j[nonadjacent]
    return bool(np.any(arcs_cross(points[i], ends[i], points[j], ends[j])))


def point_in_spherical_polygon(
    vertices: ArrayLike, point: ArrayLike, *, interior_point: ArrayLike | None = None,
    interior_side: Literal["left", "right"] | None = None, validate_assumptions: bool = True,
) -> PointLocation | IntArray:
    """Classify the directed right region by default; unresolved queries return INVALID."""
    if interior_side is not None and interior_point is not None:
        raise ValueError("interior_side cannot be combined with interior_point.")
    region_sign = {None: -1, "left": 1, "right": -1}[interior_side]
    vertices = normalize_vectors(vertices, name="vertices")
    if len(vertices) > 1 and np.linalg.norm(vertices[0] - vertices[-1]) <= EPS:
        vertices = vertices[:-1]
    if len(vertices) < 3 or (validate_assumptions and self_intersects(vertices) is not False):
        raise ValueError("A simple polygon requires at least 3 distinct vertices.")
    ends = np.roll(vertices, -1, axis=0)
    # Oriented edge poles point to the left of each minor arc (Eq. 6).
    normals = normalize_vectors(np.cross(vertices - ends, vertices + ends), name="polygon edges", min_norm=2.0 * np.sin(EPS))
    reference = None if interior_point is None else normalize_vector(interior_point, name="interior_point")
    [queries], scalar = broadcast_vector_rows(point)
    locations = np.full(len(queries), PointLocation.INVALID, dtype=np.int8)
    active_rows = np.flatnonzero(np.isfinite(queries).all(axis=1) & np.any(queries != 0.0, axis=1))
    queries = normalize_vectors(queries[active_rows])

    # Bound edge-query working arrays, including the reference-independent boundary test.
    batch_size = max(1, 512 // len(vertices))
    boundary = np.zeros(len(queries), dtype=bool)
    for first in range(0, len(queries), batch_size):
        batch = queries[first:first + batch_size]
        boundary[first:first + batch_size] = np.any(
            is_point_on_minor_arc(vertices[:, None, :], ends[:, None, :], batch[None, :, :], tolerance=EPS), axis=0,
        )
    locations[active_rows[boundary]] = PointLocation.BOUNDARY
    active_rows, queries = active_rows[~boundary], queries[~boundary]
    if not active_rows.size:
        return PointLocation(int(locations[0])) if scalar else locations

    convex = reference is None and is_geodesically_convex(vertices)
    orientation = hemisphere_sign = 0
    hemisphere_axes = np.zeros((2, 3))
    if convex:
        orientation = int(origin_orientation(vertices[:1], ends[:1], vertices[2:3])[0])
    elif reference is None:
        # The stored vertex sum is only a candidate; certify h.v > 0 with exact signs.
        hemisphere = vertices.sum(axis=0)
        k = int(np.argmax(np.abs(hemisphere)))
        i, j = (k + 1) % 3, (k + 2) % 3
        hemisphere_axes[0, i] = hemisphere_axes[1, j] = hemisphere[k]
        hemisphere_axes[:, k] = -hemisphere[[i, j]]
        hemisphere_sign = int(np.sign(hemisphere[k]))
        # U x V = h[k] h exactly: copying components avoids rounded cross products.
        if np.all(hemisphere_sign * origin_orientation(
            vertices, np.broadcast_to(hemisphere_axes[0], vertices.shape), np.broadcast_to(hemisphere_axes[1], vertices.shape),
        ) > 0):
            # Exact lexicographic extreme in the gnomonic chart fixes the bounded side.
            pivot = 0
            for index in range(1, len(vertices)):
                order = origin_orientation(
                    np.broadcast_to(vertices[index], (2, 3)), np.broadcast_to(vertices[pivot], (2, 3)), hemisphere_axes,
                )
                if order[0] < 0 or (order[0] == 0 and order[1] < 0):
                    pivot = index
            orientation = int(origin_orientation(*vertices[np.arange(pivot - 1, pivot + 2) % len(vertices)][:, None, :])[0])

    if orientation:
        for first in range(0, len(queries), batch_size):
            batch = queries[first:first + batch_size]
            starts = np.repeat(vertices, len(batch), axis=0)
            repeated_queries = np.tile(batch, (len(vertices), 1))
            sides = origin_orientation(starts, np.repeat(ends, len(batch), axis=0), repeated_queries).reshape(len(vertices), len(batch))
            if convex:
                inside = np.all(sides == orientation, axis=0)
            else:
                # Query-axis winding uses unperturbed 3D signs, without rounded projected coordinates.
                axes = np.eye(3)[np.argmin(np.abs(batch), axis=1)]
                y = origin_orientation(starts, repeated_queries, np.tile(axes, (len(vertices), 1))).reshape(sides.shape)
                next_y = np.roll(y, -1, axis=0)
                upward = (y <= 0) & (next_y > 0) & (sides > 0)
                downward = (y > 0) & (next_y <= 0) & (sides < 0)
                # The opposite closed hemisphere contains no point of the bounded region.
                halfspaces = hemisphere_sign * origin_orientation(
                    batch, np.broadcast_to(hemisphere_axes[0], batch.shape), np.broadcast_to(hemisphere_axes[1], batch.shape),
                )
                inside = (halfspaces > 0) & (upward.sum(axis=0) - downward.sum(axis=0) == orientation)
            inside ^= region_sign != orientation
            locations[active_rows[first:first + batch_size]] = np.where(inside, PointLocation.INTERIOR, PointLocation.EXTERIOR)
        return PointLocation(int(locations[0])) if scalar else locations

    if reference is None:
        # Find the paper's interior p by moving off edge 0 to the selected side without crossing the boundary.
        boundary_points = (minor_arc_midpoint(vertices[0], ends[0]), normalize_vector(3.0 * vertices[0] + ends[0]))
        insets = []
        inset = max(64.0 * EPS, float(np.sqrt(np.finfo(np.float64).eps)))
        while inset > 8.0 * EPS:
            insets.append(inset)
            inset *= 0.5
        candidates = (
            (boundary_point, normalize_vector(np.cos(inset) * boundary_point + region_sign * np.sin(inset) * normals[0]))
            for boundary_point, inset in product(boundary_points, insets)
        )
    else:
        candidates = ((None, reference),)
    for boundary_point, reference in candidates:
        if boundary_point is not None and (
            np.any(arcs_cross(boundary_point, reference, vertices[1:], ends[1:]))
            or np.any(minor_arc_contains(boundary_point, reference, vertices, inclusive=True))
        ):
            continue
        # Keep the boundary away from both poles of the query frame.
        if not np.any(is_point_on_minor_arc(vertices[:, None, :], ends[:, None, :], np.stack((reference, -reference))[None, :, :], tolerance=EPS)):
            break
    else:
        return PointLocation(int(locations[0])) if scalar else locations

    # An antipodal query has no unique minor arc from p (Sec. 3.1).
    nonantipodal = np.linalg.norm(reference + queries, axis=1) > 2.0 * np.sin(0.5 * EPS)
    active_rows, queries = active_rows[nonantipodal], queries[nonantipodal]

    # Ketzner et al. (2022), Sec. 3.2.1: z = p; tangent axes define a prime meridian.
    x_axis = orthogonal_unit_vector(reference)
    basis = np.column_stack((x_axis, np.cross(reference, x_axis)))
    vertex_xy = vertices @ basis
    start_longitude = np.mod(np.arctan2(vertex_xy[:, 1], vertex_xy[:, 0]), TWO_PI)[:, None]
    end_longitude = np.roll(start_longitude, -1, axis=0)
    wraps_longitude = np.abs(end_longitude - start_longitude) >= np.pi
    reference_sides = origin_orientation(vertices, ends, np.broadcast_to(reference, vertices.shape))

    for first in range(0, len(queries), batch_size):
        batch = queries[first:first + batch_size]
        query_xy = batch @ basis
        query_longitude = np.mod(np.arctan2(query_xy[:, 1], query_xy[:, 0]), TWO_PI)[None, :]
        # Necessary strike (Eqs. 7-8): half-open longitude intervals resolve vertex ties.
        edge_index, query_index = np.nonzero((start_longitude < query_longitude) ^ (end_longitude < query_longitude) ^ wraps_longitude)
        # Hemisphere check (Eq. 10): compare exact determinant signs; longitude tests remain floating.
        query_sides = origin_orientation(vertices[edge_index], ends[edge_index], batch[query_index])
        # Even crossings keep q in p's region; repeated query indices must accumulate.
        inside = np.ones(len(batch), dtype=bool)
        np.logical_xor.at(inside, query_index, query_sides * reference_sides[edge_index] < 0)
        locations[active_rows[first:first + batch_size]] = np.where(inside, PointLocation.INTERIOR, PointLocation.EXTERIOR)
    return PointLocation(int(locations[0])) if scalar else locations
