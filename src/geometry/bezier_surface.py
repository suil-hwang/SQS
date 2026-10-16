# src/geometry/bezier_surface.py
from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations
from math import comb
from typing import NamedTuple

import numpy as np
from geomdl import NURBS

from .constants import EPS, FloatArray, Vec3
from .vector import normalize_vectors
from .spherical_polygon import is_geodesically_convex
from .bezier_curve import arc_to_homogeneous_control_points


_DEGREE = 4
_CONTROL_COUNT = _DEGREE + 1
_KNOT_VECTOR = [0.0] * _CONTROL_COUNT + [1.0] * _CONTROL_COUNT


class FacesToSurfacesResult(NamedTuple):
    """Face-ordered patches with aligned unit directions and parameter corners."""

    surfaces: list[NURBS.Surface]
    sweep_directions: list[Vec3]
    patch_corners_per_surface: list[list[Vec3]]


def build_spherical_biquartic_quad_patch(p00: Vec3, p10: Vec3, p01: Vec3, p11: Vec3) -> NURBS.Surface:
    """Build a rational biquartic patch bounded by unit-sphere minor arcs."""
    corners = tuple(normalize_vectors((p00, p10, p01, p11), name="corners"))
    ring = np.asarray(corners, dtype=np.float64)[[0, 1, 3, 2]]

    # Within an open hemisphere, at most three vertices support the minimum cap.
    i, j = np.triu_indices(4, k=1)
    sums, differences = ring[i] + ring[j], ring[i] - ring[j]
    pair_centers = np.where(
        (np.vecdot(ring[i], ring[j]) < 0.0)[:, None],
        np.cross(np.cross(differences, sums), differences),
        sums,
    )
    i, j, k = np.asarray(list(combinations(range(4), 3))).T
    normals = np.cross(ring[i] - ring[j], ring[i] - ring[k])
    candidates = np.vstack((ring, pair_centers, normals, -normals))
    norms = np.hypot.reduce(candidates, axis=1)
    nonzero = norms > 0.0
    candidates = candidates[nonzero] / norms[nonzero, None]
    margins = np.min(candidates @ ring.T, axis=1)
    best = np.argmax(margins)

    # Require strict geodesic convexity and more than EPS clearance from the chart equator.
    if not (is_geodesically_convex(ring) and margins[best] > np.sin(EPS)):
        raise ValueError("Quad must be geodesically convex with a stable open-hemisphere chart.")

    pole = -candidates[best]
    v00, v10, v01, v11 = corners
    lifted_net = np.zeros((3, 3, 4), dtype=np.float64)
    lifted_net[:, 0] = arc_to_homogeneous_control_points(v00, v10, 2)
    lifted_net[:, 2] = arc_to_homogeneous_control_points(v01, v11, 2)
    lifted_net[0, :] = arc_to_homogeneous_control_points(v00, v01, 2)
    lifted_net[2, :] = arc_to_homogeneous_control_points(v10, v11, 2)
    # Linear homogeneous projection commutes with averaging the eight boundary controls.
    lifted_net[1, 1] = lifted_net.sum(axis=(0, 1)) / 8.0

    # Dedoncker et al. (2018), Eq. (2): homogeneous tangent-plane coordinates at -pole.
    axial = lifted_net[..., :3] @ pole
    chart_net = np.empty_like(lifted_net)
    chart_net[..., :3] = 2.0 * (lifted_net[..., :3] - axial[..., None] * pole)
    chart_net[..., 3] = lifted_net[..., 3] - axial
    # Farin (5th ed.), Eq. (6.27): B_i^2 B_k^2 = coefficients[i, k, i+k] B_{i+k}^4.
    coefficients = np.zeros((3, 3, 5), dtype=np.float64)
    for i, k in np.ndindex(3, 3):
        coefficients[i, k, i + k] = comb(2, i) * comb(2, k) / comb(4, i + k)
    products = []
    for left, right in ((chart_net, chart_net), (chart_net[..., :3], chart_net[..., 3])):
        u_product = np.einsum("ij...,kl...,ikm->mjl...", left, right, coefficients)
        products.append(np.einsum("mjl...,jln->mn...", u_product, coefficients))
    squared, weighted = products
    norm_squared, weight_squared = squared[..., :3].sum(axis=-1), squared[..., 3]

    # Dedoncker et al. (2018), Eq. (5): inverse stereographic projection in homogeneous coordinates.
    sphere_net = np.empty((_CONTROL_COUNT, _CONTROL_COUNT, 4), dtype=np.float64)
    sphere_net[..., :3] = 4.0 * weighted + (norm_squared - 4.0 * weight_squared)[..., None] * pole
    sphere_net[..., 3] = norm_squared + 4.0 * weight_squared

    surface = NURBS.Surface()
    surface.degree_u = surface.degree_v = _DEGREE
    surface.set_ctrlpts(sphere_net.reshape(-1, 4).tolist(), _CONTROL_COUNT, _CONTROL_COUNT)
    surface.knotvector_u, surface.knotvector_v = _KNOT_VECTOR.copy(), _KNOT_VECTOR.copy()
    return surface


def primal_faces_to_bezier_surfaces(
    primal_faces: Sequence[Sequence[int]],
    primal_vertex_positions: Sequence[Vec3] | FloatArray,
    primal_face_direction_indices: Sequence[int],
    direction_vectors: Sequence[Vec3] | FloatArray,
) -> FacesToSurfacesResult:
    """Build rational biquartic patches from exterior-clockwise primal quads."""
    raw_vertices, directions = (np.asarray(value, dtype=np.float64) for value in (primal_vertex_positions, direction_vectors))
    vertices = normalize_vectors(raw_vertices.reshape(0, 3) if raw_vertices.shape == (0,) else raw_vertices, name="primal_vertex_positions")
    directions = normalize_vectors(directions.reshape(0, 3) if directions.shape == (0,) else directions, name="direction_vectors")

    # Dedoncker et al. (2018), Eqs. (10)-(12): compose each convex quad's biquadratic chart into a biquartic sphere patch.
    surfaces, patch_corners, sweep_directions = [], [], []
    limits = (len(vertices),) * 4 + (len(directions),)
    for (a, b, c, d), direction_index in zip(primal_faces, primal_face_direction_indices, strict=True):
        if not all(
            isinstance(index, (int, np.integer)) and not isinstance(index, bool) and 0 <= index < limit
            for index, limit in zip((a, b, c, d, direction_index), limits)
        ):
            raise ValueError("Face and direction indices must be in-range integers.")
        # Map the exterior clockwise ring [a, b, c, d] to (p00, p10, p01, p11).
        ring = np.asarray((a, d, b, c), dtype=np.intp)
        p00, p10, p01, p11 = vertices[ring]
        patch_corners.append([p00, p10, p01, p11])
        surfaces.append(build_spherical_biquartic_quad_patch(*raw_vertices[ring]))
        sweep_directions.append(directions[direction_index].copy())

    return FacesToSurfacesResult(surfaces, sweep_directions, patch_corners)
