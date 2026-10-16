# src/geometry/arc.py
from __future__ import annotations

from fractions import Fraction

import numpy as np
from igl.predicates import orient3d
from numpy.typing import ArrayLike, NDArray

from .constants import EPS, FloatArray
from .vector import broadcast_vector_rows, normalize_vector, normalize_vectors, unit_vector_angle


def origin_orientation(a: FloatArray, b: FloatArray, c: FloatArray) -> NDArray[np.int32]:
    """Return the signs of det(a, b, c) for finite (N, 3) vector rows."""
    triples = np.stack((a, b, c), axis=1)
    _, exponents = np.frexp(triples)
    _, axis_exponents = np.frexp(np.max(np.abs(triples), axis=1, keepdims=True))
    native = np.all((triples == 0.0) | (exponents - axis_exponents >= -126), axis=(1, 2))
    scaled = np.ldexp(triples[native], -axis_exponents[native])
    signs = np.zeros(len(triples), dtype=np.int32)
    if len(scaled):
        first, second, third = (np.ascontiguousarray(scaled[:, k, :]) for k in range(3))
        signs[native] = np.asarray(orient3d(first, second, third, np.zeros((len(scaled), 3))), dtype=np.int32)
    for row in np.flatnonzero(~native):
        x, y, z = [[Fraction(float(value)) for value in vector] for vector in triples[row]]
        determinant = x[0] * (y[1] * z[2] - y[2] * z[1]) - x[1] * (y[0] * z[2] - y[2] * z[0]) + x[2] * (y[0] * z[1] - y[1] * z[0])
        signs[row] = (determinant > 0) - (determinant < 0)
    return signs


def arcs_cross(a: FloatArray, b: FloatArray, c: FloatArray, d: FloatArray) -> NDArray[np.bool_]:
    """Return flags for proper crossings of open minor arcs on the unit sphere."""
    arrays = [np.asarray_chkfinite(value, dtype=np.float64) for value in (a, b, c, d)]
    if any(array.ndim == 0 or array.shape[-1] != 3 for array in arrays):
        raise ValueError("Arc vectors must have trailing dimension 3.")
    arrays = np.broadcast_arrays(*arrays)
    shape = arrays[0].shape[:-1]
    a, b, c, d = (array.reshape(-1, 3) for array in arrays)
    cyclic_orientations = origin_orientation(
        np.concatenate((a, c, b, d)), np.concatenate((c, b, d, a)), np.concatenate((b, d, a, c))
    ).reshape(4, len(a))
    crosses = np.all(cyclic_orientations > 0, axis=0) | np.all(cyclic_orientations < 0, axis=0)
    return crosses.reshape(shape)


def has_crossing_arcs(points: FloatArray, endpoints: ArrayLike) -> bool:
    """Return whether two arcs that share no endpoint index properly cross."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    endpoints = np.asarray(endpoints, dtype=np.intp).reshape(-1, 2)
    first, second = np.triu_indices(len(endpoints), k=1)
    disjoint = np.all(endpoints[first, :, None] != endpoints[second, None, :], axis=(1, 2))
    first, second = first[disjoint], second[disjoint]
    return bool(np.any(arcs_cross(
        points[endpoints[first, 0]], points[endpoints[first, 1]],
        points[endpoints[second, 0]], points[endpoints[second, 1]],
    )))


def interpolate_minor_arc(start: ArrayLike, end: ArrayLike, parameters: ArrayLike) -> FloatArray:
    """Interpolate the minor arc"""
    t = np.asarray(parameters, dtype=np.float64)
    a = normalize_vector(start, name="start")
    b = normalize_vector(end, name="end")
    if np.linalg.norm(a + b) <= 2.0 * np.sin(0.5 * EPS):
        raise ValueError("Minor arc interpolation is undefined near antipodal endpoints.")

    pole = np.cross(a - b, a + b)
    if np.any(pole):
        tangent = normalize_vector(np.cross(pole, a), name="arc tangent")
        phase = t * np.arctan2(0.5 * np.hypot.reduce(pole), np.vecdot(a, b))
        path = np.cos(phase)[..., None] * a + np.sin(phase)[..., None] * tangent
    else:
        path = np.broadcast_to(a, t.shape + (3,)).copy()
    path[t == 0.0] = a
    path[t == 1.0] = b
    return path


def minor_arc_midpoint(arc_start: FloatArray, arc_end: FloatArray) -> FloatArray:
    """Return minor arc midpoints."""
    [starts, ends], scalar = broadcast_vector_rows(arc_start, arc_end)
    a = normalize_vectors(starts, name="arc_start")
    b = normalize_vectors(ends, name="arc_end")
    midpoints = normalize_vectors(a + b, name="arc midpoint", min_norm=2.0 * np.sin(0.5 * EPS))
    obtuse = np.vecdot(a, b) < 0.0
    difference = a[obtuse] - b[obtuse]
    midpoints[obtuse] = normalize_vectors(np.cross(np.cross(difference, midpoints[obtuse]), difference))
    return midpoints[0] if scalar else midpoints


def minor_arc_contains(start: FloatArray, end: FloatArray, point: FloatArray, *, inclusive: bool) -> NDArray[np.bool_]:
    """Test membership in regular unit minor arcs with exact orientation signs."""
    arrays = [np.asarray_chkfinite(value, dtype=np.float64) for value in (start, end, point)]
    if any(array.ndim == 0 or array.shape[-1] != 3 for array in arrays):
        raise ValueError("Arc vectors must have trailing dimension 3.")
    arrays = np.broadcast_arrays(*arrays)
    shape = arrays[0].shape[:-1]
    a, b, p = (array.reshape(-1, 3) for array in arrays)
    pole_signs = origin_orientation(
        np.repeat(a, 3, axis=0), np.repeat(b, 3, axis=0), np.tile(np.eye(3), (len(a), 1))
    ).reshape(-1, 3)
    axis_index = np.argmax(np.abs(pole_signs), axis=1)
    axes, direction = np.eye(3)[axis_index], pole_signs[np.arange(len(a)), axis_index]
    on_circle, from_start, from_end = origin_orientation(
        np.concatenate((a, a, p)), np.concatenate((b, p, b)), np.concatenate((p, axes, axes))
    ).reshape(3, len(a))
    compare = np.greater_equal if inclusive else np.greater
    between = compare(from_start * direction, 0) & compare(from_end * direction, 0)
    return ((direction != 0) & (on_circle == 0) & between).reshape(shape)


def is_point_on_minor_arc(
    arc_start: ArrayLike, arc_end: ArrayLike, point: ArrayLike, *, tolerance: float = EPS,
) -> bool | NDArray[np.bool_]:
    """Test whether a point lies on a closed minor arc within tolerance."""
    if not 0.0 <= tolerance < np.inf:
        raise ValueError("tolerance must be a finite non-negative float.")

    units = []
    for name, value in (("arc_start", arc_start), ("arc_end", arc_end), ("point", point)):
        array = np.asarray(value, dtype=np.float64)
        units.append(normalize_vectors(array.reshape((-1,) + array.shape[-1:]), name=name).reshape(array.shape))
    a, b, p = units

    summed = a + b
    bisector = normalize_vectors(summed.reshape(-1, 3), name="arc bisectors", min_norm=2.0 * np.sin(0.5 * EPS)).reshape(summed.shape)
    pole = np.cross(a - b, bisector)
    scale = np.max(np.abs(pole), axis=-1, keepdims=True)
    regular = scale[..., 0] > 0.0
    pole = np.divide(pole, scale, out=np.zeros_like(pole), where=scale > 0.0)

    pole_cross_point = np.cross(pole, p)
    foot_inside = regular & (np.vecdot(a - p, pole_cross_point) <= 0.0) & (np.vecdot(b - p, pole_cross_point) >= 0.0)

    circle_distance = np.arctan2(np.abs(np.vecdot(p, pole)), np.linalg.norm(pole_cross_point, axis=-1))
    endpoint_distance = np.minimum(unit_vector_angle(p, a), unit_vector_angle(p, b))

    result = (endpoint_distance <= tolerance) | (foot_inside & (circle_distance <= tolerance))
    return bool(result) if result.ndim == 0 else result
