# src/geometry/vector.py
from __future__ import annotations

from typing import Literal, overload

import numpy as np
from numpy.typing import ArrayLike

from .constants import FloatArray


def spherical_to_cartesian(colatitude: ArrayLike, longitude: ArrayLike, radius: ArrayLike = 1.0) -> FloatArray:
    """Convert colatitude, longitude, and radius to Cartesian vectors."""
    colatitude, longitude, radius = np.broadcast_arrays(
        np.asarray(colatitude, dtype=np.float64),
        np.asarray(longitude, dtype=np.float64),
        np.asarray(radius, dtype=np.float64),
    )
    xy_radius = radius * np.sin(colatitude)
    return np.stack((xy_radius * np.cos(longitude), xy_radius * np.sin(longitude), radius * np.cos(colatitude)), axis=-1)


def cartesian_to_spherical(cartesian_vectors: ArrayLike) -> FloatArray:
    """Return radius, colatitude, and signed longitude."""
    vectors = np.asarray(cartesian_vectors, dtype=np.float64)
    # Rescale only at float64 extremes to preserve direction and avoid overflow.
    float_info = np.finfo(vectors.dtype)
    max_abs = np.max(np.abs(vectors), axis=-1)
    needs_rescaling = np.isfinite(max_abs) & (max_abs > 0.0) & (
        (max_abs < float_info.tiny) | (max_abs > float_info.max / np.sqrt(3.0))
    )
    _, exponents = np.frexp(max_abs)
    exponents = np.where(needs_rescaling, exponents - 1, 0)
    with np.errstate(over="ignore", under="ignore"):
        x, y, z = np.moveaxis(np.ldexp(vectors, -exponents[..., None]), -1, 0)
        scaled_xy_radii = np.hypot(x, y)
        radii = np.ldexp(np.hypot(scaled_xy_radii, z), exponents)
        colatitudes = np.where(max_abs == 0.0, 0.0, np.arctan2(scaled_xy_radii, z))
        # Longitude is undefined at the poles; use +0.0.
        longitudes = np.where(np.all(vectors[..., :2] == 0.0, axis=-1), 0.0, np.arctan2(vectors[..., 1], vectors[..., 0]))
    return np.stack((radii, colatitudes, longitudes), axis=-1)


@overload
def normalize_vectors(vectors: ArrayLike, *, name: str = ..., min_norm: float = ..., return_norms: Literal[False] = ...) -> FloatArray: ...
@overload
def normalize_vectors(vectors: ArrayLike, *, name: str = ..., min_norm: float = ..., return_norms: Literal[True]) -> tuple[FloatArray, FloatArray]: ...
def normalize_vectors(
    vectors: ArrayLike, *, name: str = "vectors", min_norm: float = 0.0, return_norms: bool = False,
) -> FloatArray | tuple[FloatArray, FloatArray]:
    """Normalize ``(N, 3)`` rows with norm strictly above ``min_norm``."""
    if not 0.0 <= min_norm < np.inf:
        raise ValueError("min_norm must be a finite non-negative float.")
    rows = np.asarray(vectors, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3), got {rows.shape}.")

    # Each finite nonzero scaled row has squared norm in [0.25, 3].
    _, exponents = np.frexp(np.max(np.abs(rows), axis=1, keepdims=True))
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        scaled = np.ldexp(rows, -exponents)
        scaled_norms = np.linalg.vector_norm(scaled, axis=1, keepdims=True)
        norms = np.ldexp(scaled_norms, exponents)[:, 0]

    if not np.isfinite(rows).all() or not np.all(norms > min_norm):
        raise ValueError(f"{name}: Non-finite or norm <= {min_norm:.1e}.")

    use_original = np.isfinite(norms) & (norms >= np.finfo(rows.dtype).tiny)
    # Original division preserves representable small components on ordinary rows.
    with np.errstate(under="ignore"):
        unit_rows = scaled / scaled_norms
        np.divide(rows, norms[:, None], out=unit_rows, where=use_original[:, None])
    return (unit_rows, norms) if return_norms else unit_rows


@overload
def normalize_vector(vector: ArrayLike, *, name: str = ..., min_norm: float = ..., return_norm: Literal[False] = ...) -> FloatArray: ...
@overload
def normalize_vector(vector: ArrayLike, *, name: str = ..., min_norm: float = ..., return_norm: Literal[True]) -> tuple[FloatArray, float]: ...
def normalize_vector(
    vector: ArrayLike, *, name: str = "vector", min_norm: float = 0.0, return_norm: bool = False,
) -> FloatArray | tuple[FloatArray, float]:
    """Normalize one ``(3,)`` vector with norm strictly above ``min_norm``."""
    units, norms = normalize_vectors(np.asarray(vector, dtype=np.float64)[None, ...], name=name, min_norm=min_norm, return_norms=True)
    return (units[0], float(norms[0])) if return_norm else units[0]


def broadcast_vector_rows(vector: ArrayLike, *other_vectors: ArrayLike) -> tuple[list[FloatArray], bool]:
    """Broadcast vector inputs to read-only rows and report whether every input has shape ``(3,)``."""
    arrays = [np.asarray(value, dtype=np.float64) for value in (vector, *other_vectors)]
    if any(array.ndim not in (1, 2) or array.shape[-1] != 3 for array in arrays):
        raise ValueError("vectors must have shape (3,) or (N, 3).")

    shape = np.broadcast_shapes((1, 3), *(array.shape for array in arrays))
    return [np.broadcast_to(array, shape) for array in arrays], all(array.ndim == 1 for array in arrays)


def unit_vector_angle(first: FloatArray, second: FloatArray) -> FloatArray:
    """Return angles in radians between broadcast unit vectors."""
    # Hypot avoids losing very small separations to squared-norm underflow.
    difference_norm = np.hypot.reduce(first - second, axis=-1)
    sum_norm = np.hypot.reduce(first + second, axis=-1)
    return np.asarray(2.0 * np.arctan2(difference_norm, sum_norm), dtype=np.float64)


def angular_distance(first: ArrayLike, second: ArrayLike) -> float | FloatArray:
    """Return radians between corresponding finite nonzero directions."""
    a, b = (np.asarray(value, dtype=np.float64) for value in (first, second))
    angles = unit_vector_angle(normalize_vectors(np.atleast_2d(a), name="first"), normalize_vectors(np.atleast_2d(b), name="second"))
    return float(angles[0]) if a.ndim == b.ndim == 1 else angles


def orthogonal_unit_vector(vector: ArrayLike, preferred_direction: ArrayLike | None = None) -> FloatArray:
    """Return a unit vector orthogonal to ``vector``, favoring a stable projection."""
    normal = normalize_vector(vector, name="vector")

    if preferred_direction is not None:
        preferred = normalize_vector(preferred_direction, name="preferred_direction")
        dot = float(np.dot(normal, preferred))
        if abs(dot) < 0.9:
            tangent = preferred - dot * normal
            return tangent / np.linalg.vector_norm(tangent)

    axis = np.zeros(3, dtype=np.float64)
    axis[np.argmin(np.abs(normal))] = 1.0
    tangent = np.cross(axis, normal)
    return tangent / np.linalg.vector_norm(tangent)
