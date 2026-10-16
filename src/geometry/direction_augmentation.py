# src/geometry/direction_augmentation.py
from __future__ import annotations

from collections.abc import Callable
from typing import cast

import numpy as np
from scipy.optimize import NonlinearConstraint, minimize

from .constants import EPS
from .spherical_sampling import spherical_fibonacci_direction_set
from .vector import normalize_vectors

def find_unresolved_direction_pair(unit_directions: np.ndarray) -> tuple[int, int] | None:
    """Return the first pair with chord distance at or below EPS."""
    unit_directions = np.asarray_chkfinite(np.asarray(unit_directions).astype(np.float64, casting="same_kind", copy=False))
    if unit_directions.ndim != 2 or unit_directions.shape[1] != 3:
        raise ValueError("unit_directions must have shape (N, 3).")

    for first_direction_index, first_direction in enumerate(unit_directions[:-1]):
        separation_vectors = unit_directions[first_direction_index + 1 :] - first_direction
        unresolved_offsets = np.flatnonzero(np.vecdot(separation_vectors, separation_vectors) <= EPS**2)
        if unresolved_offsets.size:
            return first_direction_index, first_direction_index + 1 + int(unresolved_offsets[0])

    return None


def coulomb_energy_and_gradient(
    free_directions: np.ndarray,
    fixed_directions: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Return conditional Coulomb energy and its ambient free-direction gradient."""
    free_directions = np.asarray_chkfinite(np.asarray(free_directions).astype(np.float64, casting="same_kind", copy=False))
    fixed_directions = np.asarray_chkfinite(np.asarray(fixed_directions).astype(np.float64, casting="same_kind", copy=False))

    gradient_shape = free_directions.shape
    if free_directions.ndim == 1:
        free_directions = free_directions.reshape(-1, 3)
    if free_directions.shape[1:] != (3,) or fixed_directions.shape[1:] != (3,):
        raise ValueError("free_directions and fixed_directions must have shape (N, 3).")

    # Compute Cartesian distances to free then fixed rows without assuming unit length.
    free_count = len(free_directions)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        delta = free_directions[:, None, :] - np.vstack((free_directions, fixed_directions))[None, :, :]
        squared = np.vecdot(delta, delta)

        # Only the free-free block has self-interactions, along this diagonal.
        np.fill_diagonal(squared, np.inf)
        if np.any(squared <= EPS**2):
            raise FloatingPointError("unresolved Coulomb interaction")

        inverse_distances = 1.0 / np.sqrt(squared)
        # Halve only free-free energy; each gradient row already differentiates its own free variable.
        energy = 0.5 * inverse_distances[:, :free_count].sum() + inverse_distances[:, free_count:].sum()
        gradient = -np.einsum("ijk,ij->ik", delta, inverse_distances / squared)

    return float(energy), gradient.reshape(gradient_shape)


def augment_directions(
    fixed_directions: np.ndarray | None,
    auxiliary_direction_count: int,
    max_optimizer_iterations: int = 200,
    optimizer_tolerance: float = 1.0e-8,
    *,
    seed_pool_scale: int = 12,
    minimum_seed_pool_size: int = 200,
) -> np.ndarray:
    """Append the best valid projected candidate without moving normalized inputs."""
    fixed_rows = np.asarray([] if fixed_directions is None else fixed_directions, dtype=np.float64)
    if fixed_rows.shape in ((0,), (3,)):
        fixed_rows = fixed_rows.reshape(-1, 3)
    fixed_rows = normalize_vectors(fixed_rows, name="fixed_directions")

    # The conditional objective omits fixed-fixed pairs, so check their separation here.
    if find_unresolved_direction_pair(fixed_rows) is not None:
        raise FloatingPointError("unresolved Coulomb interaction")
    if auxiliary_direction_count <= 0:
        return np.ascontiguousarray(fixed_rows)

    # Greedy angular maximin seeds with incremental nearest-neighbor cosines.
    pool_size = max(
        auxiliary_direction_count,
        minimum_seed_pool_size,
        seed_pool_scale * (len(fixed_rows) + auxiliary_direction_count),
    )
    pool = np.asarray(spherical_fibonacci_direction_set(pool_size), dtype=np.float64)
    nearest_cosines = np.max(pool @ fixed_rows.T, axis=1, initial=-np.inf)
    initial_rows = np.empty((auxiliary_direction_count, 3), dtype=np.float64)

    for direction in initial_rows:
        selected = np.argmin(nearest_cosines)
        direction[:] = pool[selected]
        np.maximum(nearest_cosines, pool @ direction, out=nearest_cosines)
        nearest_cosines[selected] = np.inf

    initial_rows = normalize_vectors(initial_rows, name="initial_auxiliary_directions")

    best_rows = initial_rows.copy()
    best_energy, _ = coulomb_energy_and_gradient(initial_rows, fixed_rows)

    # Store SLSQP iterate copies for later selection without a stateful callback.
    candidates: list[np.ndarray] = []
    result = minimize(
        coulomb_energy_and_gradient,
        x0=initial_rows.reshape(-1), args=(fixed_rows,),
        method="SLSQP", jac=True,
        constraints=NonlinearConstraint(_sphere_residuals, 0.0, 0.0, jac=_sphere_jacobian),
        callback=cast(Callable[..., None], candidates.append),
        options={"maxiter": max(1, max_optimizer_iterations), "ftol": optimizer_tolerance},
    )
    candidates.append(result.x)

    # Evaluate projected unit directions; the objective checks free-free and free-fixed separation.
    for x in candidates:
        rows = normalize_vectors(x.reshape(-1, 3), name="optimized_free_directions")
        energy, _ = coulomb_energy_and_gradient(rows, fixed_rows)
        if energy < best_energy:
            best_rows, best_energy = rows, energy

    return np.vstack((fixed_rows, best_rows))


def _sphere_residuals(flat_directions: np.ndarray) -> np.ndarray:
    """Return the squared-norm residual for each direction."""
    directions = flat_directions.reshape(-1, 3)
    return np.sum(directions * directions, axis=1) - 1.0


def _sphere_jacobian(x: np.ndarray) -> np.ndarray:
    """Return the analytic block-diagonal sphere-constraint Jacobian."""
    rows = x.reshape(-1, 3)
    count = len(rows)
    jacobian = np.zeros((count, count, 3), dtype=np.float64)
    indices = np.arange(count)
    jacobian[indices, indices] = 2.0 * rows
    return jacobian.reshape(count, x.size)
