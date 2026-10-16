# src/geometry/spherical_sampling.py
from __future__ import annotations

import math

import numpy as np


def spherical_fibonacci_direction_set(num_points: int) -> list[list[float]]:
    """Generate a midpoint-offset spherical Fibonacci direction set on ``S^2``."""
    if isinstance(num_points, bool) or not isinstance(num_points, int):
        raise TypeError("num_points must be an integer")
    if num_points <= 0:
        raise ValueError("num_points must be positive")

    golden_ratio = (1.0 + math.sqrt(5.0)) / 2.0
    indices = np.arange(num_points, dtype=np.float64)

    # theta_i = 2 pi {i / golden_ratio}.
    theta = 2.0 * math.pi * (np.remainder(indices, golden_ratio) / golden_ratio)

    # Midpoints of n equal-area horizontal bands, ordered from north to south.
    # The centered form preserves z_i = -z_(n-1-i) in floating point.
    z = (num_points - 1.0 - 2.0 * indices) / num_points
    radius = np.sqrt(np.maximum(0.0, (1.0 - z) * (1.0 + z)))

    points = np.stack([radius * np.cos(theta), radius * np.sin(theta), z], axis=1)
    return points.tolist()


def iid_uniform_sphere_direction_set(
    num_points: int,
    *,
    rng: np.random.Generator | None = None,
) -> list[list[float]]:
    """Generate one ordered IID sample of surface-area-uniform directions on ``S^2``."""
    if isinstance(num_points, bool) or not isinstance(num_points, int):
        raise TypeError("num_points must be an integer")
    if num_points <= 0:
        raise ValueError("num_points must be positive")
    generator = np.random.default_rng() if rng is None else rng
    z = generator.uniform(-1.0, 1.0, size=num_points)
    theta = generator.uniform(0.0, 2.0 * math.pi, size=num_points)
    radius = np.sqrt(np.maximum(0.0, (1.0 - z) * (1.0 + z)))
    points = np.column_stack((radius * np.cos(theta), radius * np.sin(theta), z))
    return points.tolist()
