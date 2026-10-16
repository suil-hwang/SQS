# src/ui/visual_config.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Final, TypeAlias

RGB: TypeAlias = tuple[float, float, float]

_VERTEX_RADIUS: Final = 0.025
_SQS_CURVE_RADIUS: Final = 0.01


@dataclass(frozen=True, slots=True)
class VisualizationConfig:
    """Immutable sizes and tessellation settings of the scene."""

    # Radii, tessellation and scales.
    dual_vertex_radius: float = _VERTEX_RADIUS
    primal_vertex_radius: float = _VERTEX_RADIUS
    curve_radius: float = _SQS_CURVE_RADIUS
    direction_arrow_radius: float = 0.009
    geodesic_points: int = 100
    sphere_stacks: int = 48
    sphere_slices: int = 96
    primal_curve_scale: float = 1.0
    invalid_primal_vertex_scale: float = 1.6

    # Volume edges, arrow origins and ground plane.
    volume_edge_width: float = 1.0
    arrow_origin_radius: float = 0.001
    ground_plane_height: float = -1.5


@dataclass(frozen=True, slots=True)
class ColorPalette:
    """RGB colors in [0, 1] and sphere opacity."""

    # Edges
    dual_single: RGB = (1.0, 1.0, 0.0)
    dual_double: RGB = (0.0, 0.5, 1.0)
    primal: RGB = (0.8, 0.0, 0.8)

    # Vertices
    dual_vertex: RGB = (1.0, 0.0, 0.0)
    auxiliary_direction_point: RGB = (0.0, 0.50, 0.0)

    # Direction arrows
    direction_arrow: RGB = (0.6, 0.3, 0.0)
    auxiliary_direction_arrow: RGB = (0.0, 0.45, 0.0)

    # Primal vertices and sphere opacity
    primal_vertex: RGB = (1.0, 0.0, 1.0)


    sphere_alpha: float = 0.6


DEFAULT_PALETTE = ColorPalette()


# Matplotlib tab20 colors, dark tones then light tones; indexed cyclically.
VOLUME_COLORS: tuple[RGB, ...] = (
    (0.12, 0.47, 0.71), (1.00, 0.50, 0.05), (0.17, 0.63, 0.17),
    (0.84, 0.15, 0.16), (0.58, 0.40, 0.74), (0.55, 0.34, 0.29),
    (0.89, 0.47, 0.76), (0.50, 0.50, 0.50), (0.74, 0.74, 0.13),
    (0.09, 0.75, 0.81), (0.68, 0.78, 0.91), (1.00, 0.73, 0.47),
    (0.60, 0.87, 0.54), (1.00, 0.60, 0.59), (0.77, 0.69, 0.84),
    (0.77, 0.61, 0.58), (0.97, 0.71, 0.82), (0.78, 0.78, 0.78),
    (0.86, 0.86, 0.55), (0.62, 0.85, 0.90),
)
