# src/ui/view.py
from __future__ import annotations

from collections.abc import Callable, Sequence
from enum import Enum
from typing import Final, Literal

import numpy as np
import polyscope as ps
from polyscope.point_cloud import PointCloud
from polyscope.structure import Structure

from geometry.arc import interpolate_minor_arc
from geometry.constants import FloatArray, IntArray
from geometry.volume import create_uv_sphere_mesh
from sqs.dual import DualGraphData
from sqs.primal import PrimalGraphData, PrimalVertex
from ui.state import LayerVisibility
from ui.visual_config import RGB, DEFAULT_PALETTE, VOLUME_COLORS, ColorPalette, VisualizationConfig


class Layer(Enum):
    """Polyscope structure groups registered independently; each value is the structure name."""

    SPHERE = "Unit Sphere"
    INPUT_POINTS = "Dual Vertices"
    INPUT_ARROWS = "Direction Vectors"
    AUXILIARY_POINTS = "Generated Dual Vertices"
    AUXILIARY_ARROWS = "Generated Direction Vectors"
    DUAL_SINGLE_EDGES = "Dual Single Edges"
    DUAL_DOUBLE_EDGES = "Dual Double Edges"
    PRIMAL_EDGES = "Primal Edges"
    PRIMAL_VERTICES = "Primal Vertices"
    INVALID_PRIMAL_VERTICES = "Invalid Primal Vertices"
    # Volume values are name prefixes; preview k is named prefix + k.
    SHELL_VOLUMES = "vol_shell_"
    COLUMN_VOLUMES = "vol_col_"


# (label, LayerVisibility flag, gated layers), shared by view, panel and status text;
# a layer shows iff every gating flag is on.
VISIBILITY_TOGGLES: Final[tuple[tuple[str, str, tuple[Layer, ...]], ...]] = (
    ("Unit Sphere", "show_spheres", (Layer.SPHERE,)),
    ("Dual Vertices", "show_dual_vertices", (Layer.INPUT_POINTS, Layer.AUXILIARY_POINTS)),
    ("Direction Vectors", "show_direction_vectors", (Layer.INPUT_ARROWS, Layer.AUXILIARY_ARROWS)),
    ("Additional Vectors", "show_auxiliary_directions", (Layer.AUXILIARY_POINTS, Layer.AUXILIARY_ARROWS)),
    ("Dual Edges", "show_dual_edges", (Layer.DUAL_SINGLE_EDGES, Layer.DUAL_DOUBLE_EDGES)),
    ("Primal Edges", "show_primal_edges", (Layer.PRIMAL_EDGES,)),
    ("Primal Vertices", "show_primal_vertices", (Layer.PRIMAL_VERTICES, Layer.INVALID_PRIMAL_VERTICES)),
    ("Volumes", "show_volumes", (Layer.SHELL_VOLUMES, Layer.COLUMN_VOLUMES)),
)


class SceneView:
    """Polyscope scene of one editor session, as a registry of layers."""

    def __init__(self, config: VisualizationConfig, palette: ColorPalette | None = None) -> None:
        self.config = config
        self.palette = palette or DEFAULT_PALETTE
        self._arc_parameters = np.linspace(0.0, 1.0, config.geodesic_points)
        # Invariant: a layer is cleared, removing its structures, before it is registered again.
        self._layers: dict[Layer, list[Structure]] = {layer: [] for layer in Layer}

    def initialize_polyscope(self) -> None:
        """Initialize Polyscope without a prefs file: z-up, shadow-only ground plane at the configured height."""
        ps.set_use_prefs_file(False)
        ps.init()
        ps.set_up_dir("z_up")
        ps.set_ground_plane_mode("shadow_only")
        ps.set_ground_plane_height(self.config.ground_plane_height)

    def show(self, imgui_callback: Callable[[], None]) -> None:
        """Register the per-frame callback, then enter the Polyscope loop."""
        ps.set_user_callback(imgui_callback)
        ps.show()

    def has(self, layer: Layer) -> bool:
        """Whether the layer currently has registered structures."""
        return bool(self._layers[layer])

    def clear(self, *layers: Layer) -> None:
        """Remove the structures of the given layers, or of every layer when none is given."""
        for layer in layers or tuple(Layer):
            for structure in self._layers[layer]:
                structure.remove()
            self._layers[layer].clear()

    def apply_visibility(self, layer_visibility: LayerVisibility) -> None:
        """Enable each registered structure exactly when every toggle gating its layer is on."""
        for layer, structures in self._layers.items():
            visible = all(getattr(layer_visibility, flag) for _, flag, gated in VISIBILITY_TOGGLES if layer in gated)
            for structure in structures:
                structure.set_enabled(visible)

    def show_directions(self, directions: FloatArray, auxiliary_count: int) -> None:
        """Register the unit sphere and the direction rows; the last auxiliary_count rows are auxiliary."""
        self.clear(Layer.SPHERE, Layer.INPUT_POINTS, Layer.INPUT_ARROWS, Layer.AUXILIARY_POINTS, Layer.AUXILIARY_ARROWS)
        config, palette = self.config, self.palette
        sphere = ps.register_surface_mesh(
            Layer.SPHERE.value, *create_uv_sphere_mesh(stacks=config.sphere_stacks, slices=config.sphere_slices),
            smooth_shade=True,
        )
        sphere.set_transparency(palette.sphere_alpha)
        self._layers[Layer.SPHERE].append(sphere)
        split = len(directions) - min(max(0, auxiliary_count), len(directions))
        self._register_direction_group(
            directions[:split], point_layer=Layer.INPUT_POINTS, arrow_layer=Layer.INPUT_ARROWS,
            point_color=palette.dual_vertex, arrow_color=palette.direction_arrow,
        )
        self._register_direction_group(
            directions[split:], point_layer=Layer.AUXILIARY_POINTS, arrow_layer=Layer.AUXILIARY_ARROWS,
            point_color=palette.auxiliary_direction_point, arrow_color=palette.auxiliary_direction_arrow,
        )

    def show_graphs(self, dual: DualGraphData, primal: PrimalGraphData) -> None:
        """Register each nonempty edge layer as one curve network of sampled minor arcs, then the primal vertices."""
        self.clear(Layer.DUAL_SINGLE_EDGES, Layer.DUAL_DOUBLE_EDGES, Layer.PRIMAL_EDGES, Layer.PRIMAL_VERTICES)
        config, palette = self.config, self.palette
        directions, vertices = dual.vertex_positions, primal.vertex_positions
        dual_pairs = {m: [edge.endpoint_indices for edge in dual.edges if edge.multiplicity == m] for m in (1, 2)}
        self._register_edges(Layer.DUAL_SINGLE_EDGES, directions, dual_pairs[1], config.curve_radius, palette.dual_single)
        self._register_edges(Layer.DUAL_DOUBLE_EDGES, directions, dual_pairs[2], config.curve_radius, palette.dual_double)
        self._register_edges(
            Layer.PRIMAL_EDGES, vertices, [edge.endpoint_indices for edge in primal.edges],
            config.curve_radius * config.primal_curve_scale, palette.primal,
        )
        if primal.vertices:
            self._register_points(Layer.PRIMAL_VERTICES, vertices, config.primal_vertex_radius, palette.primal_vertex)

    def show_invalid_primal_vertices(
        self, primal_vertices: Sequence[PrimalVertex], invalid_vertex_indices: Sequence[int],
    ) -> None:
        """Replace the enlarged overlay of invalid primal vertices; out-of-range indices are skipped."""
        self.clear_invalid_primal_vertices()
        positions = [primal_vertices[i].position for i in invalid_vertex_indices if 0 <= i < len(primal_vertices)]
        if positions:
            config = self.config
            self._register_points(
                Layer.INVALID_PRIMAL_VERTICES, np.asarray(positions, dtype=np.float64),
                config.primal_vertex_radius * config.invalid_primal_vertex_scale, self.palette.primal_vertex,
            )

    def clear_invalid_primal_vertices(self) -> None:
        """Remove the invalid-primal-vertex overlay."""
        self.clear(Layer.INVALID_PRIMAL_VERTICES)

    def show_volume_previews(
        self, meshes: Sequence[tuple[FloatArray, IntArray]], *,
        layer: Literal[Layer.SHELL_VOLUMES, Layer.COLUMN_VOLUMES], palette_index_by_volume: Sequence[int] | None = None,
    ) -> None:
        """Replace the layer's previews; mesh k gets palette color palette_index_by_volume[k] (default k), cyclic."""
        if layer is not Layer.SHELL_VOLUMES and layer is not Layer.COLUMN_VOLUMES:
            raise ValueError("volume preview layer must be SHELL_VOLUMES or COLUMN_VOLUMES")
        indices = range(len(meshes)) if palette_index_by_volume is None else palette_index_by_volume
        # strict: a length mismatch raises before the layer is cleared.
        previews = list(zip(meshes, (VOLUME_COLORS[i % len(VOLUME_COLORS)] for i in indices), strict=True))
        self.clear(layer)
        for index, ((vertices, hexes), color) in enumerate(previews):
            self._layers[layer].append(ps.register_volume_mesh(
                f"{layer.value}{index}", vertices, hexes=hexes, color=color, edge_width=self.config.volume_edge_width,
            ))

    def clear_volume_previews(self) -> None:
        """Remove the shell and column previews."""
        self.clear(Layer.SHELL_VOLUMES, Layer.COLUMN_VOLUMES)

    @property
    def volume_preview_count(self) -> int:
        """Number of registered shell and column preview meshes."""
        return len(self._layers[Layer.SHELL_VOLUMES]) + len(self._layers[Layer.COLUMN_VOLUMES])

    def _register_direction_group(
        self, directions: FloatArray, *, point_layer: Layer, arrow_layer: Layer, point_color: RGB, arrow_color: RGB,
    ) -> None:
        """Register one group's points on the sphere and arrows from the origin."""
        if not len(directions):
            return
        config = self.config
        self._register_points(point_layer, directions, config.dual_vertex_radius, point_color)
        origins = self._register_points(
            arrow_layer, np.zeros((len(directions), 3)), config.arrow_origin_radius, arrow_color,
        )
        origins.add_vector_quantity(
            "arrows", directions, enabled=True, vectortype="ambient",
            radius=config.direction_arrow_radius, color=arrow_color,
        )

    def _register_edges(
        self, layer: Layer, positions: FloatArray, pairs: Sequence[tuple[int, int]], radius: float, color: RGB,
    ) -> None:
        """Register one curve network of sampled minor arcs; no-op without pairs."""
        if not pairs:
            return
        arcs = [interpolate_minor_arc(positions[a], positions[b], self._arc_parameters) for a, b in pairs]
        nodes = np.arange(len(arcs) * len(self._arc_parameters)).reshape(len(arcs), -1)  # Row k: nodes of arc k.
        segments = np.stack((nodes[:, :-1], nodes[:, 1:]), axis=-1).reshape(-1, 2)
        network = ps.register_curve_network(layer.value, np.concatenate(arcs), segments)
        network.set_color(color)
        network.set_radius(radius, relative=False)
        self._layers[layer].append(network)

    def _register_points(self, layer: Layer, positions: FloatArray, radius: float, color: RGB) -> PointCloud:
        cloud = ps.register_point_cloud(layer.value, positions)
        cloud.set_radius(radius, relative=False)
        cloud.set_color(color)
        self._layers[layer].append(cloud)
        return cloud
