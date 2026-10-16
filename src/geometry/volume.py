# src/geometry/volume.py
from __future__ import annotations

from collections.abc import Sequence
from operator import index

import numpy as np
from geomdl import NURBS
from numpy.typing import ArrayLike
from scipy.interpolate import BSpline

from . import bezier_surface
from .constants import FloatArray, IntArray, Vec3
from .vector import normalize_vector


def create_radial_shell_volume(surface: NURBS.Surface, inner_radius: float, outer_radius: float) -> NURBS.Volume:
    """Create a radial NURBS shell from an origin-centered unit-sphere patch.

    Preserved weights give V=((1-w)*inner_radius+w*outer_radius)*S(u,v),
    with w in [0,1]. The input patch is assumed to lie on the unit sphere.
    """
    if not 0.0 < inner_radius < outer_radius < np.inf:
        raise ValueError("radii must be finite and satisfy 0 < inner_radius < outer_radius")

    # V=r(w)S, det DV=r^2 r' dot(S_u x S_v,S).
    # Each layer copies the homogeneous controls and scales XYZ, preserving weights.
    layers: list[FloatArray] = []
    for radius in (inner_radius, outer_radius):
        layer = np.array(surface.ctrlptsw, dtype=np.float64, copy=True)
        if layer.ndim == 2 and layer.shape[1] == 4:
            with np.errstate(over="ignore", invalid="ignore"):
                layer[:, :3] *= radius
        if layer.ndim != 2 or layer.shape[1] != 4 or not np.isfinite(layer).all():
            raise ValueError("Invalid or overflowing homogeneous controls.")
        layers.append(layer)
    start_layer, end_layer = layers

    # Equal layer weights make the radial interpolation affine in w.
    volume = NURBS.Volume(normalize_kv=False)
    volume.degree = [*surface.degree, 1]
    # geomdl stores contiguous w layers with v varying fastest within each layer.
    volume.set_ctrlpts(np.concatenate((start_layer, end_layer)).tolist(), *surface.cpsize, 2)
    volume.knotvector = [*(list(knots) for knots in surface.knotvector), [0.0, 0.0, 1.0, 1.0]]
    return volume


def create_projected_column_volume(
    surface: NURBS.Surface, direction_vector: Vec3, outer_radius: float, column_height: float,
) -> NURBS.Volume:
    """Create V=(1-w)B+wT between a scaled surface and a planar bilinear cap."""
    if not 0.0 < outer_radius < np.inf:
        raise ValueError("outer_radius must be positive and finite.")

    # Copy the homogeneous controls and scale XYZ, preserving weights.
    base_layer = np.array(surface.ctrlptsw, dtype=np.float64, copy=True)
    if base_layer.ndim == 2 and base_layer.shape[1] == 4:
        with np.errstate(over="ignore", invalid="ignore"):
            base_layer[:, :3] *= outer_radius
    if base_layer.ndim != 2 or base_layer.shape[1] != 4 or not np.isfinite(base_layer).all():
        raise ValueError("Invalid or overflowing homogeneous controls.")

    # The bilinear top uses corner order (00, 10, 01, 11).
    plane_offset = outer_radius + column_height
    if column_height <= 0.0 or not np.isfinite(plane_offset):
        raise ValueError("column_height must be positive with a finite plane offset")

    unit_direction = normalize_vector(direction_vector, name="direction_vector")
    nu, nv = surface.cpsize
    rows = base_layer[[0, (nu - 1) * nv, nv - 1, nu * nv - 1]]
    base_corners = rows[:, :3] / rows[:, 3, None]
    # Orthogonally project the base corners onto the requested top plane.
    top_corners = base_corners + (plane_offset - base_corners @ unit_direction)[:, None] * unit_direction

    if top_corners.shape != (4, 3) or not np.isfinite(top_corners).all():
        raise ValueError("top corners must be finite with shape (4, 3)")

    # Build each basis function's products with normalized factors (1-t,t).
    product_bases, product_knots = [], []
    for knotvector, degree, count in zip(surface.knotvector, surface.degree, surface.cpsize, strict=True):
        knots = np.asarray(knotvector, dtype=np.float64)
        spline = BSpline(knots, np.eye(count), degree)
        breaks, multiplicities = np.unique(knots, return_counts=True)
        lower, width = breaks[0], float(breaks[-1]) - float(breaks[0])
        # BSpline ensures positive width; check overflow and clamped multiplicities.
        if not (np.isfinite(width) and multiplicities[0] == multiplicities[-1] == multiplicities.max() == degree + 1):
            raise ValueError("column surface must have regular clamped knot vectors")

        # Interior multiplicity p exposes each degree-p Bernstein span.
        for knot, multiplicity in zip(breaks[1:-1], multiplicities[1:-1], strict=True):
            if multiplicity < degree:
                spline = spline.insert_knot(float(knot), degree - int(multiplicity))

        parts = []
        alpha = (np.arange(degree + 1, dtype=np.float64) / (degree + 1))[:, None, None]
        for span in np.flatnonzero(np.diff(spline.t) > 0.0):
            local = spline.c[span - degree:span + 1, :, None]
            # Express the global linear factors at this span's two endpoints.
            t = (spline.t[span:span + 2] - lower) / width
            linear = np.column_stack((1.0 - t, t))
            # Multiplication by a linear factor raises the degree from p to p+1.
            factors = np.zeros((degree + 2, count, 2), dtype=np.float64)
            factors[:-1] = (1.0 - alpha) * local * linear[0]
            factors[1:] += (1.0 - alpha[::-1]) * local * linear[1]

            # Share continuous endpoints; retain existing knot discontinuities.
            separate = np.count_nonzero(knots == spline.t[span]) == degree + 1
            parts.append(factors if separate else factors[1:])

        product_bases.append(np.concatenate(parts))
        product_knots.append(np.repeat(breaks, np.maximum(multiplicities, degree) + 1).tolist())

    bu, bv = product_bases
    base_net = base_layer.reshape(*surface.cpsize, 4)

    # Summing the linear factors elevates (N,W); top (WT,W) preserves W, making Cartesian w affine.
    with np.errstate(over="ignore", invalid="ignore"):
        bottom = np.einsum("ai,bj,ijc->abc", bu.sum(axis=-1), bv.sum(axis=-1), base_net, optimize=True)
        top = np.empty_like(bottom)
        top[..., :3] = np.einsum("ais,bjt,ij,tsc->abc", bu, bv, base_net[..., 3], top_corners.reshape(2, 2, 3), optimize=True)
        top[..., 3] = bottom[..., 3]

    if not (np.isfinite(bottom).all() and np.isfinite(top).all()):
        raise ValueError("Invalid or overflowing homogeneous controls.")

    # geomdl stores w layers contiguously, with v varying fastest in each layer.
    volume = NURBS.Volume(normalize_kv=False)
    volume.degree = [surface.degree_u + 1, surface.degree_v + 1, 1]
    volume.set_ctrlpts(np.stack((bottom, top)).reshape(-1, 4).tolist(), len(bu), len(bv), 2)
    volume.knotvector = [*product_knots, [0.0, 0.0, 1.0, 1.0]]
    return volume


def build_shell_geometry(
    *, primal_vertex_positions: ArrayLike, direction_vectors: ArrayLike,
    primal_faces: Sequence[Sequence[int]], face_direction_indices: Sequence[int], shell_thickness: float,
    degenerate_primal_vertex_indices: Sequence[int] = (),
) -> tuple[tuple[NURBS.Volume, ...], tuple[NURBS.Surface, ...], FloatArray, tuple[int, ...]]:
    """Snapshot SQS inputs and build face-ordered shells with outer radius 1."""
    positions = np.array(primal_vertex_positions, dtype=np.float64, order="C", copy=True)
    directions = np.array(direction_vectors, dtype=np.float64, order="C", copy=True)
    faces, mapping = tuple(map(tuple, primal_faces)), tuple(face_direction_indices)
    degenerate = tuple(degenerate_primal_vertex_indices)
    if degenerate:
        raise ValueError(f"Shell generation: great-circle-degenerate primal vertices at {degenerate}")
    batch = bezier_surface.primal_faces_to_bezier_surfaces(faces, positions, mapping, directions)
    projection_directions = np.array(batch.sweep_directions, dtype=np.float64, order="C", copy=True)
    if projection_directions.shape == (0,):
        projection_directions = projection_directions.reshape(0, 3)
    projection_directions.setflags(write=False)
    return (
        tuple(create_radial_shell_volume(surface, 1.0 - shell_thickness, 1.0) for surface in batch.surfaces),
        tuple(batch.surfaces), projection_directions, mapping,
    )


def build_column_volumes(
    *, spherical_patches: Sequence[NURBS.Surface], projection_directions: ArrayLike,
    outer_radius: float, column_height: float, face_indices: Sequence[int] | None = None,
) -> list[NURBS.Volume]:
    """Build selected columns from aligned patches and projection directions."""
    patches = tuple(spherical_patches)
    directions = np.array(projection_directions, dtype=np.float64, order="C", copy=True)
    if directions.shape == (0,):
        directions = directions.reshape(0, 3)
    count = len(patches)
    if directions.shape != (count, 3) or not np.isfinite(directions).all():
        raise ValueError(f"projection_directions must have finite shape ({count}, 3)")
    selected = tuple(range(count)) if face_indices is None else tuple(face_indices)
    if any(not isinstance(i, (int, np.integer)) or isinstance(i, bool) or not 0 <= i < count for i in selected) or len(set(selected)) != len(selected):
        raise ValueError("face_indices must be distinct in-range integers")
    return [
        create_projected_column_volume(patches[i], directions[i], outer_radius, column_height)
        for i in selected
    ]


def sample_volume_hex_meshes(
    volumes: Sequence[NURBS.Volume], samples_per_parameter_axis: int | np.integer = 5,
) -> list[tuple[FloatArray, IntArray]]:
    """Sample NURBS volumes on uniform parameter grids for hex-mesh previews."""
    n = index(samples_per_parameter_axis)
    if n < 2:
        raise ValueError("samples_per_parameter_axis must be >= 2")
    if not volumes:
        return []

    # Flattened (u,v,w) vertices have strides (n**2, n, 1).
    cell_start_indices = np.arange(n**3, dtype=np.int64).reshape((n,) * 3)[:-1, :-1, :-1].reshape(-1, 1)
    u_stride = n**2
    hex_corner_offsets = np.array((0, u_stride, u_stride + n, n, 1, u_stride + 1, u_stride + n + 1, n + 1), dtype=np.int64)
    hex_connectivity = cell_start_indices + hex_corner_offsets
    hex_connectivity.setflags(write=False)
    basis_cache: dict[tuple[int, tuple[float, ...]], FloatArray] = {}
    meshes: list[tuple[FloatArray, IntArray]] = []
    for volume in volumes:
        nu, nv, nw = volume.cpsize
        # geomdl stores (w, u, v); contract u, v, w and append sampled axes.
        homogeneous_control_net = np.asarray_chkfinite(volume.ctrlptsw, dtype=np.float64).reshape(nw, nu, nv, 4)
        homogeneous_samples = homogeneous_control_net.transpose(1, 2, 0, 3)
        for degree, knot_vector in zip(volume.degree, volume.knotvector, strict=True):
            knots = tuple(knot_vector)
            key = degree, knots
            if key not in basis_cache:
                parameters = np.linspace(knots[degree], knots[-degree - 1], n)
                basis_cache[key] = BSpline.design_matrix(parameters, knots, degree).toarray()
            homogeneous_samples = np.tensordot(homogeneous_samples, basis_cache[key], axes=(0, 1))
        homogeneous_samples = np.asarray_chkfinite(np.moveaxis(homogeneous_samples, 0, -1).reshape(-1, 4), dtype=np.float64)
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            vertices = np.asarray_chkfinite(homogeneous_samples[:, :3] / homogeneous_samples[:, 3:], dtype=np.float64)
        meshes.append((vertices, hex_connectivity))

    return meshes


def create_uv_sphere_mesh(radius: float = 1.0, stacks: int = 40, slices: int = 80) -> tuple[FloatArray, IntArray]:
    """Create a fresh, read-only UV-sphere triangle mesh for rendering."""
    import trimesh

    sphere_mesh = trimesh.creation.uv_sphere(radius=radius, count=[stacks, slices])
    vertices, triangles = sphere_mesh.vertices.astype(np.float64), sphere_mesh.faces.astype(int)
    vertices.flags.writeable = triangles.flags.writeable = False
    return vertices, triangles
