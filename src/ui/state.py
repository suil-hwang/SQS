# src/ui/state.py
from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, fields, replace
from typing import Literal, TypeAlias

from geomdl import NURBS

from geometry.constants import FloatArray
from sqs.dual import DualGraphData
from sqs.primal import PrimalGraphData
from sqs.validation import SQSValidationReport


# Notice levels and their console tags.
NoticeLevel = Literal["ok", "info", "warn", "error"]
NOTICE_TAGS: dict[NoticeLevel, str] = {"ok": "[OK]", "info": "[INFO]", "warn": "[!]", "error": "[X]"}

# Aggregate-audit fields of SQSValidationReport and their panel labels.
AUDIT_CHECK_LABELS: tuple[tuple[str, str], ...] = (
    ("is_dual_embedding_valid", "Dual geodesic embedding is valid"),
    ("is_primal_embedding_valid", "Primal geodesic embedding is valid"),
    ("has_expected_primal_counts", "Expected primal V/E/F counts"),
    ("is_primal_connected", "Primal graph connected"),
    ("is_primal_dual_incidence_consistent", "Primal-dual incidences are consistent"),
    ("has_primal_face_direction_containment", "Corresponding dual directions lie strictly inside primal faces"),
)

# Result of geometry.volume.build_shell_geometry: (shell volumes, spherical patches, projection directions,
# face-direction mapping), volumes and patches in primal-face order.
ShellGeometry: TypeAlias = tuple[tuple[NURBS.Volume, ...], tuple[NURBS.Surface, ...], FloatArray, tuple[int, ...]]


@dataclass
class LayerVisibility:
    """Scene-layer visibility toggles."""

    show_dual_edges: bool = True
    show_primal_edges: bool = True
    show_primal_vertices: bool = True
    show_dual_vertices: bool = True
    show_direction_vectors: bool = True
    show_auxiliary_directions: bool = True
    show_volumes: bool = True
    show_spheres: bool = True

    def reset_visibility(self) -> None:
        """Restore default visibility."""
        for info in fields(self):
            setattr(self, info.name, info.default)


@dataclass(slots=True)
class ValidationState:
    """Cached validation of the displayed graphs; None means not computed, a fresh instance is stale."""

    is_dual_crossing_free: bool | None = None
    is_primal_crossing_free: bool | None = None
    is_primal_vertex_placement_valid: bool | None = None
    invalid_primal_vertex_indices: list[int] = field(default_factory=list)
    great_circle_face_indices: list[int] = field(default_factory=list)
    is_audit_valid: bool | None = None
    audit_checks: dict[str, bool] = field(default_factory=dict)
    audit_error: str | None = None
    needs_refresh: bool = True
    refresh_error: str | None = None

    @classmethod
    def compute(
        cls, dual: DualGraphData, primal: PrimalGraphData,
        report: SQSValidationReport | None = None, audit_error: str | None = None,
    ) -> ValidationState:
        """Run the checks whose input graphs exist, plus a given aggregate-audit report; skipped checks stay None."""
        state = cls(needs_refresh=False, audit_error=audit_error)
        if dual.edges:
            state.is_dual_crossing_free = dual.is_crossing_free()
        # No primal edges means nothing to screen, not a vacuous pass.
        if primal.edges:
            state.is_primal_crossing_free = primal.is_crossing_free()
        if dual.faces and primal.vertices:
            invalid, great_circle = dual.classify_primal_vertices(primal.vertices)
            state.is_primal_vertex_placement_valid = not invalid
            state.invalid_primal_vertex_indices = list(invalid)
            state.great_circle_face_indices = list(great_circle)
        if report is not None:
            state.is_audit_valid = report.is_valid
            state.audit_checks = {label: getattr(report, name) for name, label in AUDIT_CHECK_LABELS}
        return state


@dataclass(eq=False)
class VolumeState:
    """Shell and column results with their generation and preview settings."""

    shell_geometry: ShellGeometry | None = field(default=None, repr=False, kw_only=True)
    column_volumes: list[NURBS.Volume] | None = field(default=None, repr=False, kw_only=True)
    column_direction_indices: list[int] | None = field(default=None, repr=False, kw_only=True)

    preview_resolution: int = 10
    shell_thickness: float = 0.05
    column_height: float = 0.2

    @property
    def has_shell(self) -> bool:
        return self.shell_geometry is not None

    @property
    def has_column_result(self) -> bool:
        """Whether column generation completed, including an empty result."""
        return self.column_volumes is not None

    def set_shell_geometry(self, result: ShellGeometry | None) -> None:
        """Set or clear the shell; columns are always discarded."""
        self.shell_geometry = result
        self.set_column_volumes(None)

    def set_column_volumes(
        self, volumes: Sequence[NURBS.Volume] | None, *, direction_indices: Iterable[int] | None = None
    ) -> None:
        """Set the columns and their input-direction indices; None means not generated, [] a completed empty result."""
        self.column_volumes = None if volumes is None else list(volumes)
        self.column_direction_indices = None if direction_indices is None else list(direction_indices)

    def update_shell_thickness(self, value: float) -> bool:
        """Set shell thickness and return whether it changed; a change discards shell and columns."""
        if abs(value - self.shell_thickness) <= 1e-12:
            return False
        self.shell_thickness = value
        self.set_shell_geometry(None)
        return True

    def update_column_height(self, value: float) -> bool:
        """Set column height and return whether it changed; a change discards the columns only."""
        if abs(value - self.column_height) <= 1e-12:
            return False
        self.column_height = value
        self.set_column_volumes(None)
        return True

    def update_preview_resolution(self, value: int) -> bool:
        """Set preview resolution and return whether it changed; generated geometry is kept."""
        changed = value != self.preview_resolution
        self.preview_resolution = value
        return changed

    def clone(self) -> VolumeState:
        """Return a rollback copy; list fields are copied shallowly, the shell tuple is shared."""
        return replace(self, **{
            info.name: list(value) for info in fields(self) if isinstance(value := getattr(self, info.name), list)
        })
