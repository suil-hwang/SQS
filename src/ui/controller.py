# src/ui/controller.py
from __future__ import annotations

import time
import traceback
from collections import deque
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Literal

import numpy as np

from assignments.candidate_graphs import CandidateDualGraphId, SUPPORTED_DIRECTION_COUNTS
from assignments.results import SearchResult
from geometry.spherical_sampling import spherical_fibonacci_direction_set
from geometry.volume import build_column_volumes, build_shell_geometry, sample_volume_hex_meshes
from sqs import validation
from sqs.dual import DualGraphData
from sqs.primal import PrimalGraphData
from sqs.validation import SQSValidationReport
from ui.messages import commit_caveats, graph_label, pinned_status, trial_summary
from ui.model import GraphCounts, SQSModel
from ui.sqs_solver import solve_sqs
from ui.state import NOTICE_TAGS, NoticeLevel, ValidationState
from ui.view import VISIBILITY_TOGGLES, Layer, SceneView

BuildStage = Literal["auto", "dual", "primal"]

_MAX_LOG_ENTRIES = 200
_REPORTED_DURATION_S = 0.5  # Actions at least this long report their wall time.


@dataclass(frozen=True, slots=True)
class SearchAttempt:
    """The latest search, committed or not, with the direction count it ran on."""

    num_directions: int
    result: SearchResult


class SQSController:
    """Run editor actions as transactions on the model and keep the view in step."""

    def __init__(self, model: SQSModel, view: SceneView):
        """Bind the model to edit and the view to drive."""
        self.model = model
        self.view = view
        self._committed_build: tuple[SearchResult | None, SQSValidationReport | None] = (None, None)
        # last_attempt and log lie outside session snapshots and survive a rollback.
        self.last_attempt: SearchAttempt | None = None
        self.log: deque[tuple[NoticeLevel, str]] = deque(maxlen=_MAX_LOG_ENTRIES)

    @property
    def committed_result(self) -> SearchResult | None:
        """Most recent committed search result."""
        return self._committed_build[0]

    @property
    def committed_audit_report(self) -> SQSValidationReport | None:
        """Aggregate-audit report paired with the committed search result."""
        return self._committed_build[1]

    # --- Feedback, the action boundary and the transaction ---

    def notify(self, level: NoticeLevel, text: str) -> None:
        """Print a tagged console line and append it to the log read by the panel."""
        print(f"{NOTICE_TAGS[level]} {text}")
        self.log.append((level, text))

    def _refuse(self, reason: str) -> Literal[False]:
        """Warn why an action did not run (nothing changed) and return False."""
        self.notify("warn", reason)
        return False

    def run_action(self, label: str, action: Callable[[], object]) -> object | None:
        """Run an action; report failure or cancellation instead of raising into the frame loop."""
        started_at = time.perf_counter()
        try:
            return action()
        except KeyboardInterrupt:
            self.notify("warn", f"{label} cancelled")
        except Exception as error:
            traceback.print_exc()
            self.notify("error", f"{label} failed: {error}")
        finally:
            if (elapsed := time.perf_counter() - started_at) >= _REPORTED_DURATION_S:
                self.notify("info", f"{label}: {elapsed:.2f} s")
        return None

    @contextmanager
    def _transaction(self) -> Generator[None, None, None]:
        """Restore the session snapshot, the build cache and the scene when the body raises."""
        snapshot, cache = self.model.create_snapshot(), self._committed_build
        try:
            yield
        except BaseException:
            self.model.restore_snapshot(snapshot)
            self._committed_build = cache
            try:
                self.rebuild_view()
            except BaseException as restore_error:  # Never mask the original failure.
                self.notify("error", f"View restore failed: {restore_error}")
            raise

    # --- Scene ---

    def rebuild_view(self) -> None:
        """Re-register all layers from the model, then apply the saved visibility."""
        model, view = self.model, self.view
        prepared_directions, validation_state, volume_state = (
            model.prepared_directions, model.validation_state, model.volume_state,
        )
        view.clear()
        view.show_directions(
            model.dual.vertex_positions,
            0 if prepared_directions is None else prepared_directions.auxiliary_direction_count,
        )
        view.show_graphs(model.dual, model.primal)
        if validation_state.is_primal_vertex_placement_valid is False:
            view.show_invalid_primal_vertices(model.primal.vertices, validation_state.invalid_primal_vertex_indices)

        # The shell snapshot owns the face-direction mapping; columns keep the input-direction subset.
        shell = volume_state.shell_geometry
        shell_volumes, shell_mapping = ((), None) if shell is None else (shell[0], shell[3])
        column_mapping = volume_state.column_direction_indices
        for volumes, layer, mapping in (
            (shell_volumes, Layer.SHELL_VOLUMES, shell_mapping),
            (volume_state.column_volumes, Layer.COLUMN_VOLUMES, shell_mapping if column_mapping is None else column_mapping),
        ):
            if volumes:
                view.show_volume_previews(
                    sample_volume_hex_meshes(volumes, volume_state.preview_resolution),
                    layer=layer, palette_index_by_volume=mapping,
                )
        view.apply_visibility(model.layer_visibility)

    def apply_visibility(self) -> None:
        """Apply the saved visibility toggles to every registered layer."""
        self.view.apply_visibility(self.model.layer_visibility)

    # --- Direction actions ---

    def generate_directions(self, num_directions: int) -> bool:
        """Replace the input with n Fibonacci directions; clear graphs, volumes and results."""
        if type(num_directions) is not int or num_directions not in SUPPORTED_DIRECTION_COUNTS:
            return self._refuse(
                f"Direction count must be an integer in {min(SUPPORTED_DIRECTION_COUNTS)}..{max(SUPPORTED_DIRECTION_COUNTS)}"
            )
        with self._transaction():
            directions = np.asarray(spherical_fibonacci_direction_set(num_directions), dtype=np.float64)
            self.model.replace_input_direction_set(directions)
            self.rebuild_view()
            self._committed_build, self.last_attempt = (None, None), None  # Earlier results no longer apply.
        self.notify("ok", f"Generated {num_directions} Fibonacci directions")
        return True

    def activate_auxiliary_directions(self) -> bool:
        """Activate the effective directions: input plus the configured auxiliary directions."""
        model = self.model
        n = model.configured_effective_direction_count()
        if model.dual.num_directions == 0:
            return self._refuse("Generate direction vectors first")
        if model.configured_auxiliary_direction_count <= 0:
            return self._refuse("Set Additional Vectors to a positive value first")
        if n not in SUPPORTED_DIRECTION_COUNTS:
            return self._refuse(f"Effective direction count {n} is unsupported")
        # Preparation changes nothing, so only the commit needs the transaction.
        prepared = model.prepare_solve_directions()
        with self._transaction():
            model.activate_effective_direction_set(
                prepared.direction_vectors, prepared_directions=prepared,
            )
            model.layer_visibility.show_auxiliary_directions = True
            self.rebuild_view()
            self._committed_build, self.last_attempt = (None, None), None  # Earlier results no longer apply.
        self.notify("ok", f"Activated {prepared.auxiliary_direction_count} additional directions (n={n})")
        return True

    def deactivate_auxiliary_directions(self) -> bool:
        """Deactivate auxiliary directions and restore the input directions."""
        model = self.model
        if model.prepared_directions is None or not model.prepared_directions.has_auxiliary_directions:
            return self._refuse("No additional direction vectors are currently active")
        with self._transaction():
            model.restore_input_direction_set()
            self.rebuild_view()
            self._committed_build, self.last_attempt = (None, None), None  # Earlier results no longer apply.
        self.notify("ok", f"Restored the {model.dual.num_directions} input directions")
        return True

    # --- Build actions ---

    def build_structure(self, stage: BuildStage = "auto", pinned_candidate: CandidateDualGraphId | None = None) -> bool:
        """Commit the graphs of an SQS that passed the aggregate audit."""
        model = self.model
        if model.dual.num_directions == 0:
            return self._refuse("Generate direction vectors first")
        if stage == "primal":
            # Reuse the approved SQS without solving or reorienting faces.
            (result, report), prepared_directions = self._committed_build, None
            if result is None or result.selected is None:
                return self._refuse("Build dual graph first (no cached SQS available)")
        else:
            prepared_directions = model.prepare_solve_directions()
            n = len(prepared_directions.direction_vectors)
            result = solve_sqs(
                prepared_directions.direction_vectors,
                solver_config=model.solver_config, pinned_candidate=pinned_candidate,
            )
            self.last_attempt = SearchAttempt(n, result)
            scope = graph_label(n, result.graph_class, None if pinned_candidate is None else pinned_candidate.graph_id)
            # Acceptance gates: a realization, of exactly these directions, that passes the aggregate audit.
            if result.selected is None:
                verdict = f"{result.outcome.value} ({trial_summary(result)})" if pinned_candidate is None else pinned_status(result)
                return self._refuse(f"{scope}: {verdict}, nothing committed")
            if not prepared_directions.is_realized_by(result.selected.sqs.dual):
                return self._refuse(f"{scope}: solver directions do not match the prepared direction set, nothing committed")
            report = validation.validate(result.selected.sqs)
            if not report.is_valid:
                return self._refuse(f"{scope}: the final SQS audit failed, nothing committed")

        selected = result.selected
        sqs = selected.sqs
        primal = PrimalGraphData(sqs.primal.vertices, (), ()) if stage == "dual" else sqs.primal
        with self._transaction():
            model.dual, model.primal = sqs.dual, primal
            model.face_direction_indices = tuple(range(primal.num_faces)) if primal.faces else None
            if prepared_directions is not None:
                model.prepared_directions = (
                    prepared_directions if prepared_directions.has_auxiliary_directions else None
                )
            model.volume_state.set_shell_geometry(None)
            model.invalidate_validation()
            self.rebuild_view()
            self._committed_build = (result, report)

        built = {"dual": "Dual graph", "primal": "Primal graph", "auto": "Dual and primal graphs"}[stage]
        # A commit is not a settled search: the geometry and what the search left open are named.
        caveats = commit_caveats(result)
        message = "; ".join([
            f"{built} committed: {graph_label(sqs.dual.num_directions, result.graph_class, selected.graph_id)}",
            f"geometry {selected.geometry_status.value.lower()} (pipeline screening)", *caveats,
        ])
        hidden = [label for label, flag, _ in VISIBILITY_TOGGLES if not getattr(model.layer_visibility, flag)]
        self.notify("info", GraphCounts.from_graphs(model.dual, model.primal).summary())
        self.notify("warn" if caveats else "ok", message + (f" (hidden: {', '.join(hidden)})" if hidden else ""))
        # A dual-only display has no aggregate audit of its displayed graphs; refresh all other diagnostics now.
        self.run_action("Validation", lambda: self.refresh_validation(report if stage != "dual" else None))
        return True

    # --- Validation ---

    def refresh_validation(self, report: SQSValidationReport | None = None) -> None:
        """Recompute validation of the displayed graphs, log one summary line and update the invalid-vertex overlay."""
        model = self.model
        dual, primal = model.dual, model.primal
        audit_error = None
        # The aggregate audit applies only to a complete primal graph.
        if report is None and primal.faces:
            try:
                report = validation.validate(model.sqs)
            except Exception as error:  # For example a face-direction mapping that is not a permutation.
                audit_error = str(error)
        try:
            validation_state = ValidationState.compute(dual, primal, report, audit_error)
        except Exception as error:  # Recorded for the panel, so a failing check does not rerun every frame.
            validation_state = ValidationState(needs_refresh=False, refresh_error=str(error))
            self.notify("error", f"Validation failed: {error}")
        model.validation_state = validation_state

        # One log line names every failed check; the panel also keeps crossing failures in view.
        checks = {
            "dual crossing screening": validation_state.is_dual_crossing_free,
            "primal crossing screening": validation_state.is_primal_crossing_free,
            "primal vertex placement": validation_state.is_primal_vertex_placement_valid,
            "aggregate audit": validation_state.is_audit_valid,
        }
        problems = [f"{name} failed" for name, passed in checks.items() if passed is False]
        if failed_checks := [label for label, passed in validation_state.audit_checks.items() if passed is False]:
            problems.append(f"failed audit checks: {'; '.join(failed_checks)}")
        if validation_state.audit_error is not None:
            problems.append(f"aggregate audit raised: {validation_state.audit_error}")
        if validation_state.great_circle_face_indices:
            problems.append(
                f"great-circle-degenerate faces {validation_state.great_circle_face_indices}"
                " (admissible, unsuitable for volumes)"
            )
        if problems or any(passed is not None for passed in checks.values()):
            self.notify("warn" if problems else "ok", f"Validation: {'; '.join(problems) or 'all computed checks passed'}")

        if validation_state.invalid_primal_vertex_indices:
            self.view.show_invalid_primal_vertices(primal.vertices, validation_state.invalid_primal_vertex_indices)
            self.apply_visibility()
        else:
            self.view.clear_invalid_primal_vertices()

    def refresh_validation_if_needed(self) -> None:
        """Refresh validation only when the displayed graphs changed."""
        if self.model.validation_state.needs_refresh:
            self.refresh_validation()

    # --- Volume actions ---

    def generate_shell(self) -> bool:
        """Build the shell and sample its previews outside the transaction; commit them inside."""
        model = self.model
        primal, dual, volume_state = model.primal, model.dual, model.volume_state
        if not primal.faces or model.face_direction_indices is None:
            return self._refuse("Build primal graph first (need primal faces)")
        # Build actions commit a validated SQS and its face-direction mapping.
        shell = build_shell_geometry(
            shell_thickness=volume_state.shell_thickness,
            primal_vertex_positions=primal.vertex_positions,
            direction_vectors=dual.vertex_positions,
            primal_faces=tuple(face.vertex_indices for face in primal.faces),
            face_direction_indices=model.face_direction_indices,
            degenerate_primal_vertex_indices=DualGraphData.great_circle_face_indices(dual.vertex_positions, dual.faces),
        )
        shell_volumes, _, _, shell_mapping = shell
        meshes = sample_volume_hex_meshes(shell_volumes, volume_state.preview_resolution)
        with self._transaction():
            volume_state.set_shell_geometry(shell)
            self.view.clear_volume_previews()
            self.view.show_volume_previews(meshes, layer=Layer.SHELL_VOLUMES, palette_index_by_volume=shell_mapping)
            self.apply_visibility()
        self.notify("ok", f"Shell: {len(shell_volumes)} volumes (thickness {volume_state.shell_thickness:.3f})")
        return True

    def generate_columns(self) -> bool:
        """Build columns on the shell faces of input directions only (not auxiliary)."""
        model = self.model
        volume_state = model.volume_state
        if volume_state.shell_geometry is None:
            return self._refuse("Generate shell first")
        _, patches, projection_directions, face_direction_indices = volume_state.shell_geometry
        input_count = model.input_direction_count()
        face_indices = tuple(face for face, direction in enumerate(face_direction_indices) if direction < input_count)
        palette_indices = tuple(face_direction_indices[face] for face in face_indices)
        columns = build_column_volumes(
            spherical_patches=patches, projection_directions=projection_directions, outer_radius=1.0,
            column_height=volume_state.column_height, face_indices=face_indices,
        )
        meshes = sample_volume_hex_meshes(columns, volume_state.preview_resolution)
        with self._transaction():
            volume_state.set_column_volumes(columns, direction_indices=palette_indices)
            self.view.show_volume_previews(meshes, layer=Layer.COLUMN_VOLUMES, palette_index_by_volume=palette_indices)
            self.apply_visibility()
        self.notify("ok", f"Columns: {len(columns)} volumes (height {volume_state.column_height:.3f})")
        return True

    def set_volume_parameters(self, thickness: float, height: float, resolution: int) -> None:
        """Apply edited parameters; geometry built at old values is cleared with its previews."""
        volume_state = self.model.volume_state
        with self._transaction():
            volume_state.update_preview_resolution(resolution)  # Applies to the next sampling; invalidates nothing.
            if volume_state.update_shell_thickness(thickness):
                self.view.clear_volume_previews()
            if volume_state.update_column_height(height):
                self.view.clear(Layer.COLUMN_VOLUMES)

    def clear_volumes(self) -> None:
        """Clear shell and dependent column results and their previews."""
        self.view.clear_volume_previews()
        self.model.volume_state.set_shell_geometry(None)

    def clear_columns(self) -> None:
        """Clear column volumes only."""
        self.view.clear(Layer.COLUMN_VOLUMES)
        self.model.volume_state.set_column_volumes(None)
