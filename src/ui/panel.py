# src/ui/panel.py
from __future__ import annotations

import math
import os
from collections.abc import Callable, Sequence
from functools import partial
from typing import Final

import polyscope.imgui as psim

from assignments.candidate_graphs import (
    CandidateDualGraphId, SUPPORTED_DIRECTION_COUNTS, PlaneGraphClass, catalogue_path,
)
from assignments.results import SearchResult
from ui.controller import SQSController
from ui.messages import commit_caveats, graph_label, pinned_status
from ui.model import GraphCounts
from ui.state import NOTICE_TAGS
from ui.view import VISIBILITY_TOGGLES


NOTICE_COLORS: Final = {
    "ok": (0.45, 0.85, 0.45, 1.0), "info": (0.80, 0.80, 0.80, 1.0),
    "warn": (0.95, 0.75, 0.30, 1.0), "error": (1.00, 0.45, 0.45, 1.0),
}


class EditorPanel:
    """Collapsible-section ImGui panel; a long action runs only after its busy line has been drawn."""

    def __init__(self, controller: SQSController):
        self.ctrl = controller
        self._requested_input_count = 6
        self._pending: tuple[str, Callable[[], object], bool] | None = None
        self._pending_shown = False  # Whether a whole frame has drawn the pending action's busy line.
        self._status_height = 0.0  # The last frame's status lines and bottom padding, kept free below the sections.
        # Browse cursor (class, 1-based Graph ID); the committed graph is the controller's last result.
        self.cursor: tuple[PlaneGraphClass, int] = (PlaneGraphClass.ALL, 1)
        self._cursor_n: int | None = None
        self._synced_result: SearchResult | None = None
        self._catalogue_classes_by_n: dict[int, tuple[PlaneGraphClass, ...]] = {}

    def render(self) -> None:
        """Run ready actions, then draw scrollable sections above persistent status."""
        # Show the busy line for a whole frame before running an action, outside all widget scopes.
        if self._pending is not None and self._pending_shown:
            self.run_pending()
        elif self._pending is not None:
            self._pending_shown = True

        ctrl = self.ctrl
        # Validation and search policy apply even when their sections are closed.
        ctrl.run_action("Validation", ctrl.refresh_validation_if_needed)
        n, supported, available = self._solve_scope()

        # Reserve last frame's status height, keeping at least 120 pixels for sections.
        section_height = psim.GetIO().DisplaySize[1] - psim.GetCursorScreenPos()[1] - self._status_height
        psim.SetNextWindowSizeConstraints((-1.0, 0.0), (-1.0, max(120.0, section_height)))
        psim.BeginChild("##sections", (0.0, 0.0), psim.ImGuiChildFlags_AutoResizeY)
        if psim.CollapsingHeader("Directions###directions", psim.ImGuiTreeNodeFlags_DefaultOpen):
            self._draw_directions_section()
        if psim.CollapsingHeader("Solver###solver", psim.ImGuiTreeNodeFlags_DefaultOpen):
            self._draw_solver_policy_section(n, available)
            self._draw_build_section(n, supported, available)
            if supported:
                self._draw_graph_selection_section(n, available)
            self._draw_quick_stats_section()
        if psim.CollapsingHeader("Volume###volume"):
            self._draw_volume_section()
        if psim.CollapsingHeader("Visibility###visibility"):
            self._draw_visibility_section()
        if ctrl.log and psim.CollapsingHeader(f"Log ({len(ctrl.log)})###log"):
            psim.PushTextWrapPos(0.0)
            for level, text in reversed(ctrl.log):
                psim.TextColored(NOTICE_COLORS[level], f"{NOTICE_TAGS[level]} {text}")
            psim.PopTextWrapPos()
        psim.EndChild()

        # Status stays below the sections; reserve scrollbar width for stable wrapping.
        status_top, style = psim.GetCursorPosY(), psim.GetStyle()
        psim.PushTextWrapPos(psim.GetWindowWidth() - style.WindowPadding[0] - style.ScrollbarSize)
        psim.Separator()
        if self._pending is not None:
            label, _, solve = self._pending
            detail = (
                f" (time limit {ctrl.model.solver_config.time_limit} s per topology). The window does not respond; "
                "Ctrl-C in the terminal cancels after the current CP-SAT solve returns."
            ) if solve else ". The window does not respond until it finishes."
            psim.TextColored(NOTICE_COLORS["info"], f"[...] {label}{detail}")
        validation = ctrl.model.validation_state
        for name, status in (
            ("Dual", validation.is_dual_crossing_free), ("Primal", validation.is_primal_crossing_free),
        ):
            if status is False:
                psim.TextColored(NOTICE_COLORS["error"], f"[X] {name} proper crossings detected")
        result = ctrl.committed_result
        if result is not None and (caveats := commit_caveats(result)):
            psim.TextColored(NOTICE_COLORS["warn"], f"[!] Committed search: {'; '.join(caveats)}")
            if psim.IsItemHovered(psim.ImGuiHoveredFlags_AllowWhenDisabled):
                psim.SetTooltip(
                    "The priority result is the realization the search policy selects (first convex, first realized "
                    "or best-ranked); it is settled when no trial is unresolved and the scan is complete or its target "
                    "was reached."
                )
        if ctrl.log:
            level, text = ctrl.log[-1]
            psim.TextColored(NOTICE_COLORS[level], f"{NOTICE_TAGS[level]} {text}")
        psim.PopTextWrapPos()
        self._status_height = psim.GetCursorPosY() - status_top + 2 * style.WindowPadding[1]

    # --- Deferred actions and graph browsing (also the headless driving surface) ---

    def request(self, label: str, action: Callable[[], object], *, solve: bool = False) -> None:
        """Defer an action until a whole frame has drawn its busy line."""
        if self._pending is not None:
            self.ctrl.notify("warn", f"{label} ignored: {self._pending[0]} is pending")
            return
        self._pending, self._pending_shown = (label, action, solve), False

    def run_pending(self) -> None:
        """Run the deferred action, if any, through the controller's exception boundary."""
        if self._pending is not None:
            (label, action, _), self._pending = self._pending, None
            self.ctrl.run_action(label, action)

    def browse(self, n: int, delta: int) -> None:
        """Step the cursor (clamped) and request a build; a failed build keeps the step, so browsing never stalls."""
        graph_class, graph_id = self.cursor
        self.cursor = (graph_class, max(1, min(graph_class.counts.get(n, 0), graph_id + delta)))
        self.build_cursor(n)

    def build_cursor(self, n: int) -> None:
        """Request a pinned build of the topology under the cursor."""
        graph_class, graph_id = self.cursor
        ctrl, candidate = self.ctrl, CandidateDualGraphId(n, graph_class, graph_id - 1)
        self.request(f"Build {graph_label(n, graph_class, graph_id - 1)}", lambda: ctrl.build_structure("auto", candidate), solve=True)

    # --- Directions ---

    def _draw_directions_section(self) -> None:
        ctrl, model = self.ctrl, self.ctrl.model
        min_n, max_n = min(SUPPORTED_DIRECTION_COUNTS), max(SUPPORTED_DIRECTION_COUNTS)
        psim.SeparatorText("Input")
        psim.SetNextItemWidth(70.0)
        changed, count = psim.DragInt(
            f"Directions (n, {min_n}-{max_n})##direction_count",
            max(min_n, min(self._requested_input_count, max_n)), 1.0, min_n, max_n,
        )
        if changed:
            self._requested_input_count = max(min_n, min(max_n, int(count)))
        if psim.Button("Generate##generate_vectors"):
            count = self._requested_input_count
            self.request(f"Generate {count} directions", lambda: ctrl.generate_directions(count))
        psim.SameLine()
        psim.TextUnformatted("(Fibonacci points on the unit sphere)")

        psim.SeparatorText("Additional Directions")
        input_count = model.input_direction_count()
        max_additional = max(0, max_n - input_count)
        psim.SetNextItemWidth(70.0)
        changed, additional = psim.DragInt(
            "Additional Vectors##single_additional_directions",
            max(0, min(model.configured_auxiliary_direction_count, max_additional)), 1.0, 0, max_additional,
        )
        if changed:
            model.configured_auxiliary_direction_count = max(0, min(max_additional, int(additional)))
        if psim.IsItemHovered(psim.ImGuiHoveredFlags_AllowWhenDisabled):
            psim.SetTooltip("Auxiliary directions take part in the solve; columns use only the input directions.")
        add_reason = (
            "Generate direction vectors first" if model.dual.num_directions == 0
            else "Set Additional Vectors to a positive value first"
            if model.configured_auxiliary_direction_count <= 0 else None
        )
        psim.BeginDisabled(add_reason is not None)
        clicked = psim.Button("Add DVs##add_additional")
        psim.EndDisabled()
        if add_reason is not None and psim.IsItemHovered(psim.ImGuiHoveredFlags_AllowWhenDisabled):
            psim.SetTooltip(add_reason)
        if clicked and add_reason is None:
            self.request("Add additional directions", ctrl.activate_auxiliary_directions)
        psim.SameLine()
        reset_reason = None if model.prepared_directions is not None else "No additional directions are active"
        psim.BeginDisabled(reset_reason is not None)
        clicked = psim.Button("Reset DVs##reset_additional")
        psim.EndDisabled()
        if reset_reason is not None and psim.IsItemHovered(psim.ImGuiHoveredFlags_AllowWhenDisabled):
            psim.SetTooltip(reset_reason)
        if clicked and reset_reason is None:
            self.request("Reset additional directions", ctrl.deactivate_auxiliary_directions)
        psim.TextUnformatted(
            f"Next solve: n = {input_count} input + {model.configured_auxiliary_direction_count} additional"
            f" = {model.configured_effective_direction_count()}",
        )

    # --- Solver ---

    def _solve_scope(self) -> tuple[int, bool, tuple[PlaneGraphClass, ...]]:
        """Return the next solve's effective n, whether it is supported, and the classes whose catalogue file exists.

        Switches the session to first-convex search, with a notice, when exhaustive search lacks an ALL catalogue.
        """
        ctrl, model = self.ctrl, self.ctrl.model
        config = model.solver_config
        n = model.configured_effective_direction_count()
        supported = model.dual.num_directions > 0 and n in SUPPORTED_DIRECTION_COUNTS
        if supported and n not in self._catalogue_classes_by_n:
            self._catalogue_classes_by_n[n] = tuple(
                graph_class for graph_class in PlaneGraphClass
                if graph_class.counts.get(n, 0) > 0 and catalogue_path(n, graph_class).is_file()
            )
        available = self._catalogue_classes_by_n[n] if supported else ()
        # Exhaustive search scans ALL; without that catalogue, fall back to first-convex search.
        if supported and PlaneGraphClass.ALL not in available and not config.early_stop_first_convex:
            config.early_stop_first_convex = True
            ctrl.notify(
                "warn", f"Exhaustive search requires an available ALL catalogue (n={min(PlaneGraphClass.ALL.counts)}.."
                f"{max(PlaneGraphClass.ALL.counts)}); using first-convex search for n={n}",
            )
        return n, supported, available

    def _draw_solver_policy_section(self, n: int, available: Sequence[PlaneGraphClass]) -> None:
        config = self.ctrl.model.solver_config
        logical_cores = os.cpu_count()
        max_workers = logical_cores if logical_cores is not None and logical_cores >= 1 else 4
        psim.SeparatorText("Policy")
        psim.SetNextItemWidth(80.0)
        changed, limit = psim.DragInt(
            "Time limit per topology (s, wall)##solver_time", max(1, min(config.time_limit, 3600)), 1.0, 1, 3600,
        )
        if changed:
            config.time_limit = max(1, min(3600, int(limit)))
        psim.SetNextItemWidth(70.0)
        changed, workers = psim.DragInt(
            "Workers##cpsat_workers", max(0, min(config.cpsat_num_workers, max_workers)), 1.0, 0, max_workers,
        )
        if changed:
            config.cpsat_num_workers = max(0, min(max_workers, int(workers)))
        if psim.IsItemHovered(psim.ImGuiHoveredFlags_AllowWhenDisabled):
            psim.SetTooltip(
                "0 = automatic (OR-Tools picks). With more than one worker or a binding time limit, "
                "results can vary between runs; the seed is fixed at 42."
            )
        all_unavailable = (
            n in SUPPORTED_DIRECTION_COUNTS and n not in PlaneGraphClass.ALL.counts
        ) or (bool(available) and PlaneGraphClass.ALL not in available)
        psim.BeginDisabled(all_unavailable)
        _, config.early_stop_first_convex = psim.Checkbox("First-convex search##first_convex", config.early_stop_first_convex)
        psim.EndDisabled()
        if psim.IsItemHovered(psim.ImGuiHoveredFlags_AllowWhenDisabled):
            psim.SetTooltip("Stops at the first convex topology; scans DIGON_FREE when it is nonempty, otherwise ALL.")
        if all_unavailable:
            psim.BeginDisabled()
            psim.TextWrapped(
                f"(exhaustive search requires an available ALL catalogue, supported for "
                f"n={min(PlaneGraphClass.ALL.counts)}..{max(PlaneGraphClass.ALL.counts)})"
            )
            psim.EndDisabled()
        policy_class = config.graph_class(n) if n in SUPPORTED_DIRECTION_COUNTS else None
        class_size = 0 if policy_class is None else policy_class.counts.get(n, 0)
        max_prefix = max(1, class_size)
        psim.SetNextItemWidth(80.0)
        changed, prefix = psim.DragInt(
            "Candidate prefix (0 = all)##candidate_prefix",
            max(0, min(config.max_candidate_graphs, max_prefix)), 1.0, 0, max_prefix,
        )
        if changed:
            config.max_candidate_graphs = max(0, min(max_prefix, int(prefix)))
        if psim.IsItemHovered(psim.ImGuiHoveredFlags_AllowWhenDisabled):
            psim.SetTooltip("Scans only the first K topologies in catalogue order; a partial scan does not settle the class.")
        _, config.minimize_dual_edge_length = psim.Checkbox(
            "Minimize total dual edge arc length (multiplicity-weighted)##minimize_length", config.minimize_dual_edge_length,
        )
        if policy_class is not None:
            prefix = config.max_candidate_graphs
            scanned = class_size if prefix == 0 else min(prefix, class_size)
            psim.TextWrapped(f"Scope: {graph_label(n, policy_class)}, {scanned:,} of {class_size:,} topologies")

    def _draw_build_section(self, n: int, supported: bool, available: Sequence[PlaneGraphClass]) -> None:
        ctrl, model = self.ctrl, self.ctrl.model
        psim.SeparatorText("Build")
        policy_class = model.solver_config.graph_class(n) if supported else None
        min_n, max_n = min(SUPPORTED_DIRECTION_COUNTS), max(SUPPORTED_DIRECTION_COUNTS)
        if model.dual.num_directions == 0:
            reason = "Generate direction vectors first"
        elif policy_class is None:
            reason = f"Effective n={n} is unsupported (n={min_n}..{max_n})"
        elif n not in policy_class.counts:
            reason = f"{policy_class.value} supports n={min(policy_class.counts)}..{max(policy_class.counts)}"
        else:
            reason = f"No {policy_class.value} catalogue for n={n}" if policy_class not in available else None
        psim.BeginDisabled(reason is not None)
        clicked = psim.Button("Build Dual Graph##build_dual")
        psim.EndDisabled()
        if reason is not None and psim.IsItemHovered(psim.ImGuiHoveredFlags_AllowWhenDisabled):
            psim.SetTooltip(reason)
        if clicked and reason is None:
            self.request("Dual build", lambda: ctrl.build_structure("dual"), solve=True)
        psim.SameLine()
        result = ctrl.committed_result
        has_cached = result is not None and result.sqs is not None and bool(model.dual.faces)
        psim.BeginDisabled(not has_cached)
        clicked = psim.Button("Build Primal##build_primal")
        psim.EndDisabled()
        if not has_cached and psim.IsItemHovered(psim.ImGuiHoveredFlags_AllowWhenDisabled):
            psim.SetTooltip("Build the dual graph first")
        if clicked and has_cached:
            self.request("Primal build", lambda: ctrl.build_structure("primal"))
        psim.SameLine()
        psim.BeginDisabled(reason is not None)
        clicked = psim.Button("Auto Build (Dual + Primal)##auto_build")
        psim.EndDisabled()
        if reason is not None and psim.IsItemHovered(psim.ImGuiHoveredFlags_AllowWhenDisabled):
            psim.SetTooltip(reason)
        if clicked and reason is None:
            self.request("Auto build", lambda: ctrl.build_structure("auto"), solve=True)
        if reason is not None:
            psim.BeginDisabled()
            psim.TextWrapped(f"({reason})")
            psim.EndDisabled()

    def _draw_graph_selection_section(self, n: int, available: Sequence[PlaneGraphClass]) -> None:
        """Draw the Graph ID cursor controls; the cursor is separate from the committed graph."""
        psim.SeparatorText("Graph Selection")
        if not available:
            psim.BeginDisabled()
            psim.TextWrapped(f"No graph catalogue is present for n={n}")
            psim.EndDisabled()
            return
        self._sync_cursor(n, available)
        names = [graph_class.value for graph_class in available]
        psim.SetNextItemWidth(150.0)
        changed, index = psim.Combo("Class##graph_class", available.index(self.cursor[0]), names)
        if psim.IsItemHovered(psim.ImGuiHoveredFlags_AllowWhenDisabled):
            psim.SetTooltip("Classes with a positive expected count whose catalogue file is present (not verified here).")
        if changed:
            self.cursor = (available[index], 1)
        graph_class, graph_id = self.cursor
        count = graph_class.counts.get(n, 0)
        psim.SetNextItemWidth(80.0)
        changed, edited_id = psim.DragInt(
            "Graph##graph_id", max(1, min(graph_id, count)), 1.0, 1, count,
        )
        if changed:
            graph_id = max(1, min(count, int(edited_id)))
            self.cursor = (graph_class, graph_id)
        psim.SameLine()
        psim.TextUnformatted(f"/ {count:,}")
        psim.BeginDisabled(graph_id <= 1)
        clicked = psim.Button("< Prev##graph_prev")
        psim.EndDisabled()
        if graph_id <= 1 and psim.IsItemHovered(psim.ImGuiHoveredFlags_AllowWhenDisabled):
            psim.SetTooltip("First topology of the class")
        if clicked and graph_id > 1:
            self.browse(n, -1)
        psim.SameLine()
        if psim.Button(f"Build #{graph_id}##graph_build"):
            self.build_cursor(n)
        if psim.IsItemHovered(psim.ImGuiHoveredFlags_AllowWhenDisabled):
            psim.SetTooltip("Pinned build of this one topology (dual + primal); it makes no class-wide claim.")
        psim.SameLine()
        psim.BeginDisabled(graph_id >= count)
        clicked = psim.Button("Next >##graph_next")
        psim.EndDisabled()
        if graph_id >= count and psim.IsItemHovered(psim.ImGuiHoveredFlags_AllowWhenDisabled):
            psim.SetTooltip("Last topology of the class")
        if clicked and graph_id < count:
            self.browse(n, 1)

        result, attempt = self.ctrl.committed_result, self.ctrl.last_attempt
        selected = None if result is None else result.selected
        psim.TextUnformatted("Committed: none" if result is None or selected is None else (
            f"Committed: {graph_label(selected.sqs.dual.num_directions, result.graph_class, selected.graph_id)}"
        ))
        graph_class, graph_id = self.cursor  # Prev/Next above may have moved it.
        last_build = ""
        if attempt is not None and attempt.result is not result and (
            attempt.result.config.pinned_candidate == CandidateDualGraphId(n, graph_class, graph_id - 1)
        ):
            last_build = f" (last build: {pinned_status(attempt.result)})"
        psim.TextUnformatted(f"Cursor: #{graph_id}{last_build}")

    def _sync_cursor(self, n: int, available: Sequence[PlaneGraphClass]) -> None:
        """Reset the cursor when n changes; follow a newly committed graph of the same n."""
        if self._cursor_n != n or self.cursor[0] not in available:
            self._cursor_n = n
            preferred = self.ctrl.model.solver_config.graph_class(n)
            graph_class = preferred if preferred in available or not available else available[0]
            self.cursor = (graph_class, 1)
            self._synced_result = None  # Follow the committed graph again if it belongs to this n.
        result = self.ctrl.committed_result
        if result is not self._synced_result:
            self._synced_result = result
            selected = None if result is None else result.selected
            if result is not None and selected is not None and selected.sqs.dual.num_directions == n and result.graph_class in available:
                self.cursor = (result.graph_class, selected.graph_id + 1)

    def _draw_quick_stats_section(self) -> None:
        model = self.ctrl.model
        counts = GraphCounts.from_graphs(model.dual, model.primal)
        psim.SeparatorText("Quick Stats")
        psim.TextUnformatted(f"Displayed directions: {counts.dual_vertices} (input {model.input_direction_count()})")
        psim.TextUnformatted(f"Dual: V*={counts.dual_vertices}, E*={counts.dual_edges}, F*={counts.dual_faces}")
        psim.TextUnformatted(
            f"Primal: V={counts.primal_vertices}, E={counts.primal_edges}, F={counts.primal_faces}"
            if counts.primal_faces else "Primal: (not built)"
        )
        result = self.ctrl.committed_result
        selected = None if result is None else result.selected
        if selected is not None and model.primal is selected.sqs.primal:  # Only while its primal graph is displayed.
            quality = selected.quality
            psim.TextWrapped(
                f"Geometry (pipeline screening): {selected.geometry_status.value.lower()}, "
                f"angles {math.degrees(quality.min_angle):.1f}-{math.degrees(quality.max_angle):.1f} deg"
            )
        psim.BeginDisabled()
        psim.TextWrapped("Validation lines report screening and the aggregate audit; neither certifies full-domain or assembly validity.")
        psim.EndDisabled()

    # --- Volume and visibility ---

    def _draw_volume_section(self) -> None:
        ctrl, volume_state = self.ctrl, self.ctrl.model.volume_state
        psim.SeparatorText("Parameters")
        psim.SetNextItemWidth(90.0)
        changed, value = psim.DragFloat(
            "Shell Thickness##vol_shell_thick", volume_state.shell_thickness, 0.01, 0.01, 0.5, "%.3f",
        )
        thickness = max(0.01, min(0.5, float(value))) if changed else volume_state.shell_thickness
        psim.SetNextItemWidth(90.0)
        changed, value = psim.DragFloat(
            "Column Height##vol_col_height", volume_state.column_height, 0.01, 0.01, 2.0, "%.3f",
        )
        height = max(0.01, min(2.0, float(value))) if changed else volume_state.column_height
        psim.SetNextItemWidth(70.0)
        changed, value = psim.DragInt(
            "Resolution##vol_res", max(3, min(volume_state.preview_resolution, 30)), 1.0, 3, 30,
        )
        resolution = max(3, min(30, int(value))) if changed else volume_state.preview_resolution
        if psim.IsItemHovered(psim.ImGuiHoveredFlags_AllowWhenDisabled):
            psim.SetTooltip("Preview sampling only: it applies to the next generation and keeps generated geometry.")
        if (thickness, height, resolution) != (
            volume_state.shell_thickness, volume_state.column_height, volume_state.preview_resolution,
        ):
            ctrl.run_action("Volume parameters", partial(ctrl.set_volume_parameters, thickness, height, resolution))
            volume_state = ctrl.model.volume_state  # A rolled-back edit replaces the state object.

        psim.SeparatorText("Actions")
        shell_reason = None if ctrl.model.primal.faces else "Build primal graph first"
        psim.BeginDisabled(shell_reason is not None)
        clicked = psim.Button("Generate Shell##gen_shell")
        psim.EndDisabled()
        if shell_reason is not None and psim.IsItemHovered(psim.ImGuiHoveredFlags_AllowWhenDisabled):
            psim.SetTooltip(shell_reason)
        if clicked and shell_reason is None:
            self.request("Generate shell", ctrl.generate_shell)
        psim.SameLine()
        has_volumes = volume_state.has_shell or volume_state.has_column_result
        psim.BeginDisabled(not has_volumes)
        clicked = psim.Button("Clear All##clear_all_volumes")
        psim.EndDisabled()
        if not has_volumes and psim.IsItemHovered(psim.ImGuiHoveredFlags_AllowWhenDisabled):
            psim.SetTooltip("No volumes")
        if clicked and has_volumes:
            ctrl.run_action("Clear volumes", ctrl.clear_volumes)
        if shell_reason is not None:
            psim.BeginDisabled()
            psim.TextWrapped(f"({shell_reason})")
            psim.EndDisabled()
        column_reason = None if volume_state.has_shell else "Generate shell first"
        psim.BeginDisabled(column_reason is not None)
        clicked = psim.Button("Generate Columns##gen_columns")
        psim.EndDisabled()
        if column_reason is not None and psim.IsItemHovered(psim.ImGuiHoveredFlags_AllowWhenDisabled):
            psim.SetTooltip(column_reason)
        if clicked and column_reason is None:
            self.request("Generate columns", ctrl.generate_columns)
        psim.SameLine()
        has_columns = volume_state.has_column_result
        psim.BeginDisabled(not has_columns)
        clicked = psim.Button("Clear Columns##clear_columns")
        psim.EndDisabled()
        if not has_columns and psim.IsItemHovered(psim.ImGuiHoveredFlags_AllowWhenDisabled):
            psim.SetTooltip("No columns")
        if clicked and has_columns:
            ctrl.run_action("Clear columns", ctrl.clear_columns)
        if column_reason is not None:
            psim.BeginDisabled()
            psim.TextWrapped(f"({column_reason})")
            psim.EndDisabled()

        psim.SeparatorText("Status")
        if volume_state.shell_geometry is not None:
            psim.TextUnformatted(
                f"Shell: {len(volume_state.shell_geometry[0])} volumes (thickness {volume_state.shell_thickness:.3f})"
            )
        if volume_state.column_volumes is not None:
            psim.TextUnformatted(
                f"Columns: {len(volume_state.column_volumes)} volumes"
                f" (height {volume_state.column_height:.3f}, input directions only)"
            )
        if volume_state.shell_geometry is None and volume_state.column_volumes is None:
            psim.TextUnformatted("No volumes generated")
        psim.TextUnformatted(
            f"Preview resolution setting: {volume_state.preview_resolution} (next generation or scene rebuild)"
        )
        psim.BeginDisabled()
        psim.TextWrapped("Sampled previews do not certify positive Jacobians or global injectivity.")
        psim.EndDisabled()

    def _draw_visibility_section(self) -> None:
        ctrl, layer_visibility = self.ctrl, self.ctrl.model.layer_visibility
        absent = []
        for label, flag, layers in VISIBILITY_TOGGLES:
            if not any(ctrl.view.has(layer) for layer in layers):
                absent.append(label)
                continue
            changed, visible = psim.Checkbox(f"{label}##visibility_{flag}", getattr(layer_visibility, flag))
            if changed:
                setattr(layer_visibility, flag, visible)
                ctrl.run_action("Visibility", ctrl.apply_visibility)
        if psim.Button("Reset visibility##reset_visibility"):
            layer_visibility.reset_visibility()
            ctrl.run_action("Visibility", ctrl.apply_visibility)
        if absent:
            psim.BeginDisabled()
            psim.TextWrapped(f"Not available yet: {', '.join(absent)}")
            psim.EndDisabled()
