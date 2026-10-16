# src/assignments/pipeline.py
from __future__ import annotations

import time
from collections import Counter
from collections.abc import Callable
from itertools import pairwise
import numpy as np

from sqs import validation
from sqs.evaluation import GeometryClass, SQSQuality

from .candidate_arcs import CandidateArcs
from .candidate_graphs import PlaneGraphClass, load_candidate_graphs
from .config import SearchConfig, SearchMode, SolverConfig
from .results import SQSRealization, SearchResult, TrialResult, TrialStatus
from .solver import CPSATAssignmentSolver, SolveFailure


def _separation_loop(
    solver: CPSATAssignmentSolver,
    *,
    deadline: float | None,
) -> TrialResult:
    """Solve, validate, and exclude each rejected assignment with a fresh cut."""
    clock = time.perf_counter

    # Build the static formulation inside the topology wall budget.
    started_at = clock()
    if deadline is not None and started_at >= deadline:
        return TrialResult(TrialStatus.TIMEOUT)
    solver.prepare()
    clockwise_ranks: list[list[int]] = solver.candidate_arcs.clockwise_ranks.tolist()

    # Exhaustion precedence:
    # STRUCTURE_MISMATCH > GEOMETRY_DEGENERATE > INFEASIBLE.
    exhaustion_status = TrialStatus.INFEASIBLE

    while True:
        started_at = clock()
        remaining = None if deadline is None else deadline - started_at
        if remaining is not None and remaining <= 0:
            return TrialResult(TrialStatus.TIMEOUT)

        solve_result = solver.solve(max_time_seconds=remaining)
        finished_at = clock()

        deadline_reached = deadline is not None and finished_at >= deadline
        if solve_result is SolveFailure.LIMIT_REACHED:
            return TrialResult(TrialStatus.TIMEOUT if deadline_reached else TrialStatus.LIMIT_REACHED)
        if solve_result is SolveFailure.INFEASIBLE:
            if deadline_reached:
                return TrialResult(TrialStatus.TIMEOUT, late_exhaustion_status=exhaustion_status)
            return TrialResult(exhaustion_status)
        assignment = solve_result

        # Condition (iii): all support stars must follow rho* or, globally, its reversal. The reversed rotation
        # embeds the mirror map, which lies in the same orientation-reversing class of G*_n (Sec. 3.2); only a
        # star matching neither orientation gets a local cut, and mixed orientations get an exact no-good.
        violating_vertices: list[int] = []
        forbids_reversal = requires_reversal = False
        for vertex, neighbors in enumerate(solver.support_rotations):
            if len(neighbors) < 3:
                continue
            rank_row = clockwise_ranks[assignment[vertex]]
            ranks = [rank_row[assignment[neighbor]] for neighbor in neighbors]
            descents = sum(a > b for a, b in pairwise((*ranks, ranks[0])))
            if descents == 1:
                forbids_reversal = True
            elif descents == len(ranks) - 1:
                requires_reversal = True
            else:
                violating_vertices.append(vertex)

        # Reconstruct only assignments admitting one global orientation.
        realization = sqs = None
        build_degenerate = False
        if not violating_vertices and not (forbids_reversal and requires_reversal):
            try:
                sqs = solver._build_structure(assignment, reverse_rotation_system=requires_reversal)
            except ValueError:
                build_degenerate = True

            # Screen the reconstruction: selectable quality and strict direction containment.
            if sqs is not None:
                primal = sqs.primal
                quality = SQSQuality.from_face_vertices(
                    [[primal.vertices[index].position for index in face.vertex_indices] for face in primal.faces],
                    total_primal_edge_arc_length=sum(edge.arc_length for edge in primal.edges),
                    primal_embedding_is_valid=validation.is_primal_embedding_valid(sqs),
                )
                if quality.is_selectable and validation.has_primal_face_direction_containment(sqs, validate_boundary=False):
                    realization = SQSRealization(sqs=sqs, graph_id=solver.graph_id, assignment=assignment, quality=quality)

        # Keep a late valid realization as TIMEOUT; an expired rejection adds no cut.
        expired = deadline is not None and clock() >= deadline
        if realization is not None or expired:
            return TrialResult(TrialStatus.TIMEOUT if expired else TrialStatus.FOUND, realization)

        # Classify the rejection, then emit fresh local or exact cuts.
        if not build_degenerate:
            exhaustion_status = TrialStatus.STRUCTURE_MISMATCH
        elif exhaustion_status is TrialStatus.INFEASIBLE:
            exhaustion_status = TrialStatus.GEOMETRY_DEGENERATE

        cuts_added = solver.add_nogood_cuts(assignment, violating_vertex_indices=violating_vertices or None)
        if violating_vertices:
            cut_context = "local cyclic-order"
        elif build_degenerate:
            cut_context = "geometry-degenerate assignment"
        else:
            cut_context = "exact assignment"
        if not cuts_added:
            raise RuntimeError(f"{cut_context}: rejected assignment produced no fresh nogood")


def realization_selection_key(
    realization: SQSRealization,
) -> tuple[int, int, float, float, float, float, int]:
    """Return the ascending key for best-realization selection."""
    quality = realization.quality
    return (
        -realization.geometry_status.preference_rank,
        realization.sqs.dual.num_double_support_edges,
        quality.max_deviation_from_right_angle,
        -quality.min_angle,
        quality.max_edge_length_ratio,
        quality.total_primal_edge_arc_length,
        realization.graph_id,
    )


def solve_with_lazy_separation(
    dual_vertices: np.ndarray,
    *,
    solver_config: SolverConfig | None = None,
    search_config: SearchConfig | None = None,
    on_trial: Callable[[TrialResult], None] | None = None,
) -> SearchResult:
    """Search a count-checked topology graph class with lazy separation."""
    search_config = search_config or SearchConfig()
    solver_config = SolverConfig() if solver_config is None else solver_config
    if not isinstance(solver_config, SolverConfig):
        raise TypeError("solver_config must be a SolverConfig")

    # Reuse assignment-independent arc geometry across all candidate topologies.
    candidate_arcs = CandidateArcs.from_directions(dual_vertices)
    num_directions = len(candidate_arcs.unit_directions)
    mode = search_config.mode
    pinned_candidate = search_config.pinned_candidate
    if pinned_candidate is not None:
        graph_class = pinned_candidate.graph_class
    elif mode is SearchMode.FIRST_CONVEX:
        graph_class = PlaneGraphClass.for_first_convex(num_directions)
    else:
        graph_class = PlaneGraphClass.ALL

    candidates = load_candidate_graphs(
        num_directions,
        graph_class,
        candidate_prefix_limit=search_config.candidate_prefix_limit,
        pinned_candidate=pinned_candidate,
    )

    status_counts: Counter[TrialStatus] = Counter()
    geometry_counts: Counter[GeometryClass] = Counter()
    late_exhaustion_counts: Counter[TrialStatus] = Counter()
    selection: SQSRealization | None = None
    best_key: tuple[int | float, ...] | None = None

    # Each topology contributes at most its first pipeline-valid assignment.
    time_limit_seconds = solver_config.time_limit_seconds
    for plane_graph in candidates:
        # Start each wall budget before topology context and solver construction.
        started_at = time.perf_counter()
        deadline = started_at + time_limit_seconds if time_limit_seconds > 0 else None
        result = _separation_loop(CPSATAssignmentSolver(candidate_arcs, plane_graph, solver_config), deadline=deadline)
        status_counts[result.status] += 1
        if result.late_exhaustion_status is not None:
            late_exhaustion_counts[result.late_exhaustion_status] += 1
        if on_trial is not None:
            on_trial(result)
        realization = result.realization
        if realization is None:
            continue

        geometry_status = realization.geometry_status
        geometry_counts[geometry_status] += 1
        if mode is SearchMode.FIRST_REALIZED or (mode is SearchMode.FIRST_CONVEX and geometry_status is GeometryClass.CONVEX):
            selection = realization
            break

        candidate_key: tuple[int | float, ...] = (
            (-geometry_status.preference_rank, realization.sqs.dual.num_double_support_edges, realization.quality.total_primal_edge_arc_length)
            if mode is SearchMode.FIRST_CONVEX
            else realization_selection_key(realization)
        )
        # Strict comparison keeps the earliest trial among equal keys.
        if best_key is None or candidate_key < best_key:
            best_key = candidate_key
            selection = realization

    return SearchResult(
        graph_class=candidates.graph_class,
        config=search_config,
        selection=selection,
        expected_count=candidates.expected_count,
        status_counts=status_counts,
        geometry_counts=geometry_counts,
        late_exhaustion_counts=late_exhaustion_counts,
    )
