# src/assignments/solver.py
from __future__ import annotations

from collections.abc import Sequence
from enum import Enum
from math import isfinite

from ortools.sat.python import cp_model
from pyplantri.plantri import PlaneDual

from sqs.structure import SQS

from .candidate_arcs import CandidateArcs
from .config import SolverConfig
from .formulation import AssignmentFormulation


class SolveFailure(Enum):
    """Non-assignment outcome of one solve attempt."""

    INFEASIBLE = "INFEASIBLE"
    LIMIT_REACHED = "LIMIT_REACHED"


class CPSATAssignmentSolver:
    """A mutable CP-SAT model repeatedly solved after lazy cut injection."""

    __slots__ = ("_candidate_arcs", "_plane_graph", "_support_rotations", "_model", "_solver", "_formulation")

    def __init__(
        self,
        candidate_arcs: CandidateArcs,
        plane_graph: PlaneDual,
        config: SolverConfig,
    ) -> None:
        """Initialize one solver from validated geometry and topology."""
        if not isinstance(candidate_arcs, CandidateArcs):
            raise TypeError("candidate_arcs must be a CandidateArcs")
        if not isinstance(config, SolverConfig):
            raise TypeError("config must be a SolverConfig")
        if type(plane_graph) is not PlaneDual:
            raise TypeError("plane_graph must be PlaneDual")

        num_dual_vertices = len(candidate_arcs.unit_directions)
        if num_dual_vertices != plane_graph.num_vertices:
            raise ValueError(f"direction count {num_dual_vertices} != topology vertices {plane_graph.num_vertices}")

        support_edges = tuple((u, v, multiplicity) for (u, v), multiplicity in plane_graph.edge_multiplicity.items())
        self._candidate_arcs = candidate_arcs
        self._plane_graph = plane_graph
        self._support_rotations = support_rotations = tuple(tuple(dict.fromkeys(rotation)) for rotation in plane_graph.embedding)
        self._model = cp_model.CpModel()
        self._solver = cp_model.CpSolver()

        # Configure CP-SAT; a zero configured time limit means unlimited.
        parameters = self._solver.parameters
        for name in (
            "num_workers", "random_seed", "cp_model_presolve", "cp_model_probing_level", "symmetry_level",
            "linearization_level", "cut_level", "relative_gap_limit", "absolute_gap_limit",
        ):
            setattr(parameters, name, getattr(config, name))
        parameters.max_time_in_seconds = float("inf") if config.time_limit_seconds == 0 else config.time_limit_seconds

        self._formulation = AssignmentFormulation(
            self._model,
            candidate_arcs,
            underlying_edges=support_edges,
            adjacency=support_rotations,
            minimize_dual_edge_length=config.minimize_dual_edge_length,
        )

    @property
    def candidate_arcs(self) -> CandidateArcs:
        """Return the exact candidate-arc snapshot owned by this model."""
        return self._candidate_arcs

    @property
    def support_rotations(self) -> tuple[tuple[int, ...], ...]:
        """Return the solver-owned candidate support rotations."""
        return self._support_rotations

    @property
    def graph_id(self) -> int:
        """Return the namespace-local candidate graph ID."""
        return self._plane_graph.graph_id

    def _build_structure(
        self,
        assignment: tuple[int, ...],
        *,
        reverse_rotation_system: bool,
    ) -> SQS:
        """Construct Q* and Q from the solver-owned inputs."""
        return SQS.construct(
            self._candidate_arcs.unit_directions,
            self._plane_graph,
            assignment,
            reverse_rotation_system=reverse_rotation_system,
        )

    def prepare(self) -> None:
        """Build the static model once without deciding feasibility."""
        self._formulation.build()

    def solve(
        self,
        *,
        max_time_seconds: float | None = None,
    ) -> tuple[int, ...] | SolveFailure:
        """Solve once; a zero one-call limit skips the CP-SAT backend."""
        if max_time_seconds is not None and (not isfinite(max_time_seconds) or max_time_seconds < 0):
            raise ValueError("max_time_seconds must be finite and >= 0")
        self.prepare()
        if max_time_seconds == 0:
            return SolveFailure.LIMIT_REACHED

        parameters = self._solver.parameters
        configured_time_limit = parameters.max_time_in_seconds
        if max_time_seconds is not None:
            parameters.max_time_in_seconds = max_time_seconds

        try:
            status = self._solver.solve(self._model)
        finally:
            parameters.max_time_in_seconds = configured_time_limit

        if status == cp_model.MODEL_INVALID:
            # CP-SAT reports the model-validation error as the solution info.
            raise RuntimeError(f"CP-SAT MODEL_INVALID: {self._solver.solution_info().strip()}")

        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            # Extract the vertex-to-direction assignment from x.
            assignment: list[int] = []
            for vertex, row in enumerate(self._formulation.assignment_variables):
                direction = next((d for d, variable in enumerate(row) if self._solver.boolean_value(variable)), None)
                if direction is None:
                    raise RuntimeError(f"x[{vertex}] has no selected direction")
                assignment.append(direction)
            return tuple(assignment)

        if status == cp_model.UNKNOWN:
            return SolveFailure.LIMIT_REACHED
        if status == cp_model.INFEASIBLE:
            return SolveFailure.INFEASIBLE
        raise RuntimeError(f"unexpected CP-SAT status: {status.name}")

    def add_nogood_cuts(
        self,
        assignment: tuple[int, ...],
        *,
        violating_vertex_indices: Sequence[int] | None = None,
    ) -> int:
        """Add exact or local no-good cuts and return the new clause count."""
        return self._formulation.add_nogood_cuts(assignment, violating_vertex_indices=violating_vertex_indices)
