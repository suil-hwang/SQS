# src/assignments/formulation.py
from __future__ import annotations

from collections.abc import Sequence
from itertools import chain

from ortools.sat.python import cp_model

from .candidate_arcs import CandidateArcs


class AssignmentFormulation:
    """CP-SAT formulation for a crossing-free assignment on a candidate dual graph.

    Two binary variable families:
    1. x[j][i] = 1 iff dual vertex j is assigned to input direction p_i.
    2. y[e][ell] = 1 iff underlying dual edge e is realized by allowed candidate arc a_ell.
    """

    __slots__ = (
        "_model", "_candidate_arcs", "_underlying_edges", "_degrees",
        "_adjacency", "_minimize_dual_edge_length", "_x", "_model_built", "_added_nogood_patterns",
    )

    def __init__(
        self,
        model: cp_model.CpModel,
        candidate_arcs: CandidateArcs,
        *,
        underlying_edges: Sequence[tuple[int, int, int]],
        adjacency: Sequence[Sequence[int]] = (),
        minimize_dual_edge_length: bool = False,
    ) -> None:
        # Snapshot candidate graph data before lazy model construction.
        self._underlying_edges = tuple(underlying_edges)
        self._model = model
        self._candidate_arcs = candidate_arcs
        degrees = [0] * len(candidate_arcs.unit_directions)
        for first, second, _ in self._underlying_edges:
            degrees[first] += 1
            degrees[second] += 1
        self._degrees = tuple(degrees)
        self._adjacency = tuple(tuple(neighbors) for neighbors in adjacency)
        self._minimize_dual_edge_length = minimize_dual_edge_length
        self._x: tuple[tuple[cp_model.IntVar, ...], ...] = ()
        self._model_built = False
        self._added_nogood_patterns: set[tuple[tuple[int, int], ...]] = set()

    @property
    def assignment_variables(self) -> tuple[tuple[cp_model.IntVar, ...], ...]:
        """Return vertex-to-direction assignment variables."""
        return self._x

    def build(self) -> None:
        """Build the crossing-free assignment model."""
        if self._model_built:
            return

        # Create the vertex-to-direction assignment variables x.
        arcs, model = self._candidate_arcs, self._model
        num_vertices = len(arcs.unit_directions)
        new_bool_var = model.new_bool_var
        x = tuple(tuple(new_bool_var(f"x_{vertex}_{direction}") for direction in range(num_vertices)) for vertex in range(num_vertices))

        # Row and column ExactlyOne constraints make x a permutation matrix.
        for variables in chain(x, zip(*x, strict=True)):
            model.add_exactly_one(variables)
        self._x = x
        # Exclude directions with too few allowed arcs for the vertex's support degree.
        for vertex, degree in enumerate(self._degrees):
            for direction, incident_arc_indices in enumerate(arcs.allowed_arc_incidence):
                if degree > len(incident_arc_indices):
                    model.add(x[vertex][direction] == 0)

        # Link each support edge to one allowed arc through Eq. (4)'s four clauses.
        constraints = model.proto.constraints
        x_indices = tuple(tuple(var.index for var in row) for row in x)
        # Each row uses compact allowed-arc order; columns use global pair indices.
        y_rows: list[list[int]] = []
        y_columns: list[list[int]] = [[] for _ in range(arcs.num_candidate_pairs)]

        for edge_index, (j, k, _) in enumerate(self._underlying_edges):
            xj, xk = x_indices[j], x_indices[k]
            y_row: list[int] = []
            for ell, a, b in arcs.allowed_arcs:
                y = new_bool_var(f"y_{edge_index}_{ell}").index
                y_row.append(y)
                y_columns[ell].append(y)

                # Proto literals encode NOT(variable) as -index - 1.
                constraints.add().bool_or.literals.extend((xj[a], xj[b], -y - 1))
                constraints.add().bool_or.literals.extend((xk[a], xk[b], -y - 1))
                constraints.add().bool_or.literals.extend((-xj[a] - 1, -xk[b] - 1, y))
                constraints.add().bool_or.literals.extend((-xj[b] - 1, -xk[a] - 1, y))
            constraints.add().exactly_one.literals.extend(y_row)
            y_rows.append(y_row)

        # Conflicting arcs share one at-most-one constraint across all edges; isolated arcs get implied uniqueness.
        for first_arc, second_arc in arcs.conflicts:
            constraints.add().at_most_one.literals.extend(y_columns[first_arc] + y_columns[second_arc])
        for arc_index in arcs.isolated_arc_indices:
            constraints.add().at_most_one.literals.extend(y_columns[arc_index])

        # Only when every support degree is four, each direction uses four arcs.
        if all(degree == 4 for degree in self._degrees):
            for incident_arc_indices in arcs.allowed_arc_incidence:
                # Transpose arc columns to preserve edge-major variable order.
                incident_y_indices = list(chain.from_iterable(zip(*(y_columns[arc_index] for arc_index in incident_arc_indices), strict=True)))
                degree_equality = constraints.add().linear
                degree_equality.vars.extend(incident_y_indices)
                degree_equality.coeffs.extend([1] * len(incident_y_indices))
                degree_equality.domain.extend((4, 4))

        # Optionally minimize dual edge length, weighting each y variable by edge multiplicity times arc length.
        if self._minimize_dual_edge_length:
            variable_indices = list(chain.from_iterable(y_rows))
            # Replace any caller objective; with no y variables, leave it unset.
            model.clear_objective()
            if variable_indices:
                objective = model.proto.floating_point_objective
                objective.vars.extend(variable_indices)
                objective.coeffs.extend(
                    multiplicity * length for _, _, multiplicity in self._underlying_edges for length in arcs.allowed_arc_lengths
                )

        # Mark completion only after every construction step succeeds.
        self._model_built = True

    def add_nogood_cuts(
        self,
        assignment: tuple[int, ...],
        *,
        violating_vertex_indices: Sequence[int] | None = None,
    ) -> int:
        """Add no-good cuts and return the new clause count."""
        # An exact cut fixes the whole assignment; a local cut fixes one violating vertex's sorted closed support star.
        stars = (
            [range(len(assignment))] if violating_vertex_indices is None
            else [sorted({vertex, *self._adjacency[vertex]}) for vertex in violating_vertex_indices]
        )
        patterns = dict.fromkeys(tuple((member, assignment[member]) for member in star) for star in stars)
        fresh = [pattern for pattern in patterns if pattern not in self._added_nogood_patterns]
        if fresh:
            self.build()
        # Insert in first-occurrence order, recording each insertion so a retry adds only the remaining cuts.
        for pattern in fresh:
            literals = [-self._x[vertex][direction].index - 1 for vertex, direction in pattern]
            self._model.proto.constraints.add().bool_or.literals.extend(literals)
            self._added_nogood_patterns.add(pattern)
        return len(fresh)
