# src/ui/sqs_solver.py
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from assignments.candidate_graphs import CandidateDualGraphId, PlaneGraphClass
from assignments.config import DEFAULT_RANDOM_SEED, SearchConfig, SearchMode
from assignments.config import SolverConfig as EngineSolverConfig
from assignments.pipeline import solve_with_lazy_separation
from assignments.results import SearchResult


@dataclass
class SolverConfig:
    """Mutable session solver policy; solve_sqs works on a normalized copy."""

    time_limit: int = 300  # Seconds per topology (>= 1).
    max_candidate_graphs: int = 0  # 0: no cap on candidate graphs.
    early_stop_first_convex: bool = True
    cpsat_num_workers: int = 0  # 0: CP-SAT chooses.
    minimize_dual_edge_length: bool = True

    def __post_init__(self) -> None:
        self.time_limit = max(1, self.time_limit)
        self.max_candidate_graphs = max(0, self.max_candidate_graphs)
        self.cpsat_num_workers = max(0, self.cpsat_num_workers)

    def graph_class(self, num_directions: int) -> PlaneGraphClass:
        """Return the catalogue an unpinned search scans under this policy."""
        return PlaneGraphClass.for_first_convex(num_directions) if self.early_stop_first_convex else PlaneGraphClass.ALL


def solve_sqs(
    direction_vectors: np.ndarray, solver_config: SolverConfig, max_candidate_graphs: int | None = None,
    random_seed: int | None = None, minimize_dual_edge_length: bool | None = None, digon_free_only: bool = False,
    pinned_candidate: CandidateDualGraphId | None = None, *, cpsat_num_workers_override: int | None = None,
) -> SearchResult:
    """Run one realization search from session settings; a non-None override applies to this solve only."""
    config = replace(solver_config)  # Normalized copy; overrides never mutate the session settings.
    n = len(direction_vectors)
    # An explicit DIGON_FREE request raises instead of falling back to ALL.
    if pinned_candidate is None:
        digon_free = digon_free_only
    else:
        digon_free = pinned_candidate.graph_class is PlaneGraphClass.DIGON_FREE
    if digon_free and PlaneGraphClass.for_first_convex(n) is not PlaneGraphClass.DIGON_FREE:
        raise ValueError(f"DIGON_FREE candidate graph class is empty for n={n}")

    limit = config.max_candidate_graphs if max_candidate_graphs is None else max_candidate_graphs
    workers = config.cpsat_num_workers if cpsat_num_workers_override is None else max(0, cpsat_num_workers_override)
    minimize = config.minimize_dual_edge_length if minimize_dual_edge_length is None else minimize_dual_edge_length
    seed = DEFAULT_RANDOM_SEED if random_seed is None else random_seed
    engine_config = EngineSolverConfig(
        time_limit_seconds=config.time_limit, num_workers=workers, minimize_dual_edge_length=minimize, random_seed=seed,
    )

    # digon_free_only forces first-convex mode (DIGON_FREE catalogue); exhaustive mode scans ALL.
    first_convex = digon_free_only or config.early_stop_first_convex
    search_config = SearchConfig(
        mode=SearchMode.FIRST_CONVEX if first_convex else SearchMode.EXHAUSTIVE_TOPOLOGIES,
        # The limit applies to unpinned searches only; non-positive means uncapped.
        candidate_prefix_limit=limit if pinned_candidate is None and limit > 0 else None,
        pinned_candidate=pinned_candidate,
    )
    return solve_with_lazy_separation(direction_vectors, solver_config=engine_config, search_config=search_config)
