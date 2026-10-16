from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite

from .candidate_graphs import CandidateDualGraphId

DEFAULT_RANDOM_SEED: int = 42
_INT32_MIN = -(1 << 31)
_INT32_MAX = (1 << 31) - 1
_MAX_CP_SAT_WORKERS = 10_000


@dataclass(frozen=True, slots=True, kw_only=True)
class SolverConfig:
    """Trusted configuration for the CP-SAT assignment solver."""

    # Finite and >= 0; topology wall budget and standalone backend default; 0=unlimited.
    time_limit_seconds: float = 300.0
    # In [0, 10_000]; 0 delegates automatic worker selection to OR-Tools.
    num_workers: int = 0

    # Signed int32 seed used by CP-SAT.
    random_seed: int = DEFAULT_RANDOM_SEED

    # Objective function: minimize multiplicity-weighted total dual edge arc length
    minimize_dual_edge_length: bool = True

    # Finite and >= 0; 0 disables relative-gap stopping; default permits a 1% gap.
    relative_gap_limit: float = 0.01
    # Finite and >= 0; 0 permits no positive absolute gap.
    absolute_gap_limit: float = 0.0

    # Whether CP-SAT presolve is enabled.
    cp_model_presolve: bool = True
    # In [0, 2**31 - 1]; 0 disables CP-SAT probing.
    cp_model_probing_level: int = 0
    # In [0, 2**31 - 1]; 0 disables CP-SAT symmetry processing.
    symmetry_level: int = 2
    # In [0, 2**31 - 1]; 0 disables CP-SAT linearization.
    linearization_level: int = 1
    # In [0, 2**31 - 1]; 0 disables CP-SAT cut generation.
    cut_level: int = 1

    def __post_init__(self) -> None:
        for field_name in (
            "time_limit_seconds",
            "relative_gap_limit",
            "absolute_gap_limit",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{field_name} must be a real number")
            try:
                valid = isfinite(value) and value >= 0
            except OverflowError:
                valid = False
            if not valid:
                raise ValueError(f"{field_name} must be finite and >= 0")

        for field_name, minimum, maximum in (
            ("num_workers", 0, _MAX_CP_SAT_WORKERS),
            ("cp_model_probing_level", 0, _INT32_MAX),
            ("symmetry_level", 0, _INT32_MAX),
            ("linearization_level", 0, _INT32_MAX),
            ("cut_level", 0, _INT32_MAX),
            ("random_seed", _INT32_MIN, _INT32_MAX),
        ):
            value = getattr(self, field_name)
            if type(value) is not int:
                raise TypeError(f"{field_name} must be an integer")
            if not minimum <= value <= maximum:
                raise ValueError(f"{field_name} must be an integer in [{minimum}, {maximum}]")

        for field_name in ("minimize_dual_edge_length", "cp_model_presolve"):
            if type(getattr(self, field_name)) is not bool:
                raise TypeError(f"{field_name} must be a bool")


class SearchMode(Enum):
    """Candidate-graph scope and stopping mode."""

    FIRST_CONVEX = "first_convex"
    FIRST_REALIZED = "first_realized"
    EXHAUSTIVE_TOPOLOGIES = "exhaustive_topologies"


@dataclass(frozen=True, slots=True, kw_only=True)
class SearchConfig:
    """Mode and finite-selection controls for candidate-graph search."""

    mode: SearchMode = SearchMode.FIRST_CONVEX
    candidate_prefix_limit: int | None = None
    pinned_candidate: CandidateDualGraphId | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.mode, SearchMode):
            raise TypeError(f"mode must be SearchMode, got {type(self.mode).__name__}")
        prefix_limit = self.candidate_prefix_limit
        if prefix_limit is not None:
            if type(prefix_limit) is not int:
                raise TypeError("candidate_prefix_limit must be an integer or None")
            if prefix_limit <= 0:
                raise ValueError("candidate_prefix_limit must be positive or None")
        pinned_candidate = self.pinned_candidate
        if pinned_candidate is not None and not isinstance(pinned_candidate, CandidateDualGraphId):
            raise TypeError("pinned_candidate must be a CandidateDualGraphId or None")
        if prefix_limit is not None and pinned_candidate is not None:
            raise ValueError("candidate_prefix_limit and pinned_candidate are mutually exclusive")
