# src/assignments/results.py
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from enum import Enum
from functools import partial
from types import MappingProxyType
from typing import cast

from sqs.evaluation import GeometryClass, SQSQuality
from sqs.structure import SQS

from .candidate_graphs import PlaneGraphClass
from .config import SearchConfig, SearchMode


_COUNT_FIELDS = ("status_counts", "geometry_counts", "late_exhaustion_counts")


class TrialStatus(Enum):
    """Termination state for one fixed candidate dual graph."""

    FOUND = "FOUND"
    TIMEOUT = "TIMEOUT"
    LIMIT_REACHED = "LIMIT_REACHED"
    INFEASIBLE = "INFEASIBLE"
    GEOMETRY_DEGENERATE = "GEOMETRY_DEGENERATE"
    STRUCTURE_MISMATCH = "STRUCTURE_MISMATCH"


class SearchOutcome(Enum):
    """Pipeline-valid SQS discovery within the recorded graph class."""

    SOLVED = "SOLVED"
    NO_VALID_REALIZATION = "NO_VALID_REALIZATION"
    SEARCH_INCONCLUSIVE = "SEARCH_INCONCLUSIVE"


@dataclass(frozen=True, slots=True, eq=False)
class SQSRealization:
    """A pipeline-accepted SQS with its source assignment and quality."""

    sqs: SQS
    graph_id: int
    assignment: tuple[int, ...]
    quality: SQSQuality

    @property
    def geometry_status(self) -> GeometryClass:
        """Return the classification supplied by pipeline screening."""
        return cast(GeometryClass, self.quality.geometry_class)


@dataclass(frozen=True, slots=True, eq=False)
class TrialResult:
    """Operational termination and optional realization for one candidate-graph trial."""

    status: TrialStatus
    realization: SQSRealization | None = None
    late_exhaustion_status: TrialStatus | None = None


@dataclass(frozen=True, slots=True, eq=False, kw_only=True)
class SearchResult:
    """Primitive evidence of one candidate-graph search."""

    graph_class: PlaneGraphClass
    config: SearchConfig
    selection: SQSRealization | None
    expected_count: int
    status_counts: Mapping[TrialStatus, int]
    geometry_counts: Mapping[GeometryClass, int]
    late_exhaustion_counts: Mapping[TrialStatus, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in _COUNT_FIELDS:
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))

    def __reduce__(self) -> tuple[partial[SearchResult], tuple[()]]:
        """Reconstruct read-only count snapshots across process boundaries."""
        state = {item.name: getattr(self, item.name) for item in fields(self) if item.init}
        state.update((name, dict(state[name])) for name in _COUNT_FIELDS)
        return partial(type(self), **state), ()

    @property
    def mode(self) -> SearchMode:
        """Return the configured search mode."""
        return self.config.mode

    @property
    def candidate_count(self) -> int:
        """Return the number of candidates admitted into this search."""
        if self.config.pinned_candidate is not None:
            return 1
        limit = self.config.candidate_prefix_limit
        return self.expected_count if limit is None else min(limit, self.expected_count)

    @property
    def unresolved_count(self) -> int:
        """Return the number of trials stopped without a proof."""
        return self.status_count(TrialStatus.TIMEOUT) + self.status_count(TrialStatus.LIMIT_REACHED)

    @property
    def target_reached(self) -> bool:
        """Whether the selected realization satisfies the configured target."""
        selected = self.selected
        return selected is not None and (self.mode is not SearchMode.FIRST_CONVEX or selected.geometry_status is GeometryClass.CONVEX)

    @property
    def priority_result_conclusive(self) -> bool:
        """Whether the priority result is settled for this search policy."""
        return not self.unresolved_count and (
            self.scan_complete
            or (self.mode is not SearchMode.EXHAUSTIVE_TOPOLOGIES and self.config.pinned_candidate is None and self.target_reached)
        )

    @property
    def selected(self) -> SQSRealization | None:
        """Return the selected realization, if any."""
        return self.selection

    @property
    def outcome(self) -> SearchOutcome:
        """Report SQS discovery, separately from convexity and priority."""
        if self.has_solution:
            return SearchOutcome.SOLVED
        if self.scan_complete and self.unresolved_count == 0:
            return SearchOutcome.NO_VALID_REALIZATION
        return SearchOutcome.SEARCH_INCONCLUSIVE

    @property
    def scan_complete(self) -> bool:
        """Whether the verified graph class was fully evaluated."""
        return self.evaluated_count == self.candidate_count == self.expected_count

    @property
    def evaluated_count(self) -> int:
        """Return the number of completed candidate-graph trials."""
        return sum(self.status_counts.values())

    @property
    def retained_count(self) -> int:
        """Return the number of trials retaining selectable geometry."""
        return sum(self.geometry_counts.values())

    @property
    def has_solution(self) -> bool:
        """Whether the mode selected a pipeline-valid realization."""
        return self.selected is not None

    @property
    def sqs(self) -> SQS | None:
        """Return the selected SQS structure, if any."""
        selected = self.selected
        return None if selected is None else selected.sqs

    def status_count(self, status: TrialStatus) -> int:
        """Return the count for one trial status."""
        return self.status_counts.get(status, 0)

    def geometry_count(self, status: GeometryClass) -> int:
        """Return the count for one retained geometry status."""
        return self.geometry_counts.get(status, 0)
