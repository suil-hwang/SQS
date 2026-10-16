# src/assignments/candidate_graphs.py
from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from itertools import islice
from pathlib import Path
from types import MappingProxyType
from typing import Final, overload

from pyplantri import plane_graph_cache
from pyplantri.plantri import PlaneDual


PLANE_GRAPH_STORAGE_ORDER_NAME = "sqs_priority"

# Brinkmann et al. (2005), Table 1:
# q1(n + 2), indexed by dual order n = primal face count.
# Duals of all simple quadrangulations of the sphere:
# loop-free, 4-edge-connected quartic plane multigraphs.
# Supported ALL catalogues stop at n=10.
PLANE_GRAPH_COUNT: Final[Mapping[int, int]] = MappingProxyType({
    3: 1, 4: 2, 5: 3, 6: 9,
    7: 18, 8: 62, 9: 198, 10: 803,
})

# Brinkmann et al. (2005), Table 2 and the minimum-order bound:
# q2(n + 2), indexed by dual order n = primal face count.
# Duals of minimum-degree-3 simple quadrangulations:
# 4-edge-connected simple quartic plane graphs, hence digon-free.
DIGON_FREE_PLANE_GRAPH_COUNT: Final[Mapping[int, int]] = MappingProxyType({
    3: 0, 4: 0, 5: 0, 6: 1,
    7: 0, 8: 1, 9: 1, 10: 3,
    11: 3, 12: 12, 13: 19, 14: 64,
    15: 155, 16: 510, 17: 1_514, 18: 5_146,
    19: 16_966, 20: 58_782,
})

SUPPORTED_DIRECTION_COUNTS: Final = frozenset(PLANE_GRAPH_COUNT) | frozenset(DIGON_FREE_PLANE_GRAPH_COUNT)

_DEFAULT_PLANE_GRAPH_CACHE_DIR = Path(__file__).resolve().parents[2] / "resources" / "plantri"


class PlaneGraphClass(Enum):
    """Plane graph classes with independent Graph ID sequences."""

    ALL = "ALL"
    DIGON_FREE = "DIGON_FREE"

    @property
    def counts(self) -> Mapping[int, int]:
        """Published sizes of supported catalogues, indexed by direction count."""
        return DIGON_FREE_PLANE_GRAPH_COUNT if self is PlaneGraphClass.DIGON_FREE else PLANE_GRAPH_COUNT

    @classmethod
    def for_first_convex(cls, num_dual_vertices: int) -> PlaneGraphClass:
        """Return the graph class selected by the first-convex policy."""
        expected_count = DIGON_FREE_PLANE_GRAPH_COUNT.get(num_dual_vertices)
        if expected_count is None:
            raise ValueError(f"unsupported DIGON_FREE direction count n={num_dual_vertices}")
        return cls.DIGON_FREE if expected_count > 0 else cls.ALL


@dataclass(frozen=True, slots=True)
class CandidateDualGraphId:
    """A candidate dual graph qualified by direction count and graph class."""

    num_directions: int
    graph_class: PlaneGraphClass
    graph_id: int


@dataclass(frozen=True, slots=True)
class CandidateDualGraphs(Sequence[PlaneDual]):
    """A bounded lazy view of a candidate dual graph source."""

    _source: Sequence[PlaneDual]
    graph_class: PlaneGraphClass
    expected_count: int
    selected_count: int

    def __len__(self) -> int:
        """Return the selected graph count."""
        return self.selected_count

    def __iter__(self) -> Iterator[PlaneDual]:
        """Iterate lazily over the selected source prefix."""
        return islice(self._source, self.selected_count)

    @overload
    def __getitem__(self, index: int) -> PlaneDual: ...

    @overload
    def __getitem__(self, index: slice) -> list[PlaneDual]: ...

    def __getitem__(self, index: int | slice) -> PlaneDual | list[PlaneDual]:
        """Return selected graphs by index or slice."""
        # Indexing the selected range normalizes negative positions and raises IndexError beyond the prefix.
        if isinstance(index, slice):
            return [self._source[position] for position in range(self.selected_count)[index]]
        return self._source[range(self.selected_count)[index]]


def catalogue_path(num_dual_vertices: int, graph_class: PlaneGraphClass, directory: Path | None = None) -> Path:
    """Return the cache file of one graph class; neither existence nor contents are checked."""
    suffix = "_digonfree" if graph_class is PlaneGraphClass.DIGON_FREE else ""
    directory = _DEFAULT_PLANE_GRAPH_CACHE_DIR if directory is None else directory  # The default is read at call time.
    return directory / f"plane_graphs_n{num_dual_vertices}{suffix}.cache"


def load_candidate_graphs(
    num_dual_vertices: int,
    graph_class: PlaneGraphClass,
    candidate_prefix_limit: int | None = None,
    pinned_candidate: CandidateDualGraphId | None = None,
) -> CandidateDualGraphs:
    """Return a count-checked full catalogue, positive prefix, or one pinned graph."""
    # SearchConfig validates the prefix and pin options; a pin must still name this direction count and class.
    if pinned_candidate and (pinned_candidate.num_directions, pinned_candidate.graph_class) != (num_dual_vertices, graph_class):
        raise ValueError("pinned candidate does not match the requested graph scope")

    digon_free = graph_class is PlaneGraphClass.DIGON_FREE
    expected_count = graph_class.counts.get(num_dual_vertices)
    if expected_count is None:
        raise ValueError(
            f"unsupported {graph_class.value} direction count n={num_dual_vertices} "
            f"(supported n={min(graph_class.counts)}..{max(graph_class.counts)})"
        )
    if pinned_candidate is not None and not 0 <= pinned_candidate.graph_id < expected_count:
        raise ValueError(f"graph_id={pinned_candidate.graph_id} not found in {graph_class.value} n={num_dual_vertices}")
    if expected_count == 0:
        return CandidateDualGraphs((), graph_class, expected_count, 0)

    cache_path = catalogue_path(num_dual_vertices, graph_class)
    cache = plane_graph_cache.open_graph_cache(cache_path)
    metadata = cache.metadata
    for field, required in (
        ("dual_vertex_count", num_dual_vertices),
        ("graph_class", "simple_quartic" if digon_free else "quartic_multigraph"),
    ):
        actual = getattr(metadata, field)
        if actual != required:
            raise ValueError(f"cache: {field} {actual!r}!={required!r} ({cache_path})")
    expected_order = PLANE_GRAPH_STORAGE_ORDER_NAME
    if cache.storage_order != expected_order:
        raise ValueError(f"cache: storage_order {cache.storage_order!r}!={expected_order!r} ({cache_path})")

    if (cache_count := len(cache)) != expected_count:
        raise ValueError(f"{graph_class.value} count {cache_count} != {expected_count} for n={num_dual_vertices}")

    if pinned_candidate is not None:
        return CandidateDualGraphs((cache.get_by_graph_id(pinned_candidate.graph_id),), graph_class, expected_count, 1)

    selected_count = expected_count if candidate_prefix_limit is None else min(candidate_prefix_limit, expected_count)
    return CandidateDualGraphs(cache, graph_class, expected_count, selected_count)
