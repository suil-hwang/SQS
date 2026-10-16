# src/ui/messages.py
from __future__ import annotations

from assignments.candidate_graphs import PlaneGraphClass
from assignments.results import SearchResult, TrialStatus


def graph_label(n: int, graph_class: PlaneGraphClass, graph_id: int | None = None) -> str:
    """Label a scope (n=8/DIGON_FREE) or one topology with its 1-based Graph ID (n=8/DIGON_FREE #3)."""
    return f"n={n}/{graph_class.value}" + ("" if graph_id is None else f" #{graph_id + 1}")


def pinned_status(result: SearchResult) -> str:
    """Describe a pinned search by its single topology, never by the class-scoped outcome."""
    if result.selected is not None:
        return "accepted"
    return "unresolved (timeout/limit)" if result.unresolved_count else "no accepted assignment"


def trial_summary(result: SearchResult) -> str:
    """Summarize evaluated topologies, then unresolved trials by timeout and limit."""
    text = f"{result.evaluated_count:,} of {result.expected_count:,} topologies evaluated"
    if unresolved := result.unresolved_count:
        text += (
            f", {unresolved} unresolved: timeout {result.status_count(TrialStatus.TIMEOUT)}, "
            f"limit {result.status_count(TrialStatus.LIMIT_REACHED)}"
        )
    return text


def commit_caveats(result: SearchResult) -> list[str]:
    """Return caveats of a committed search: convex target missed, priority result unsettled."""
    if result.config.pinned_candidate is not None:  # A pinned search makes no class-wide claim.
        return []
    caveats = [] if result.target_reached else ["convex target not reached"]
    if not result.priority_result_conclusive:
        caveats.append(f"priority result not settled ({trial_summary(result)})")
    return caveats
