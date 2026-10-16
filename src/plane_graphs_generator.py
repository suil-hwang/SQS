# src/plane_graphs_generator.py
from __future__ import annotations

import argparse
import re
import time
from collections.abc import Sequence
from operator import index
from pathlib import Path

from tqdm import tqdm

from assignments import candidate_graphs
from pyplantri import plane_graph_cache
from pyplantri.plantri import PlaneDual, PlantriEnumeration, PrimalMinimumDegree


def parse_n_range(n_arg: str) -> tuple[int, int]:
    """Parse one n value or an inclusive N-M range."""
    match = re.fullmatch(r"([0-9]+)(?:\s*-\s*([0-9]+))?", n_arg.strip())
    if not match:
        raise argparse.ArgumentTypeError(f"Invalid n range {n_arg!r}; expected N or N-M")

    start = int(match.group(1))
    end = int(match.group(2) or start)
    if start > end:
        raise argparse.ArgumentTypeError(f"Invalid n range {n_arg!r}: start exceeds end")
    return start, end


def sort_plane_graphs(
    plane_graphs: Sequence[PlaneDual],
) -> list[PlaneDual]:
    """Return the complete catalogue in deterministic search-priority order."""
    # FILTER guarantees one dual digon for each double edge.
    return sorted(
        plane_graphs,
        key=lambda graph: (
            (face_sizes := graph.face_size_sequence).count(2),
            face_sizes,
            graph.graph_id,
        ),
    )


def build_plane_graph_caches(
    n_range: tuple[int, int],
    *,
    overwrite: bool,
    digon_free_only: bool = False,
    project_root: Path | None = None,
    verify_existing: bool = False,
) -> None:
    """Generate caches for each n and print a summary on success."""
    if project_root is None:
        project_root = Path(__file__).resolve().parents[1]

    out_dir = project_root / "resources" / "plantri"
    plane_class = (
        candidate_graphs.PlaneGraphClass.DIGON_FREE if digon_free_only else candidate_graphs.PlaneGraphClass.ALL
    )
    expected_counts = plane_class.counts
    catalog_label = "digon-free" if digon_free_only else "full"

    # Check the whole range before creating or replacing files.
    start_n, end_n = map(index, n_range)
    min_n, max_n = min(expected_counts), max(expected_counts)
    if not min_n <= start_n <= end_n <= max_n:
        raise ValueError(f"unsupported {plane_class.value} n range {start_n}..{end_n} (supported n={min_n}..{max_n})")
    primal_minimum_degree = PrimalMinimumDegree.AT_LEAST_3 if digon_free_only else PrimalMinimumDegree.AT_LEAST_2
    graph_class: plane_graph_cache.CacheGraphClass = "simple_quartic" if digon_free_only else "quartic_multigraph"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[tuple[object, ...]] = []
    for n in tqdm(range(start_n, end_n + 1)):
        out_path = candidate_graphs.catalogue_path(n, plane_class, out_dir)
        rel_out_path = out_path.relative_to(project_root)

        # Existing cache: skip, optionally after a full validation against the independent count table.
        if out_path.exists() and not overwrite:
            graph_count: int | str = "-"
            action = "SKIP_UNVERIFIED"
            if verify_existing:
                cache = plane_graph_cache.open_graph_cache(out_path)
                for field, expected in (
                    ("dual_vertex_count", n),
                    ("graph_class", graph_class),
                ):
                    if (actual := getattr(cache.metadata, field)) != expected:
                        raise ValueError(f"cache: {field} {actual!r}!={expected!r} ({out_path})")
                expected_order = candidate_graphs.PLANE_GRAPH_STORAGE_ORDER_NAME
                if cache.storage_order != expected_order:
                    raise ValueError(f"cache: storage_order {cache.storage_order!r}!={expected_order!r} ({out_path})")
                graph_count = len(cache)
                if graph_count != (expected := expected_counts[n]):
                    raise ValueError(f"cache graph count {graph_count} != expected {expected} ({out_path})")
                cache.verify_integrity()
                action = "SKIP_VERIFIED"
            size_label = f"{rel_out_path} ({out_path.stat().st_size / (1024 * 1024):.2f} MiB)"
            rows.append((n, graph_count, "-", "-", "-", "-", size_label, action))
            continue

        expected_count = expected_counts[n]
        # One extra record detects surplus generator output.
        started_at = time.perf_counter()
        enum_result = PlantriEnumeration.enumerate(
            n, max_count=expected_count + 1, primal_minimum_degree=primal_minimum_degree,
        )
        graph_count = len(enum_result.graphs)
        if graph_count != expected_count:
            raise RuntimeError(f"{catalog_label} plane-graph count mismatch for n={n}: generated {graph_count}, expected {expected_count}")
        graphs_to_save = sort_plane_graphs(enum_result.graphs)
        full_count = candidate_graphs.PLANE_GRAPH_COUNT.get(n)
        graph_count_display = f"{graph_count}/{full_count}" if digon_free_only and full_count is not None else graph_count
        action = "OK+DIGON_FREE" if digon_free_only else "OK"

        t_save_start = time.perf_counter()
        plane_graph_cache.write_graph_cache(
            graphs_to_save, out_path, dual_vertex_count=n, graph_class=graph_class,
            compress_level=1,  # favor write speed for regenerable caches
            storage_order_name=candidate_graphs.PLANE_GRAPH_STORAGE_ORDER_NAME,
            index_mode="explicit",
        )
        finished_at = time.perf_counter()
        save_s = finished_at - t_save_start
        rows.append((
            n, graph_count_display, enum_result.time_to_first_record_s,
            enum_result.remaining_s, save_s, finished_at - started_at,
            f"{rel_out_path} ({out_path.stat().st_size / (1024 * 1024):.2f} MiB)", action,
        ))

    table = [[
        "n", "Graphs", "Time to first record (s)", "After first record (s)",
        "Save (s)", "Total (s)", "File", "Action",
    ]]
    table.extend([f"{value:.4f}" if isinstance(value, float) else str(value) for value in row] for row in rows)
    widths = [max(map(len, column)) for column in zip(*table, strict=True)]
    row_format = "| " + " | ".join(f"{{:<{width}}}" for width in widths) + " |"
    print(row_format.format(*table[0]))
    print("|-" + "-|-".join("-" * width for width in widths) + "-|")
    for row in table[1:]:
        print(row_format.format(*row))


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Generate resources/plantri/plane_graphs_nX caches via pyplantri.")
    parser.add_argument("--n", required=True, type=parse_n_range, help="n value or range: ALL 3-10, DIGON_FREE 3-20")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing cache files")
    parser.add_argument("--digon-free-only", action="store_true", help="Save only digon-free graphs to plane_graphs_nX_digonfree.cache (do not overwrite full cache filenames)")
    parser.add_argument("--verify-existing", action="store_true", help="Fully validate existing caches before skipping them")

    args = parser.parse_args()

    build_plane_graph_caches(
        args.n, overwrite=args.overwrite, digon_free_only=args.digon_free_only,
        project_root=project_root, verify_existing=args.verify_existing,
    )

# python src/plane_graphs_generator.py --n 3-10 --overwrite
# python src/plane_graphs_generator.py --n 3-20 --overwrite --digon-free-only
