#!/usr/bin/env python3
"""Phase 3.2 multi-net spike: route N Hazard3 2-pin nets independently.

Each net gets its own (L, H, W) cost grid built from its own guide rectangles.
This isolates per-net behavior and avoids the cross-net interference the full
chip-scale router will eventually need to handle. The autotune picks a
SEG_BARRIER appropriate to each net's geometry.

Reports aggregate stats: routed-fraction, total wirelength, total vias,
per-net time. No comparison to TritonRoute yet -- that's separate work
(parsing the post-DR DEF for actual routes).

Run: uv run python scripts/spike_route_many_nets.py [N] [SEED]
  N defaults to 50, SEED defaults to 0.
"""

from __future__ import annotations

import random
import sys
import time

from _hazard3_io import (
    GUIDE,
    LAYER_ORDER,
    build_grid,
    parse_guides,
    rect_center_to_grid,
)
from gpu_pnr.router import route_nets_3d


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    random.seed(seed)

    print(f"Loading guides from {GUIDE.name}...")
    all_nets = parse_guides(GUIDE)
    two_pin = [
        (name, rects) for name, rects in all_nets.items()
        if sum(1 for r in rects if r[4] == "Metal1") == 2
    ]
    print(f"  {len(all_nets)} total nets, {len(two_pin)} are 2-pin")

    # Sort by total guide-rectangle count and pick the smallest N -- keeps the
    # spike fast and avoids degenerate cases where a single net's guide spans
    # most of the chip.
    two_pin.sort(key=lambda nr: len(nr[1]))
    sample = two_pin[:n]
    print(f"  picking smallest {len(sample)} 2-pin nets for the spike\n")

    via_cost = 5.0
    total_routed = 0
    total_wl = 0
    total_vias = 0
    total_time_ms = 0.0
    route_counts_by_layer = {layer: 0 for layer in LAYER_ORDER}
    failures: list[tuple[str, str]] = []

    for net_name, rects in sample:
        # Pre-routing setup can fail on malformed guides; the router itself
        # shouldn't ever raise for routable inputs (a None path is the
        # expected "failed" signal), so we deliberately don't wrap the
        # routing call in the same except -- a kernel exception during
        # routing is a real bug we want to see.
        try:
            w, origin = build_grid(rects)
            metal1 = [r for r in rects if r[4] == "Metal1"]
            source = rect_center_to_grid(metal1[0], origin)
            sink = rect_center_to_grid(metal1[1], origin)
        except (ValueError, IndexError) as e:
            failures.append((net_name, f"setup: {type(e).__name__}: {e}"))
            continue
        t0 = time.perf_counter()
        results = route_nets_3d(w, [(source, sink)], via_cost=via_cost)
        t1 = time.perf_counter()
        total_time_ms += (t1 - t0) * 1000
        res = results[0]
        if res.path is None:
            failures.append((net_name, "router returned None"))
            continue
        total_routed += 1
        total_wl += res.length
        via_count = sum(
            1 for (la, _, _), (lb, _, _) in zip(res.path, res.path[1:]) if la != lb
        )
        total_vias += via_count
        for lyr_idx in {p[0] for p in res.path}:
            route_counts_by_layer[LAYER_ORDER[lyr_idx]] += 1

    print(f"=== Aggregate over {len(sample)} nets ===")
    print(f"  routed: {total_routed} / {len(sample)} ({100 * total_routed / len(sample):.1f}%)")
    print(f"  total wirelength: {total_wl} cells")
    print(f"  total via transitions: {total_vias}")
    print(f"  avg per-net time: {total_time_ms / len(sample):.1f} ms")
    print(f"  total elapsed routing time: {total_time_ms / 1000:.2f} s")
    print()
    print("Layer occupancy (number of routed nets that used the layer):")
    for layer in LAYER_ORDER:
        print(f"  {layer}: {route_counts_by_layer[layer]}")

    if failures:
        print(f"\nFailures ({len(failures)}):")
        for name, reason in failures[:10]:
            print(f"  {name}: {reason}")
        if len(failures) > 10:
            print(f"  ... ({len(failures) - 10} more)")


if __name__ == "__main__":
    main()
