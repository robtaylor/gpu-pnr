#!/usr/bin/env python3
"""Phase 3.2 multi-net spike: route N Hazard3 2-pin nets independently.

Each net gets its own (L, H, W) cost grid built from its own guide rectangles.
This isolates per-net behavior and avoids the cross-net interference the full
chip-scale router will eventually need to handle. The autotune picks a
SEG_BARRIER appropriate to each net's geometry.

Reports aggregate stats including a comparison to TritonRoute's wire and
via counts (parsed from final/def/...).

The optional `m1_cost` argument multiplies the cost of every Metal1 wire
cell by that penalty. With m1_cost >> 1 the router avoids using M1 as
through-wire and instead via-stacks from each pin up to M2+ for the wire
body -- approximating gf180mcuD's pin-access-only convention for M1.

Run: uv run python scripts/spike_route_many_nets.py [N] [SEED] [M1_COST]
  N defaults to 50, SEED defaults to 0, M1_COST defaults to 1.0.
"""

from __future__ import annotations

import random
import sys
import time

import torch

from _hazard3_io import (
    FINAL_DEF,
    GUIDE,
    LAYER_ORDER,
    PITCH_DBU,
    build_grid,
    parse_def_nets,
    parse_guides,
    rect_center_to_grid,
)
from gpu_pnr.router import route_nets_3d


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    m1_cost = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
    random.seed(seed)
    if m1_cost != 1.0:
        print(f"M1 wire-cost penalty: {m1_cost}x  (pin-access-only approximation)")

    print(f"Loading guides from {GUIDE.name}...")
    all_nets = parse_guides(GUIDE)
    two_pin = [
        (name, rects) for name, rects in all_nets.items()
        if sum(1 for r in rects if r[4] == "Metal1") == 2
    ]
    print(f"  {len(all_nets)} total nets, {len(two_pin)} are 2-pin")

    print(f"Loading TritonRoute output from {FINAL_DEF.name}...")
    triton = parse_def_nets(FINAL_DEF)
    print(f"  {len(triton)} TritonRoute-routed nets")

    # Sort by total guide-rectangle count and pick the smallest N -- keeps the
    # spike fast and avoids degenerate cases where a single net's guide spans
    # most of the chip.
    two_pin.sort(key=lambda nr: len(nr[1]))
    sample = two_pin[:n]
    print(f"  picking smallest {len(sample)} 2-pin nets for the spike\n")

    via_cost = 5.0
    total_routed = 0
    total_wl_cells = 0
    total_vias = 0
    total_time_ms = 0.0
    route_counts_by_layer = {layer: 0 for layer in LAYER_ORDER}
    failures: list[tuple[str, str]] = []

    # TritonRoute aggregate over the same nets we routed (cells, not DBU).
    triton_total_wl_cells = 0
    triton_total_vias = 0
    triton_missing = 0

    for net_name, rects in sample:
        # Pre-routing setup can fail on malformed guides; the router itself
        # shouldn't ever raise for routable inputs (a None path is the
        # expected "failed" signal), so we deliberately don't wrap the
        # routing call in the same except -- a kernel exception during
        # routing is a real bug we want to see.
        try:
            w, origin = build_grid(rects)
            if m1_cost != 1.0:
                # Apply pin-access-only penalty on M1 wire cells. Source/sink
                # land on M1 too, but Phase 3.4's edge model doesn't charge
                # w[dest] on a via arrival, so via-stacking off M1 to M2+ for
                # the wire body and back down at the sink is the natural
                # behavior with this penalty.
                m1 = w[0]
                w[0] = torch.where(torch.isinf(m1), m1, m1 * m1_cost)
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
        total_wl_cells += res.length
        via_count = sum(
            1 for (la, _, _), (lb, _, _) in zip(res.path, res.path[1:]) if la != lb
        )
        total_vias += via_count
        for lyr_idx in {p[0] for p in res.path}:
            route_counts_by_layer[LAYER_ORDER[lyr_idx]] += 1
        if net_name in triton:
            triton_wl_dbu, triton_vc = triton[net_name]
            # Assumes uniform 200nm pitch across all layers (gf180mcuD); a
            # per-layer pitch table would be needed for non-isotropic PDKs.
            triton_total_wl_cells += triton_wl_dbu // PITCH_DBU
            triton_total_vias += triton_vc
        else:
            triton_missing += 1

    print(f"=== Aggregate over {len(sample)} nets ===")
    print(f"  routed: {total_routed} / {len(sample)} ({100 * total_routed / len(sample):.1f}%)")
    print(f"  total wirelength: {total_wl_cells} cells")
    print(f"  total via transitions: {total_vias}")
    print(f"  avg per-net time: {total_time_ms / len(sample):.1f} ms")
    print(f"  total elapsed routing time: {total_time_ms / 1000:.2f} s")
    print()
    print("Layer occupancy (number of routed nets that used the layer):")
    for layer in LAYER_ORDER:
        print(f"  {layer}: {route_counts_by_layer[layer]}")

    # Restrict comparison to nets we actually routed AND TritonRoute also has.
    if total_routed > 0:
        matched = total_routed - triton_missing
        print()
        print(f"=== TritonRoute comparison (over {matched} matched nets) ===")
        print(f"  TritonRoute total wirelength: {triton_total_wl_cells} cells")
        print(f"  TritonRoute total vias:       {triton_total_vias}")
        wl_ratio = (
            f"{total_wl_cells / triton_total_wl_cells:.2f}x"
            if triton_total_wl_cells else "n/a"
        )
        via_ratio = (
            f"{total_vias / triton_total_vias:.2f}x"
            if triton_total_vias else "n/a"
        )
        print(f"  ours / TritonRoute wirelength: {wl_ratio}")
        print(f"  ours / TritonRoute vias:       {via_ratio}")
        if triton_missing:
            print(f"  ({triton_missing} nets we routed had no entry in TritonRoute output -- skipped)")

    if failures:
        print(f"\nFailures ({len(failures)}):")
        for name, reason in failures[:10]:
            print(f"  {name}: {reason}")
        if len(failures) > 10:
            print(f"  ... ({len(failures) - 10} more)")


if __name__ == "__main__":
    main()
