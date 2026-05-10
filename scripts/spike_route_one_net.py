#!/usr/bin/env python3
"""Phase 3.2 single-net spike: route one Hazard3 net end-to-end.

Pulls a 2-pin net from a real LibreLane GR guide (Hazard3 level_3 on
gf180mcuD), builds an (L, H, W) cost grid from the guide rectangles,
and runs sweep_sssp_3d / route_nets_3d on it. No LEF/DEF parsing yet --
pin coordinates come from the first/last Metal1 rectangle centers in
the routing order, which is good enough for a 2-pin net where the only
two Metal1 patches are the pins.

Source data: ~/Code/Apitronix/hazard-test/hazard3/librelane/runs/
              RUN_2026-05-08_22-32-24/39-openroad-globalrouting/after_grt.guide

Run: uv run python scripts/spike_route_one_net.py [NET_NAME]
"""

from __future__ import annotations

import sys

import torch

from _hazard3_io import (
    GUIDE,
    LAYER_ORDER,
    PITCH_DBU,
    build_grid,
    parse_guides,
    rect_center_to_grid,
)
from gpu_pnr.router import route_nets_3d
from gpu_pnr.sweep import backtrace_3d, sweep_sssp_3d


def main() -> None:
    net_name = sys.argv[1] if len(sys.argv) > 1 else "_00013_"
    # Optional SEG_BARRIER override; otherwise the kernel auto-tunes per call.
    seg_barrier_override = float(sys.argv[2]) if len(sys.argv) > 2 else None
    print(f"Loading guides from {GUIDE.name}...")
    nets = parse_guides(GUIDE)
    if net_name not in nets:
        print(f"net {net_name!r} not found", file=sys.stderr)
        sys.exit(1)
    rects = nets[net_name]
    print(f"\nNet {net_name}: {len(rects)} guide rectangles")
    for r in rects:
        print(f"  {r}")

    metal1_rects = [r for r in rects if r[4] == "Metal1"]
    if len(metal1_rects) != 2:
        print(
            f"\nThis spike expects exactly 2 Metal1 rectangles (2-pin nets); "
            f"got {len(metal1_rects)}.",
            file=sys.stderr,
        )
        sys.exit(1)

    w, origin = build_grid(rects)
    L, H, W = w.shape
    print(f"\nGrid: ({L}, {H}, {W}) at {PITCH_DBU} DBU/cell, origin {origin}")
    finite = (~torch.isinf(w)).sum().item()
    print(f"Routable cells: {finite} / {L * H * W} ({100 * finite / (L * H * W):.1f}%)")

    source = rect_center_to_grid(metal1_rects[0], origin)
    sink = rect_center_to_grid(metal1_rects[1], origin)
    print(f"\nSource (layer, row, col): {source}")
    print(f"Sink   (layer, row, col): {sink}")

    via_cost = 5.0  # gf180mcuD vias are several pitches' worth of wire
    print(f"via_cost = {via_cost}")
    if seg_barrier_override is not None:
        print(f"seg_barrier override: {seg_barrier_override}")
    else:
        print("seg_barrier: auto-tune")

    print("\nRunning sweep_sssp_3d...")
    d, iters = sweep_sssp_3d(
        w, source, via_cost=via_cost, max_iters=400,
        seg_barrier=seg_barrier_override,
    )
    finite_d = torch.isfinite(d[sink]).item()
    print(f"  iters: {iters}")
    print(f"  d[sink]: {d[sink].item() if finite_d else 'inf (UNREACHABLE)'}")

    if not finite_d:
        print("\nSink unreachable -- dumping a slice of d on Metal1 for debug:")
        print(d[0])
        sys.exit(2)

    path = backtrace_3d(d.cpu(), w.cpu(), source, sink, via_cost=via_cost)
    if path is None:
        print("\nbacktrace failed despite finite distance -- bug.")
        sys.exit(3)

    print(f"\nPath length: {len(path)} cells")
    layers_used = sorted({p[0] for p in path})
    print(f"Layers used: {[LAYER_ORDER[lyr] for lyr in layers_used]}")
    via_count = sum(
        1 for (la, _, _), (lb, _, _) in zip(path, path[1:]) if la != lb
    )
    print(f"Via transitions: {via_count}")
    print("\nFirst 10 path cells (layer, row, col):")
    for p in path[:10]:
        print(f"  {p}")
    if len(path) > 10:
        print(f"  ... ({len(path) - 10} more)")

    print("\nSanity check via the multi-net router (single net)...")
    results = route_nets_3d(w, [(source, sink)], via_cost=via_cost)
    print(f"  routed: {results[0].routed}, length: {results[0].length}")


if __name__ == "__main__":
    main()
