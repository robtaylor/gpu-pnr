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
from pathlib import Path

import torch

import gpu_pnr.sweep as sweep_module
from gpu_pnr.router import route_nets_3d
from gpu_pnr.sweep import backtrace_3d, sweep_sssp_3d

GUIDE = Path(
    "/Users/roberttaylor/Code/Apitronix/hazard-test/hazard3/librelane/runs/"
    "RUN_2026-05-08_22-32-24/39-openroad-globalrouting/after_grt.guide"
)
LAYER_ORDER = ["Metal1", "Metal2", "Metal3", "Metal4", "Metal5"]
PITCH_DBU = 200  # gf180mcuD: 0.20um wire pitch, 1 DBU = 1 nm


def parse_guides(path: Path) -> dict[str, list[tuple[int, int, int, int, str]]]:
    """Read all nets from a LibreLane after_grt.guide file."""
    nets: dict[str, list[tuple[int, int, int, int, str]]] = {}
    lines = path.read_text().splitlines()
    i = 0
    while i < len(lines):
        name = lines[i].strip()
        if i + 1 < len(lines) and lines[i + 1].strip() == "(":
            rects: list[tuple[int, int, int, int, str]] = []
            j = i + 2
            while j < len(lines) and lines[j].strip() != ")":
                parts = lines[j].split()
                if len(parts) == 5:
                    rects.append(
                        (int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3]), parts[4])
                    )
                j += 1
            nets[name] = rects
            i = j + 1
        else:
            i += 1
    return nets


def build_grid(
    rects: list[tuple[int, int, int, int, str]],
) -> tuple[torch.Tensor, tuple[int, int]]:
    """Convert per-layer guide rectangles to an (L, H, W) cost tensor.

    Cells inside the union of guide rectangles for a layer get cost 1.0;
    cells outside get inf (obstacle). The grid origin is the bbox lower-left
    corner; quantization is PITCH_DBU per cell.

    Returns (w, (origin_x, origin_y)) where w has shape (5, H, W).
    """
    xlo = min(r[0] for r in rects)
    ylo = min(r[1] for r in rects)
    xhi = max(r[2] for r in rects)
    yhi = max(r[3] for r in rects)
    H = (yhi - ylo) // PITCH_DBU
    W = (xhi - xlo) // PITCH_DBU
    L = len(LAYER_ORDER)
    w = torch.full((L, H, W), float("inf"))
    for x0, y0, x1, y1, layer in rects:
        if layer not in LAYER_ORDER:
            continue
        lyr = LAYER_ORDER.index(layer)
        gx0 = (x0 - xlo) // PITCH_DBU
        gy0 = (y0 - ylo) // PITCH_DBU
        gx1 = (x1 - xlo) // PITCH_DBU
        gy1 = (y1 - ylo) // PITCH_DBU
        w[lyr, gy0:gy1, gx0:gx1] = 1.0
    return w, (xlo, ylo)


def rect_center_to_grid(
    rect: tuple[int, int, int, int, str], origin: tuple[int, int]
) -> tuple[int, int, int]:
    """Center of a Metal1 rectangle, mapped to (layer, row, col)."""
    x0, y0, x1, y1, layer = rect
    cx = (x0 + x1) // 2
    cy = (y0 + y1) // 2
    return (
        LAYER_ORDER.index(layer),
        (cy - origin[1]) // PITCH_DBU,
        (cx - origin[0]) // PITCH_DBU,
    )


def main() -> None:
    net_name = sys.argv[1] if len(sys.argv) > 1 else "_00013_"
    # SEG_BARRIER auto-tune is the right long-term fix (see docs/phase32_spike.md);
    # for now allow override here. Default 5e3 works for our spike workloads where
    # max_legit_distance is ~1500 and per-row obstacle counts approach 1000.
    seg_barrier_override = float(sys.argv[2]) if len(sys.argv) > 2 else 5e3
    sweep_module.SEG_BARRIER = seg_barrier_override
    sweep_module.MAX_LEGIT_DISTANCE = seg_barrier_override / 2
    print(
        f"SEG_BARRIER={sweep_module.SEG_BARRIER}  "
        f"MAX_LEGIT_DISTANCE={sweep_module.MAX_LEGIT_DISTANCE}"
    )
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

    print("\nRunning sweep_sssp_3d...")
    d, iters = sweep_sssp_3d(w, source, via_cost=via_cost, max_iters=400)
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
