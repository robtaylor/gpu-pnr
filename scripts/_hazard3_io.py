"""Shared parsers + grid construction for the Hazard3 GF180 LibreLane fixture.

Used by spike_route_one_net.py and spike_route_many_nets.py. The fixture is the
pre-computed LibreLane run at ~/Code/Apitronix/hazard-test (see the
`hazard3_fixture` memory entry); see docs/phase32_spike.md for design notes.
"""

from __future__ import annotations

from pathlib import Path

import torch

GUIDE = Path(
    "/Users/roberttaylor/Code/Apitronix/hazard-test/hazard3/librelane/runs/"
    "RUN_2026-05-08_22-32-24/39-openroad-globalrouting/after_grt.guide"
)
LAYER_ORDER = ["Metal1", "Metal2", "Metal3", "Metal4", "Metal5"]
PITCH_DBU = 200  # gf180mcuD: 0.20um wire pitch, 1 DBU = 1 nm


def parse_guides(path: Path) -> dict[str, list[tuple[int, int, int, int, str]]]:
    """Read all nets from a LibreLane after_grt.guide file.

    Format: net name on its own line, '(' on the next, then one
    'xlo ylo xhi yhi LayerName' line per guide rectangle, terminated by ')'.
    """
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
    """Center of a guide rectangle, mapped to (layer, row, col)."""
    x0, y0, x1, y1, layer = rect
    cx = (x0 + x1) // 2
    cy = (y0 + y1) // 2
    return (
        LAYER_ORDER.index(layer),
        (cy - origin[1]) // PITCH_DBU,
        (cx - origin[0]) // PITCH_DBU,
    )
