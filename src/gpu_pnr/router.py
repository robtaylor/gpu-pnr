"""Sequential multi-net routing on a 2D grid.

Reserves all pin cells (sources + sinks of every net) as obstacles before
routing starts; temporarily un-reserves a net's own pins while it routes.
This stops earlier nets from running their wires through later nets'
pins, which would otherwise force the later nets to fail.

A net's path cells (including its endpoints, which become wires once
routed) are committed to a separate `routed_cells` set. Once a cell is
in that set we never un-reserve it -- two nets touching the same wire
would be an electrical short.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from gpu_pnr.sweep import backtrace, sweep_sssp


@dataclass
class NetResult:
    source: tuple[int, int]
    sink: tuple[int, int]
    path: list[tuple[int, int]] | None

    @property
    def routed(self) -> bool:
        return self.path is not None

    @property
    def length(self) -> int:
        return len(self.path) - 1 if self.path else 0


def _is_finite(w: torch.Tensor, ij: tuple[int, int]) -> bool:
    return bool(torch.isfinite(w[ij]).item())


def route_nets(
    w: torch.Tensor,
    nets: list[tuple[tuple[int, int], tuple[int, int]]],
    reserve_pins: bool = True,
) -> list[NetResult]:
    """Route nets sequentially on a working copy of `w`.

    Args:
        w: (H, W) tensor of cell-entry costs. float('inf') for obstacles.
        nets: ordered list of (source, sink) pin pairs.
        reserve_pins: if True (default), all pin cells are reserved as
            obstacles before routing begins; each net's own pins are
            temporarily un-reserved while it routes. Set False to get
            the naive baseline that ignores pin protection.

    Returns:
        List of NetResult in the same order as `nets`. A net with `path=None`
        either had an originally-blocked endpoint, an endpoint already
        committed to a prior net's wire, or no feasible route.
    """
    inf_val = torch.tensor(float("inf"), device=w.device, dtype=w.dtype)
    w_cur = w.clone()
    routed_cells: set[tuple[int, int]] = set()

    if reserve_pins:
        pin_cells: set[tuple[int, int]] = set()
        for s, t in nets:
            pin_cells.add(s)
            pin_cells.add(t)
        for ij in pin_cells:
            if _is_finite(w_cur, ij):
                w_cur[ij] = inf_val

    results: list[NetResult] = []
    for source, sink in nets:
        if source in routed_cells or sink in routed_cells:
            results.append(NetResult(source, sink, None))
            continue
        if not _is_finite(w, source) or not _is_finite(w, sink):
            results.append(NetResult(source, sink, None))
            continue

        if reserve_pins:
            w_cur[source] = w[source]
            w_cur[sink] = w[sink]

        d, _ = sweep_sssp(w_cur, source)
        path = backtrace(d.cpu(), w_cur.cpu(), source, sink)

        if path is not None:
            for ij in path:
                w_cur[ij] = inf_val
                routed_cells.add(ij)
        elif reserve_pins:
            w_cur[source] = inf_val
            w_cur[sink] = inf_val

        results.append(NetResult(source, sink, path))

    return results
