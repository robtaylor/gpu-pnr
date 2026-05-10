"""Sequential multi-net routing on a 2D grid.

For each net in order: run sweep SSSP from its source, backtrace to its
sink, and mark every cell of the resulting path as an obstacle for all
subsequent nets. Endpoints (pins) become obstacles too -- other nets
can't route through them.
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


def _is_blocked(w: torch.Tensor, ij: tuple[int, int]) -> bool:
    return not torch.isfinite(w[ij]).item()


def route_nets(
    w: torch.Tensor,
    nets: list[tuple[tuple[int, int], tuple[int, int]]],
) -> list[NetResult]:
    """Route nets sequentially on a working copy of `w`.

    Args:
        w: (H, W) tensor of cell-entry costs. float('inf') for obstacles.
        nets: ordered list of (source, sink) pin pairs.

    Returns:
        List of NetResult in the same order as `nets`. A net with `path=None`
        either had a blocked endpoint or no feasible route given prior nets.
    """
    w_cur = w.clone()
    inf_val = torch.tensor(float("inf"), device=w.device, dtype=w.dtype)
    results: list[NetResult] = []

    for source, sink in nets:
        if _is_blocked(w_cur, source) or _is_blocked(w_cur, sink):
            results.append(NetResult(source, sink, None))
            continue

        d, _ = sweep_sssp(w_cur, source)
        path = backtrace(d.cpu(), w_cur.cpu(), source, sink)

        if path is not None:
            for i, j in path:
                w_cur[i, j] = inf_val

        results.append(NetResult(source, sink, path))

    return results
