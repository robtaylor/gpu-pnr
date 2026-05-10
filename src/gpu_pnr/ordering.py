"""Net-ordering strategies for sequential routing.

The order in which nets are routed sequentially has a big effect on
overall success rate and total wirelength. Each strategy returns a
permutation of the input net list.
"""

from __future__ import annotations

Net = tuple[tuple[int, int], tuple[int, int]]


def _hpwl(net: Net) -> int:
    (sr, sc), (tr, tc) = net
    return abs(sr - tr) + abs(sc - tc)


def order_nets(nets: list[Net], strategy: str = "hpwl_asc") -> list[Net]:
    """Return a permuted copy of `nets` according to the chosen strategy.

    Strategies:
        identity:  preserve input order (no-op).
        hpwl_asc:  shortest HPWL first. Routes inflexible short nets while
                   space is available; long nets get the leftover room.
        hpwl_desc: longest HPWL first. Lays down the long-net "spine"
                   first; short nets fill in around them.
    """
    if strategy == "identity":
        return list(nets)
    if strategy == "hpwl_asc":
        return sorted(nets, key=_hpwl)
    if strategy == "hpwl_desc":
        return sorted(nets, key=_hpwl, reverse=True)
    raise ValueError(f"Unknown ordering strategy: {strategy!r}")
