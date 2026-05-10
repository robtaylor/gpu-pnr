"""Tests for the 3D multi-net router (route_nets_3d)."""

from __future__ import annotations

import math

import torch

from gpu_pnr.router import route_nets_3d


def _path_is_valid(path, w):
    if path is None:
        return False
    for (l1, i1, j1), (l2, i2, j2) in zip(path, path[1:]):
        in_layer = l1 == l2 and abs(i1 - i2) + abs(j1 - j2) == 1
        via = abs(l1 - l2) == 1 and i1 == i2 and j1 == j2
        if not (in_layer or via):
            return False
    for lyr, i, j in path:
        if math.isinf(float(w[lyr, i, j])):
            return False
    return True


def test_single_net_open_3d_grid():
    w = torch.ones(2, 8, 8)
    nets = [((0, 0, 0), (1, 7, 7))]
    results = route_nets_3d(w, nets, via_cost=1.0)
    assert len(results) == 1
    path = results[0].path
    assert path is not None
    assert path[0] == (0, 0, 0)
    assert path[-1] == (1, 7, 7)
    assert _path_is_valid(path, w)


def test_two_nets_disjoint_routes():
    w = torch.ones(2, 8, 8)
    nets = [
        ((0, 0, 0), (0, 0, 7)),
        ((0, 7, 0), (0, 7, 7)),
    ]
    results = route_nets_3d(w, nets, via_cost=10.0)
    p0, p1 = results[0].path, results[1].path
    assert p0 is not None and p1 is not None
    assert set(p0).isdisjoint(set(p1))


def test_second_net_uses_layer_above_to_bypass_first():
    """First net occupies a horizontal stripe on layer 0; second net's pins are
    on the opposite sides of that stripe and on layer 0. With via_cost low,
    the second net should detour up to layer 1 and back."""
    L, H, W = 2, 5, 5
    w = torch.ones(L, H, W)
    nets = [
        ((0, 2, 0), (0, 2, 4)),
        ((0, 0, 2), (0, 4, 2)),
    ]
    results = route_nets_3d(w, nets, via_cost=1.0)
    p0, p1 = results[0].path, results[1].path
    assert p0 is not None
    assert p1 is not None
    assert set(p0).isdisjoint(set(p1))
    layers_used_by_p1 = {p[0] for p in p1}
    assert 1 in layers_used_by_p1, "second net should detour through layer 1"


def test_blocked_endpoint_returns_none():
    w = torch.ones(2, 5, 5)
    w[0, 2, 2] = math.inf
    nets = [((0, 2, 2), (1, 4, 4))]
    results = route_nets_3d(w, nets, via_cost=1.0)
    assert results[0].path is None


def test_endpoint_collision_blocks_second():
    w = torch.ones(2, 5, 5)
    nets = [((0, 0, 0), (1, 4, 4)), ((1, 4, 4), (0, 0, 0))]
    results = route_nets_3d(w, nets, via_cost=1.0)
    assert results[0].routed
    assert results[1].path is None


def test_pin_reservation_protects_other_nets_pins_3d():
    """Same-layer-collision case: net A's pin sits on net B's natural path."""
    w = torch.ones(2, 5, 5)
    nets = [
        ((0, 0, 2), (0, 4, 2)),
        ((0, 2, 2), (0, 0, 4)),
    ]
    no_res = route_nets_3d(w, nets, via_cost=10.0, reserve_pins=False)
    with_res = route_nets_3d(w, nets, via_cost=10.0, reserve_pins=True)
    assert no_res[0].routed
    assert no_res[1].path is None, (
        "without reservation, first net consumes (0,2,2) which is second's source"
    )
    assert with_res[0].routed
    assert with_res[1].routed


def test_obstacle_layer_forces_via_detour():
    """Layer 0 is fully walled at row=2; pins are on layer 0 either side. Path
    must hop to layer 1 to cross."""
    H, W = 5, 5
    w0 = torch.ones(H, W)
    w0[2, :] = math.inf
    w1 = torch.ones(H, W)
    w = torch.stack([w0, w1], dim=0)
    nets = [((0, 0, 0), (0, 4, 4))]
    results = route_nets_3d(w, nets, via_cost=1.0)
    p0 = results[0].path
    assert p0 is not None
    assert _path_is_valid(p0, w)
    layers_used = {p[0] for p in p0}
    assert 1 in layers_used
