"""Tests for sequential multi-net routing."""

from __future__ import annotations

import math

import torch

from gpu_pnr.router import route_nets


def _path_is_valid(path, w):
    if path is None:
        return False
    for (i1, j1), (i2, j2) in zip(path, path[1:]):
        if abs(i1 - i2) + abs(j1 - j2) != 1:
            return False
    for i, j in path:
        if math.isinf(float(w[i, j])):
            return False
    return True


def test_single_net_open_grid():
    w = torch.ones(10, 10)
    nets = [((0, 0), (9, 9))]
    results = route_nets(w, nets)
    assert len(results) == 1
    path = results[0].path
    assert path is not None
    assert path[0] == (0, 0)
    assert path[-1] == (9, 9)
    assert _path_is_valid(path, w)


def test_two_nets_no_overlap():
    w = torch.ones(10, 10)
    nets = [((0, 0), (0, 9)), ((9, 0), (9, 9))]
    results = route_nets(w, nets)
    p0, p1 = results[0].path, results[1].path
    assert p0 is not None and p1 is not None
    assert set(p0).isdisjoint(set(p1)), "second net used cells from first net's route"


def test_second_net_blocked_by_first():
    w = torch.ones(5, 5)
    w[2, 0:5] = math.inf
    w[2, 0] = 1.0
    w[2, 4] = 1.0
    nets = [((0, 0), (4, 0)), ((0, 4), (4, 4))]
    results = route_nets(w, nets)
    p0, p1 = results[0].path, results[1].path
    assert p0 is not None and p1 is not None
    assert set(p0).isdisjoint(set(p1))


def test_blocked_endpoint_returns_none():
    w = torch.ones(5, 5)
    w[2, 2] = math.inf
    nets = [((2, 2), (4, 4))]
    results = route_nets(w, nets)
    assert results[0].path is None


def test_endpoint_collision_blocks_second():
    w = torch.ones(5, 5)
    nets = [((0, 0), (4, 4)), ((4, 4), (0, 0))]
    results = route_nets(w, nets)
    assert results[0].routed
    assert results[1].path is None, "net with endpoint on prior net's pin should fail"


def test_unrouteable_returns_none_without_corrupting_state():
    w = torch.ones(5, 5)
    w[2, :] = math.inf
    nets = [((0, 0), (4, 4)), ((0, 1), (0, 3))]
    results = route_nets(w, nets)
    assert results[0].path is None
    assert results[1].routed
