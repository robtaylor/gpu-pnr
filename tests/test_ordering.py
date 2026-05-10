"""Tests for net-ordering strategies."""

from __future__ import annotations

import pytest

from gpu_pnr.ordering import _hpwl, order_nets


def test_identity_preserves_order():
    nets = [((0, 0), (1, 1)), ((5, 5), (6, 6)), ((2, 2), (4, 4))]
    assert order_nets(nets, "identity") == nets


def test_hpwl_ascending():
    nets = [((0, 0), (5, 5)), ((0, 0), (1, 0)), ((0, 0), (2, 2))]
    out = order_nets(nets, "hpwl_asc")
    hpwls = [_hpwl(n) for n in out]
    assert hpwls == sorted(hpwls)
    assert hpwls[0] == 1


def test_hpwl_descending():
    nets = [((0, 0), (5, 5)), ((0, 0), (1, 0)), ((0, 0), (2, 2))]
    out = order_nets(nets, "hpwl_desc")
    hpwls = [_hpwl(n) for n in out]
    assert hpwls == sorted(hpwls, reverse=True)
    assert hpwls[0] == 10


def test_unknown_strategy_raises():
    with pytest.raises(ValueError):
        order_nets([((0, 0), (1, 1))], "made_up")


def test_input_not_mutated():
    nets = [((0, 0), (5, 5)), ((0, 0), (1, 0))]
    snapshot = list(nets)
    _ = order_nets(nets, "hpwl_asc")
    _ = order_nets(nets, "hpwl_desc")
    assert nets == snapshot
