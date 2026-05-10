"""Multi-net demo: generate random nets on a synthetic grid, route sequentially."""

from __future__ import annotations

import argparse
import math
import random
import time

import torch

from gpu_pnr.router import route_nets


def make_grid(H: int, W: int, obstacle_frac: float, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    w = torch.ones(H, W)
    mask = torch.rand(H, W, generator=g) < obstacle_frac
    w[mask] = math.inf
    return w


def make_nets(
    w: torch.Tensor, n_nets: int, seed: int
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    rng = random.Random(seed)
    H, W = w.shape
    obstacle = torch.isinf(w)
    used: set[tuple[int, int]] = set()
    nets: list[tuple[tuple[int, int], tuple[int, int]]] = []
    attempts = 0
    while len(nets) < n_nets and attempts < n_nets * 100:
        attempts += 1
        s = (rng.randrange(H), rng.randrange(W))
        t = (rng.randrange(H), rng.randrange(W))
        if s == t or s in used or t in used:
            continue
        if bool(obstacle[s]) or bool(obstacle[t]):
            continue
        used.add(s)
        used.add(t)
        nets.append((s, t))
    return nets


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--nets", type=int, default=50)
    p.add_argument("--obstacles", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto")
    args = p.parse_args()

    if args.device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        device = args.device

    w = make_grid(args.size, args.size, args.obstacles, args.seed)
    nets = make_nets(w, args.nets, args.seed)
    print(
        f"Grid: {args.size}x{args.size}, obstacles={args.obstacles}, "
        f"nets={len(nets)}, device={device}"
    )

    w_dev = w.to(device)
    if device == "mps":
        torch.mps.synchronize()
    t0 = time.perf_counter()
    results = route_nets(w_dev, nets)
    if device == "mps":
        torch.mps.synchronize()
    elapsed = time.perf_counter() - t0

    routed = sum(1 for r in results if r.routed)
    failed = len(results) - routed
    total_wl = sum(r.length for r in results if r.routed)

    print(f"Routed: {routed}/{len(results)}   failed: {failed}")
    print(f"Total wirelength (routed): {total_wl} edges")
    print(
        f"Time: {elapsed*1000:7.1f} ms total   "
        f"{elapsed*1000/max(routed, 1):.1f} ms/routed-net"
    )


if __name__ == "__main__":
    main()
