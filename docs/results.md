# Phase 1 results

All numbers from `scripts/bench_scaling.py` and `scripts/demo_multinet.py` on an
Apple Silicon M-series with PyTorch 2.11 MPS. Seed 42, 5% obstacle density.

## Single-net SSSP scaling

| Size | Cells | Sweep (MPS) | Iters | ms/iter | Mcells/s | Dijkstra (CPU) | Speedup | Status |
|---|---|---|---|---|---|---|---|---|
| 256² | 65K | 41 ms | 24 | 1.72 | 1.6 | 45 ms | 1.09× | ✓ |
| 512² | 262K | 39 ms | 24 | 1.64 | 6.7 | 186 ms | 4.73× | ✓ |
| **1024²** | **1.05M** | **83 ms** | **40** | **2.06** | **12.7** | **784 ms** | **9.51×** | ✓ |
| 2048² | 4.19M | 381 ms | 64 | 5.95 | 11.0 | 3344 ms | 8.78× | ✓ |
| 4096² | 16.8M | 2596 ms | 104 | 24.96 | 6.5 | (skip) | — | ✗ inf |
| 8192² | 67.1M | 14821 ms | 120 | 123.51 | 4.5 | (skip) | — | ✗ inf |

### Three things this tells us

**1. Sweet spot is 1024² (~12.7 Mcells/s, 9.5× speedup).** Throughput peaks
there and falls off — almost certainly memory-bandwidth-bound past that point.
Each iter touches ~5 full-grid tensors; at 4M+ cells per tensor we're bouncing
the entire working set through M-series unified memory bandwidth (~200–400 GB/s)
repeatedly.

**2. Correctness fails above 2048² — the float32 INF_PROXY precision wall.**
At 4096²+, source-to-sink cost comes back as `inf` because cumsum magnitude ×
float32 ULP exceeds legit distance values, and the polluted-mask threshold
catches *real* distances. Documented in `sweep.py` docstring and noted in the
roadmap as the Phase 2 mask-based-obstacle target.

**3. Iteration counts scale ~linearly in N (24/24/40/64/104/120).**
Diameter-bounded as expected for 4-connected Bellman-Ford. ms/iter scales
~O(N²) — pure grid area. So total work is O(N³), and the GPU lead opens up
where parallel arithmetic beats per-cell sync overhead, then closes again
where memory bandwidth becomes the wall.

## Per-iter overhead progression

| Variant | 1024² ms/iter | Notes |
|---|---|---|
| Python-loop sweeps | ~6 | One MPS kernel per row/col = 1024 launches per sweep |
| Scan-based (cumsum+cummin) per-iter sync | 5.97 | `torch.equal` per iter forces CPU↔GPU pipeline flush |
| Scan-based + check_every=8 sync | 3.78 | Async pipelining across 8 iters between syncs |

37% per-iter improvement from removing the sync — and the gain widens with
grid size since each pipeline flush costs more on bigger tensors.

`torch.compile` adds another 10–20% on top (`inductor` slightly better than
`aot_eager` on MPS in 2.11) but is not yet wired into the production kernel.

## Multi-net sequential routing

`scripts/demo_multinet.py`, 50 random nets, 5% obstacles, seed 42:

| Grid | Routed | Per-routed-net | Total time |
|---|---|---|---|
| 256² | 23/50 | 24 ms | 0.55 s |
| 1024² | 23/50 | 145 ms | 3.34 s |

**Per-net cost** at 1024² is ~1.7× standalone single-net sweep (83 ms) — the
overhead is the per-net `sweep_sssp` invocation plus the path-marking loop.
Earlier this was 815 ms/net; the fix was running `backtrace` on a `.cpu()`
view to avoid per-cell `.item()` sync (cheap on Apple Silicon's unified
memory).

### Phase 2.1 endpoint reservation: a useful negative result

I expected pin reservation (mark all sources/sinks as obstacles up-front,
temporarily un-reserve a net's own pins while it routes) to recover the
27/50 failures by stopping early nets from running through later nets'
pins. **It did not.** Measurements at 256² with 50 nets, seed 42:

| nets | naive | reserved |
|---|---|---|
| 5 | 4/5 | 5/5 |
| 10 | 9/10 | 9/10 |
| 20 | 14/20 | 11/20 |
| 30 | 19/30 | 15/30 |
| 50 | 23/50 | 23/50 |
| 80 | 26/80 | 26/80 |

Reservation is a *correctness invariant* (no two distinct nets share a
wire — the naive version violated this by chance) but is **not** a
success-rate optimization on random workloads. Mechanism: reserving all
pins forces early nets to take longer paths around other-pin obstacles,
and those longer paths create more barriers that block later nets.

In-isolation control: every individual net is routable on the empty
grid (20/20). The failures are pure sequential-routing interference,
not anything about the kernel.

**Implication**: net ordering (Phase 2.2) and sweep-sharing /
ripup-and-reroute (Phase 2.3) are load-bearing for actual success rate,
not polish on top of reservation.

**The 23/50 success rate** is naive sequential routing on randomly-pre-chosen
endpoints. Smaller workloads succeed proportionally:

| Nets | Routed |
|---|---|
| 10 (seed 42) | 9/10 |
| 20 (seed 42) | 14/20 |
| 50 (seed 42) | 23/50 |
| 10 (seed 7) | 10/10 |

Phase 2 endpoint reservation should recover most of these.

## Surprises and learnings

- **Float32 + cumsum precision is the real ceiling** at this scale, not raw
  GPU throughput. The "obvious" `INF_PROXY = 1e10` was wrong by ~6 orders of
  magnitude; correct ceiling is `~4e6 / N` for unit-accurate distances.
- **Per-iter `torch.equal` sync is invisible until you measure it** — it
  doesn't show up as compute time but as pipeline-flush stalls.
- **`torch.compile` for MPS is cautiously useful** in PyTorch 2.11 — both
  `aot_eager` and `inductor` backends ran without crashing on the sweep
  kernel; gains modest (10–20%) because it can't fuse across PrimTorch ops
  that are already MPS-tuned.
- **Apple Silicon unified memory is genuinely a feature**, not just a
  marketing point. `.cpu()` is metadata-only — the backtrace fix wouldn't
  work nearly as cleanly on a discrete-GPU host.
