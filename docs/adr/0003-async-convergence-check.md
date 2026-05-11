# ADR 0003 — Async convergence check via `check_every`

**Status:** Accepted (2026-05-09).

## Context

The sweep iterates until the distance tensor `d` stops changing. The natural
check is `torch.equal(d, d_prev)` after each outer iteration. On MPS,
`torch.equal` forces a CPU↔GPU sync (its result is a Python `bool`), which
flushes the device's pipeline.

Measured cost at 1024²:

| Variant | ms/iter |
|---|---|
| Scan-based, `torch.equal` every iter | 5.97 |
| Scan-based, `torch.equal` every 8th iter | 3.78 |

A 37% per-iter speedup just from removing 7 of every 8 syncs. The gain widens
at larger grids because each pipeline flush costs more on bigger tensors.

The cost is detection latency: we may run up to `check_every-1` extra iterations
past the true fixed point. For the kernel's normal convergence behaviour
(monotonically tightening distances) those iterations are no-ops on the result
tensor, so the only cost is wall-clock — not correctness.

## Decision

Add a `check_every: int = 8` parameter to `sweep_sssp` (and its multi-source / 3D
siblings) that controls how often the convergence test runs. Default to 8;
expose for benchmarking.

## Consequences

- 37% per-iter improvement at 1024², larger gains at bigger grids.
- Up to 7 redundant iterations on tightly-converging inputs — negligible
  relative to total iteration count, which scales with grid diameter (24–192
  iters across 256²–8192²).
- `_converge_or_max` reuses `d_check`'s storage via in-place `copy_` rather
  than cloning per check — small but real win at very large grids.

## Walk-back options

- **If a future workload has highly variable convergence** and the 7 extra
  iters become load-bearing — make `check_every` adaptive (start at 8,
  back off on early convergence signals).
- **If `torch.equal` becomes cheaper on MPS** (e.g., async-bool support
  lands) — drop `check_every` and check per-iter; revert the dataclass
  change.

## Links

- [`../results.md`](../results.md) — per-iter overhead table.
- [ADR 0002](0002-scan-based-sweeps.md) — the scan structure that exposes
  this sync as the bottleneck.
