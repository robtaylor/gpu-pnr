# ADR 0004 — CPU-side backtrace on unified memory

**Status:** Accepted (2026-05-09).

## Context

Path reconstruction walks the distance tensor from sink to source one cell at
a time, picking the neighbour with the smallest predecessor distance. The hot
operation is reading a few scalar tensor entries per step (`d[neighbour]` and
`w[neighbour]`). On MPS, every such read is a `.item()` that forces a
CPU↔GPU sync.

For a path of length L cells, that's O(L) syncs — 145 ms/net at 1024² in the
naive implementation, versus 24 ms for the sweep that produced the distance
map. Backtrace was dominating multi-net routing.

Apple Silicon's unified memory architecture changes the calculus: `d.cpu()` is
metadata-only — same physical RAM, just a view with the CPU storage descriptor.
A CPU-side backtrace on that view costs nothing in transfer but eliminates the
per-cell sync.

## Decision

Run `backtrace` and `backtrace_3d` on `.cpu()` views of `d` and `w`. The kernel
itself stays on MPS; only the path-reconstruction loop runs on the host.

## Consequences

- **Backtrace cost drops ~6×** at 1024² (815 ms → 145 ms per routed net in the
  multi-net demo).
- Apple-Silicon-specific *in optimisation*, not in correctness — on a discrete-
  GPU host the `.cpu()` would copy memory and we'd lose the benefit, but the
  code still works.
- The router's per-net loop stays clean: `sweep_sssp_3d` returns MPS-resident
  tensors; the path-reconstruction step adds a `.cpu()` view and the rest is
  CPU-side scalar arithmetic.

## Walk-back options

- **If we port to a discrete-GPU host (CUDA box, E1)** — implement backtrace as
  a GPU kernel that produces the entire path tensor in one pass, or batch the
  sync by reading entire neighbour-slices per step.
- **If a future kernel design produces a path tensor as part of the sweep
  itself** — drop backtrace entirely.

## Links

- [`../architecture.md`](../architecture.md) — Apple-Silicon-specific notes.
- [`../results.md`](../results.md) — multi-net routing cost numbers.
- [ADR 0001](0001-pytorch-mps-host.md) — the host platform decision that makes
  this optimisation cheap.
