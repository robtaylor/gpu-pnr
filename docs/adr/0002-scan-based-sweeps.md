# ADR 0002 — Scan-based axis sweeps over Python-loop sweeps

**Status:** Accepted (2026-05-09).

## Context

The Bellman-Ford / fast-sweeping kernel needs four directional passes (H-fwd,
H-bwd, V-fwd, V-bwd) per outer iteration. The natural Python implementation is
a loop over rows (or columns) calling `torch.minimum` per row.

At 1024², that dispatches **1024 kernel launches per axis sweep**, four sweeps
per iteration, dozens of iterations. The launch overhead dominates: the loop
form ran at ~6 ms/iter on 1024², making the kernel **3× *slower* than CPU
Dijkstra** at the target grid size.

The forward-sweep recurrence per row,

```
d_new[j] = min over k <= j of (d[k] + sum(w[k+1..j]))
```

expands algebraically to

```
d_new = cumsum(w) + cummin(d - cumsum(w))
```

— a single parallel scan that PyTorch dispatches as one MPS kernel. Backward
sweep = `flip → forward → flip`. The two-cumsum identity is the load-bearing
insight; once known, the implementation is a few lines.

## Decision

Implement each per-axis sweep as **one `cumsum` plus one `cummin`** (plus the
algebraic offset/unoffset), and lift the per-row Python loop out of the kernel
entirely. Apply uniformly to 2D (`sweep_sssp`), multi-source (`sweep_sssp_multi`),
and 3D (`sweep_sssp_3d`).

## Consequences

- **Got us from 3× *slower* than CPU Dijkstra to 9.5× faster** at 1024² (Phase 1
  baseline, INF_PROXY-based; numbers in [`../results.md`](../results.md)).
- Per-iter cost is now memory-bandwidth-bound, not launch-bound — at 1024² the
  inner kernel is ~2 ms/iter, dominated by the four `cumsum`+`cummin` passes
  themselves.
- Constrains every later design choice (obstacle handling, multi-layer vias) to
  preserve the scan structure. The recurring shape of negative findings —
  see [ADR 0006](0006-sequential-via-relax.md) — is "the parallel-scan
  formulation is tempting but silently mis-models obstacles."
- `float32` precision becomes load-bearing at scale: `cumsum` accumulates with
  the working dtype, so the precision wall is the new ceiling rather than raw
  throughput. See [ADR 0005](0005-mask-based-segmented-scan.md) for how that
  ceiling was pushed back.

## Walk-back options

- **If PyTorch's MPS `cumsum`/`cummin` regress badly** — keep the algebraic form
  and dispatch the scan via a custom MPS kernel; only the host changes.
- **If a future grid model has anisotropic per-direction edge costs that don't
  factor cleanly into a per-cell `w`** — the scan trick may need a separate
  `w_h` / `w_v` tensor per axis. The kernel already supports this implicitly:
  `_precompute_axis` takes a cost tensor and an axis, so passing different
  tensors per axis is a one-line generalisation.

## Links

- [`../architecture.md`](../architecture.md) — kernel API and scan derivation.
- [`../results.md`](../results.md) — measured per-iter cost and speedup.
- [ADR 0003](0003-async-convergence-check.md) — the convergence-check
  optimisation that further amplifies this win.
