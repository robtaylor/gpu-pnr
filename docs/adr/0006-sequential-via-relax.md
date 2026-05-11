# ADR 0006 — Sequential per-layer via relax over parallel scan along layer axis

**Status:** Accepted (2026-05-10).

## Context

`sweep_sssp_3d` (Phase 3.4) operates on an `(L, H, W)` cost tensor with via
transitions between adjacent layers. Per outer iteration, after the four
intra-layer sweeps, the kernel must propagate via paths: relax `d[l]` against
`d[l ± 1] + via_cost`, respecting per-layer obstacles.

The aesthetically obvious choice is a **parallel cumsum-cummin scan along the
layer axis**, analogous to how the row/column sweeps work — one
`cumsum(via_cost * arange(L))` offset, one `cummin` along `axis=0`, in two
passes (up/down). It's a single parallel pass per direction; it keeps the
kernel uniformly scan-based; it matches the elegance of ADR 0002.

It is **wrong under obstacles.** The scan adds `via_cost * |Δl|` for any
layer-pair regardless of intermediate cells — vias can chain through blocked
layers. The bug first surfaced in `test_two_layers_zero_via_collapses_to_2d_min`
and `test_high_via` *passing*, then failing the moment a multi-net router
committed an obstacle on the destination layer.

This is a recurring shape: parallel-scan formulations tempt you with elegance
and silently mis-model the very thing the kernel exists to handle. ADR 0005
hit the same pattern around obstacles in 2D.

## Decision

Use **sequential per-layer min relaxation** along `axis=0`, in two loops (up,
then down), each followed by an obstacle re-mask:

```python
for l in range(1, L):
    d[l] = torch.minimum(d[l], d[l-1] + via_cost)
    d[l] = torch.where(obstacle_mask[l], inf, d[l])

for l in range(L - 2, -1, -1):
    d[l] = torch.minimum(d[l], d[l+1] + via_cost)
    d[l] = torch.where(obstacle_mask[l], inf, d[l])
```

Cost: `2(L-1)` `min`/`where` ops per outer iteration. Each is a full-grid
GPU op, but with `L=4-12` for typical ASIC stacks they're a small fraction
of the four intra-layer scans.

## Consequences

- **Correct under obstacles** — vias neither land on nor chain through blocked
  cells. Validated by 16 new tests in `tests/test_sweep_3d.py` and
  `tests/test_router_3d.py`, including the layer-0-wall-forces-detour case.
- The 3D kernel is no longer uniformly scan-based — the layer axis is a
  Python loop. For `L=12` that's 22 ops/iter on top of the four intra-layer
  scans; on real ASIC stacks (L=5 for gf180mcuD; L=12 for advanced nodes)
  the overhead is bounded and small.
- The negative finding is captured in [`../results.md`](../results.md) under
  Phase 3.4 — keeps the pattern visible so future attempts at "scan-ify the
  3D kernel" don't repeat it.

## Walk-back options

- **If a future ASIC stack has `L > 32`** — revisit. The Python loop's
  constant-factor cost may become load-bearing. Candidates:
  - A custom kernel that does the layer-direction min in two parallel passes
    while reading obstacle masks per layer-pair.
  - A correctness-preserving segmented-scan along `axis=0` that respects
    per-layer obstacles (the same offset trick as ADR 0005, but per layer).
- **If `via_cost` becomes a per-via-pair array** (the natural next step,
  see [`plans/phase3-detailed-routing.md`](../plans/phase3-detailed-routing.md))
  — the sequential form trivially absorbs it; the parallel scan would have
  needed a per-layer-pair cumsum.

## Links

- [`../architecture.md`](../architecture.md) — 3D kernel narrative.
- [`../results.md`](../results.md) — Phase 3.4 section, "Negative finding
  worth flagging."
- [ADR 0002](0002-scan-based-sweeps.md) — the scan trick this deliberately
  abandons for the layer axis.
