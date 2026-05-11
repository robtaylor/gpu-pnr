# ADR 0005 — Mask-based segmented scan over INF_PROXY

**Status:** Accepted (2026-05-10). Amended 2026-05-10 — added loop-invariant
hoist (Decision §3). Amended 2026-05-10 — replaced module-constant
`SEG_BARRIER` with per-call autotune (Decision §4).

## Context

The Phase 1 kernel handled obstacles by setting `w[obstacle] = INF_PROXY`
(`= 1e10`) and trusting `cumsum`/`cummin` to keep obstacle-blocked paths large
enough to lose. The scheme breaks at scale: `cumsum(w)` over a 2048-cell row
with even one obstacle accumulates an integer-scale value (`~ 1e10`) that
collides with the float32 ULP of legitimate intermediate sums. Result: legit
distances corrupt past 2048², the entire grid reports `inf`, and the kernel
silently produces wrong answers above that wall.

Diagnosed correctly: the precision wall is `~ 4e6 / N` for unit-accurate
distances. `INF_PROXY = 1e10` was wrong by ~6 orders of magnitude; there's no
single constant that works at all scales because the wall is workload-dependent.

The correct obstacle model needs a true **segmented scan**: each maximal run of
non-obstacle cells along an axis is its own segment, and `cumsum`/`cummin`
must not see across segment boundaries.

## Decision

1. **Drop `INF_PROXY` and the `_mask_polluted` post-pass entirely.** No
   fallback — Phase 1's variant is gone, not preserved behind a flag.
2. **Run `cumsum` on `w_clean = where(obstacle, 0, w)`** so magnitudes stay
   proportional to real path weight. Track segment identity with
   `seg_id = cumsum(obstacle_mask)`. Offset the `cummin` input by
   `seg_id * SEG_BARRIER`, subtract the offset back at the output. Cells in
   different segments are guaranteed to have different `seg_id`, so the
   `SEG_BARRIER` shift makes earlier-segment values larger than the current
   segment and `cummin` can never pick across a boundary. Inside a segment,
   `seg_id` is constant so the offsets cancel exactly.
3. **(Amendment 2026-05-10) Hoist loop-invariant precompute out of the
   convergence loop.** The per-axis `_ScanState` (`seg_cw`,
   `seg_id_barrier`, `obstacle_mask`, `seg_barrier`) depends only on `(w, mask)`,
   not on the evolving `d`. Computing it once per sweep call instead of per
   iteration recovers ~2× per-iter speedup (1024² from 4.06 → 2.34 ms/iter,
   restoring 8.4× vs CPU Dijkstra). Implemented as `_precompute_scan` /
   `_precompute_axis` + a `_ScanState` dataclass.
4. **(Amendment 2026-05-10) Autotune `SEG_BARRIER` per call.** The original
   module-constant `2e4` was tuned for synthetic 5%-obstacle-density grids
   and corrupted distances on real per-net guides with ~93% obstacle density
   (per-row obstacle counts of ~1000 push `seg_id * 2e4` past `1.85e7`, where
   float32 ULP corrupts distances of order 1000 — see
   [`../spikes/phase32-hazard3-real-fixture.md`](../spikes/phase32-hazard3-real-fixture.md)).
   `_autotune_seg_barrier(w, mask, via_cost)` picks the geometric mean of
   `[2 * max_legit_distance_estimate, FLOAT32_PRECISION_BUDGET / max_seg_id]`
   per sweep call. Threaded through every sweep entry point. Cost: ~3 GPU
   syncs (~1.5 ms on MPS), <2% of typical sweep time.

## Consequences

- **The 2048² wall moves to between 4096² and 8192².** With `seg_barrier`
  autotuned, legitimate distances must stay under `seg_barrier/2`; on the
  unit-weight case the constraint is `max_legit_distance * 2 * max_seg_id <
  FLOAT32_PRECISION_BUDGET = 1e7`, which works comfortably to ~5000 per side
  and starts to break around ~8000.
- **No silent corruption past the wall** — if the constraint range is empty
  (workload truly exceeds the float32 precision budget), the autotune falls
  back to the upper bound and the polluted-mask threshold becomes incorrect;
  this is what 8192² unit-weight grids hit, and it's documented behaviour
  rather than a silent wrong answer.
- The kernel is honest about precision: `(seg_barrier, max_legit_distance,
  max_seg_id)` are the three knobs that determine where the wall sits. The
  next escape (Phase 3.3 tile decomposition) splits a too-big grid into
  several smaller ones, each well inside the precision budget.
- **Per-iter cost is essentially Phase 1 parity** after the hoist (1024²: 2.06 →
  2.34 ms/iter; 8.4× vs CPU instead of 9.5×). The trade is ~12% throughput
  for full obstacle correctness past 2048².
- **Per-call autotune overhead** (~1.5 ms) is negligible at our sweep cost.
  For per-net routing at the Hazard3 spike it fires per net but the per-net
  budget is dominated by the convergence loop anyway.

## Walk-back options

- **If a future workload triggers the autotune's empty-range fallback often** —
  this is the precision wall biting. Move to [ADR 0009](0009-per-net-grids.md)'s
  per-net-grid approach or to tile decomposition before chasing this in
  float32. Bumping to `bfloat16` doesn't help; `float64` on MPS is unsupported.
- **If MPS gets a native segmented-cumsum primitive** — replace the offset
  trick with the primitive; simpler kernel, possibly fewer ops.

## Links

- [`../architecture.md`](../architecture.md) — full derivation in the
  `gpu_pnr.sweep` section.
- [`../results.md`](../results.md) — before/after scaling table for Phase 3.1.
- [`../spikes/phase32-hazard3-real-fixture.md`](../spikes/phase32-hazard3-real-fixture.md)
  — where the module-constant `SEG_BARRIER=2e4` was discovered to be wrong
  on real per-net guides.
- [ADR 0002](0002-scan-based-sweeps.md) — the scan structure this preserves.
