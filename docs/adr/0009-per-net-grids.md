# ADR 0009 — Per-net independent grids for the Hazard3 spike

**Status:** Accepted (2026-05-11).

## Context

The Hazard3 LibreLane fixture is a real 24,123-net design on gf180mcuD with a
post-GR `.guide` file giving each net a layer-tagged rectangle list (its
allowed routing region). Two integration options:

1. **Whole-chip grid.** Build one `(L, H, W)` cost tensor for the entire die
   (gf180mcuD core area ≈ 1mm × 1mm, 0.20 µm pitch → ~5000² per layer × 5
   layers). Each net's routing reserves cells globally, so subsequent nets see
   prior routes as obstacles. This is the "real router" model.
2. **Per-net independent grid.** For each net, build a small grid from the
   bounding box of its guide rectangles. Each net routes against an obstacle
   pattern derived from its own guide only — other nets' commits are not
   visible.

Option (1) hits two problems immediately: the float32 precision wall
([ADR 0005](0005-mask-based-segmented-scan.md)) constrains effective grid size
below ~5000²; and reasoning about per-net correctness becomes tangled with
chip-scale state. Option (2) sidesteps both — small grids (typically 50–500
cells per side) stay well inside the precision budget, and per-net behaviour
is isolated for debugging.

The spike's goal was to validate **kernel-on-real-geometry**, not to compete
with TritonRoute as a chip-scale router.

## Decision

For Phase 3.2 (real-fixture spike), use **per-net independent grids**: build
an `(L, H, W)` cost tensor from each net's guide bounding box, route on it,
report results. No global obstacle state across nets.

## Consequences

- Spike unblocked: 500 small 2-pin nets routed end-to-end at 41–50 ms each,
  100% success rate (see [`../spikes/phase32-hazard3-real-fixture.md`](../spikes/phase32-hazard3-real-fixture.md)).
- Per-net latency is dominated by GPU kernel launch overhead at these tiny
  grids — extrapolated to 50K Hazard3 nets, ~35 minutes total wall time
  before any batching/`torch.compile` optimisation. Comparable to TritonRoute
  on a desktop.
- **Cannot detect cross-net conflicts** by construction — net A and net B can
  both route through the same physical cell. This is a deliberate
  simplification for the spike; whole-chip integration (a future plan
  workstream) is where this is fixed.
- TritonRoute comparison is still meaningful per-net: each net's path is
  legal in its own right, just not globally consistent. Aggregate wire/via
  ratios behave well (see the spike doc and `m1_cost` experiment).

## Walk-back options

- **When whole-chip integration becomes the priority** — replace per-net grids
  with a chip-scale grid (or tile-decomposed equivalent — see Phase 3.3 in
  [`../plans/phase3-detailed-routing.md`](../plans/phase3-detailed-routing.md)).
  Gated on:
  - Preferred-direction modelling landing (otherwise wire-ratio drift
    obscures any chip-scale comparison).
  - A precision-wall mitigation strategy (tile decomposition is the
    natural path; ADR 0005 walk-back).
- **For specific debugging scenarios** — keep the per-net mode as an opt-in
  even after whole-chip lands; it's a clean isolation tool.

## Links

- [`../spikes/phase32-hazard3-real-fixture.md`](../spikes/phase32-hazard3-real-fixture.md)
  — the spike results.
- [ADR 0005](0005-mask-based-segmented-scan.md) — the precision wall this
  side-steps.
- [`../plans/phase3-detailed-routing.md`](../plans/phase3-detailed-routing.md)
  — Phase 3.3 (tile decomposition) and whole-chip integration plan.
