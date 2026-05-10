# Phase 3.2 spike — single Hazard3 net, end-to-end

This document captures the results of a deliberately-tiny Phase 3.2 spike: take
one net from a real LibreLane GR run for Hazard3 level_3 on gf180mcuD, build a
cost grid from its guide rectangles, route it through `sweep_sssp_3d` /
`route_nets_3d`, and write down what was surprising.

The fixture used is the LibreLane run at
`~/Code/Apitronix/hazard-test/hazard3/librelane/runs/RUN_2026-05-08_22-32-24/`
(50,099 instances, 5 metal layers, gf180mcuD, 0.20um wire pitch). All inputs
came from `39-openroad-globalrouting/after_grt.guide` -- 24,123 nets in
human-readable format, parsed by ad-hoc Python in 30 lines (`scripts/spike_route_one_net.py`).

No LibreLane execution was needed; no LEF/DEF parsing was needed for the spike
itself. Pin coordinates came from the centers of the first/last `Metal1`
rectangles in the guide -- correct for 2-pin nets where the two Metal1 patches
*are* the two pins.

## What worked

**Trivial 2-pin same-layer case (`_00013_`, 5 rectangles).**
Source and sink one gcell apart on Metal1. Sweep converges in 16 iterations,
distance is exactly the column delta (84 cells), zero vias, path is a straight
Metal1 line. Multi-net `route_nets_3d` agrees. Sanity-check pass.

**Multi-layer 2-pin case (`_00000_`, 10 rectangles, M1->M2->M3->M2->M1 guide).**
Source and sink 1344 cells apart in Manhattan, on opposite corners of the
guide. Sweep converges in 16 iterations, distance is 1364 = 1344 + 4 vias x 5
via_cost. Path uses Metal1, Metal2, Metal3 with 4 via transitions; takes M2
vertically from source to mid-grid, M3 horizontally across, M2 vertically down,
M1 to sink. Sensible layer hierarchy use even without preferred-direction
modelling.

End-to-end, the kernel runs on real LibreLane geometry without modification.
That's the headline.

## What didn't work, and why

**`SEG_BARRIER=2e4` (the Phase 3.1 module-constant default) corrupts distances
on real per-net guides.**

Synthetic Phase 3.1 tests had ~5% obstacle density across the whole grid (~200
obstacle cells in any one row of 4096). Real per-net guides invert this:
93% of the grid is "outside the routable region" (i.e., obstacle), so a typical
row has *thousands* of consecutive obstacle cells. For `_00000_` row 42 of
Metal1 has 924 consecutive obstacle cells; `seg_id * SEG_BARRIER` reaches
`924 * 2e4 = 1.85e7`, where float32 ULP is ~2-3. Distances of order 1000 get
corrupted by integer-scale errors and the kernel reports `d[sink] = 984`, which
is *less than the Manhattan minimum (1344)*. Backtrace then fails because no
valid predecessor chain exists.

Empirically:

| `SEG_BARRIER` | iters | `d[sink]` | Note |
|---|---|---|---|
| 2e4 | 240 | 984.0 | bogus -- below Manhattan |
| 1e4 | 16 | 1364.0 | correct |
| 5e3 | 16 | 1364.0 | correct |
| 2e3 | 16 | inf | polluted-mask false-positives (threshold too low) |

The valid range is `2 * max_legit_distance < SEG_BARRIER < 1.6e7 / max_seg_id`.
For this spike: `2 * 1364 = 2728 < SEG_BARRIER < 1.6e7 / 924 = 17,316`. The
Phase 3.1 default `2e4` is just outside this range.

The spike script accepts `SEG_BARRIER` as a CLI argument; `5e3` works for these
nets. **The proper fix is per-call auto-tuning**: derive `SEG_BARRIER` from
`(grid_size, max(seg_id_per_row))` at sweep entry. That's a small kernel API
change (constructor argument or computed-from-w default) and is the next
natural piece of work.

## What was deliberately punted in this spike (and remains punted)

1. **Multi-pin nets.** Hazard3 has many >2-pin nets; we only handled the
   12,770 two-pin ones in the guide. Multi-pin needs Steiner-tree-flavored
   handling on top of route_nets_3d.
2. **Preferred routing direction.** gf180mcuD has Metal1=H, Metal2=V,
   Metal3=H, ... Our cost model is uniform per-cell. Real routes here would
   penalize non-preferred-direction segments. We got away with it because
   the guide regions already constrain layer use to align with preferred
   direction (the GR step picked layer-direction allocation for us); but
   for tighter optimality this needs to be modelled.
3. **Per-via-pair `via_cost`.** Single scalar `via_cost` is a simplification.
4. **DRC compliance.** Not checked.
5. **Comparison to TritonRoute.** We produced a path; we haven't yet loaded
   `final/def/synth_top_level_3.def` to see what TritonRoute did for the same
   nets. That comparison is the next obvious deliverable.
6. **LEF/DEF parsing.** The guide-only approach worked for 2-pin nets where
   pin coords are inferable from Metal1 patch centers. Multi-pin or
   pin-on-non-Metal1 cases will need real LEF parsing. The `lefdef` PyPI
   package was tried first and **does not work on macOS** (ships only Windows
   `.dll` and Linux `.so`, no `.dylib`). Either build it from source, switch
   to ad-hoc DEF parsing, or use OpenROAD's `odb` Python (which requires an
   OpenROAD install).

## SEG_BARRIER autotune (landed)

Replaced the module-constant `SEG_BARRIER=2e4` with a per-call autotune that
derives the value from grid shape and obstacle density:

```
lower = 2 * max_legit_distance_estimate     # H+W times max_w_finite, plus L*via_cost
upper = FLOAT32_PRECISION_BUDGET / max_seg_id   # 1e7 / max obstacle count per axis
seg_barrier = sqrt(lower * upper)           # geometric mean of valid range
```

If `lower >= upper` the autotune falls back to `upper * 0.99` and the workload
is documented as exceeding the float32 precision budget (this is what 8192^2
unit-weight grids hit -- the new wall, same place Phase 3.1 documented).

**Cost of the autotune:** ~3 GPU syncs per sweep call (~1.5ms on MPS):
- one masked `max(w_finite)` reduction
- two `cumsum(obstacle_mask) + max + .item()` per spatial axis
At Phase 3.1's 1024^2 sweet spot of ~94ms/sweep, the autotune adds ~1.6%
overhead. For per-net routing the autotune fires per net (each has its own
obstacle pattern), but the per-net work itself is dominated by the
convergence loop, so the autotune cost is in the noise.

**Synthetic perf preserved:** post-autotune `bench_scaling.py` numbers are
within run-to-run noise of the post-Phase-3.1 hoisted-precompute version
(1024^2: 2.34 -> 2.41 ms/iter; 4096^2: 31.34 -> 32.83 ms/iter). 4096^2 still
correct; 8192^2 still hits the precision wall (best-effort fallback).

**Real-fixture perf unblocked:** `_00000_` now routes correctly with no
manual override -- d[sink]=1364 in 16 iterations.

## Multi-net spike (landed)

`scripts/spike_route_many_nets.py` runs `route_nets_3d` on N independent
2-pin nets (each with its own per-net grid built from its guide rectangles).
Sample of the 50 smallest 2-pin nets:

```
=== Aggregate over 50 nets ===
  routed: 50 / 50 (100.0%)
  total wirelength: 7664 cells
  total via transitions: 20
  avg per-net time: 51.1 ms

Layer occupancy (number of routed nets that used the layer):
  Metal1: 50
  Metal2: 10
  Metal3: 0
  Metal4: 0
  Metal5: 0
```

Scaling to 200 and 500 nets: still 100% routed, per-net time stable at 41-45ms.
Most short nets stay on M1 because M1 cost (1 per cell) beats via_cost (5 per
via, 4 vias minimum to use M3 = 20 cost) for short routes. This is the
expected cost-model behavior; preferred-direction modelling would push more
of them off M1, which is left for the next iteration.

**Per-net latency is launch-overhead-dominated** at these tiny per-net grids.
50K Hazard3 nets at 41ms = ~35 minutes total -- comparable to TritonRoute on
a desktop, but kernel-launch overhead would drop substantially with batching
or `torch.compile`. The autotune's ~1.5ms is a small fraction of the per-net
budget here.

## Next steps

In rough priority order:

1. **Compare to TritonRoute.** Parse `final/def/synth_top_level_3.def`, find
   the same nets' actual routes, compare wirelength / via count. The
   spike has a placeholder for this -- it's the *interesting* deliverable
   for honest evaluation.
2. **Multi-pin nets.** Pick from the ~11,000 nets with 3+ Metal1 rectangles
   in the guide. Likely a router-level change (sequential point-to-point
   construction with re-rooting, or Steiner-tree-flavored heuristic).
3. **Preferred-direction cost model.** Per-layer x-cost / y-cost split, or
   per-axis cost multipliers. Needed for any honest TritonRoute comparison
   on routes that span direction changes.
4. **Whole-chip integration.** Replace per-net mini-grids with a single
   chip-scale grid that tracks committed routes globally. Gates on (3) and
   probably bigger SEG_BARRIER headroom (or grid tiling -- Phase 3.3).

## Files added

- `scripts/spike_route_one_net.py` -- single-net debugging driver. Accepts a
  net name and an optional SEG_BARRIER override.
- `scripts/spike_route_many_nets.py` -- multi-net aggregate-stats driver.
- `docs/phase32_spike.md` -- this document.
- `~/.claude/projects/-Users-roberttaylor-Code-gpu-pnr/memory/hazard3_fixture.md`
  -- reference memory for the fixture location.
