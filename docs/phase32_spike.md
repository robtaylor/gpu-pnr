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

## Next steps

In rough priority order:

1. **Auto-tune `SEG_BARRIER`.** The fact that a single module constant doesn't
   fit both synthetic 4096^2 (`SEG_BARRIER=2e4` is right) and real per-net
   guides (`SEG_BARRIER=5e3` is right) means the constant should be derived.
   Smallest change: compute `SEG_BARRIER = 4 * max(seg_id_per_axis) ` (or
   similar) inside `_precompute_axis`. Falls out of obstacle-density
   automatically.
2. **Compare to TritonRoute.** Parse the final-DR DEF, find the same net's
   route, compare wirelength / via count / total cost. If the kernel's path
   diverges from TritonRoute, the divergence is the *interesting* finding.
3. **Multi-pin nets.** Pick from the ~11,000 nets with 3+ Metal1 rectangles
   in the guide. Will probably need a router-level change (sequential
   point-to-point construction with re-rooting).
4. **Preferred-direction cost model.** Per-layer x-cost / y-cost split, or
   per-axis-cost-multiplier. Probably needed for any honest TritonRoute
   comparison on routes that span direction changes.

## Files added

- `scripts/spike_route_one_net.py` -- the spike driver. Accepts a net name
  and an optional SEG_BARRIER override.
- `docs/phase32_spike.md` -- this document.
- `~/.claude/projects/-Users-roberttaylor-Code-gpu-pnr/memory/hazard3_fixture.md`
  -- reference memory for the fixture location, so future sessions don't
  rediscover it.
