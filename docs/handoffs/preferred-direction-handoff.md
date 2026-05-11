# Handoff — preferred-direction: gate-then-implement the next Phase 3 slice

**Created:** 2026-05-11
**Working tree:** clean
**Branch:** main

<!--
Reminder: a handoff is ephemeral. At resolution, every load-bearing piece
below migrates into a docs/adr/, docs/plans/, docs/spikes/, or design-doc
home, and this file is then `git rm`'d in the same commit as the migration.

See docs/handoff-discipline.md for the migration table.
-->

## Goal & next-up

**Goal of this session:** adopt the four-document discipline (ADRs, plans,
spikes, handoffs) in this repo, extracting 9 retroactive ADRs from the prior
freeform docs, splitting the roadmap into the Phase 3 plan, moving the spike
under `docs/spikes/`, and resolving the prior YAML handoff into permanent
homes per the discipline's fold-then-delete rule.

**Next session should pick up:** [Phase 3 plan WS3.2 deliverable 1 —
per-layer preferred routing direction](../plans/phase3-detailed-routing.md#ws32--real-fixture-integration-hazard3-on-gf180mcud).
Start by deciding **option A** (per-axis cost multiplier per layer) **vs
option B** (per-cell H/V edge costs) — see "Critical context" below for why
this is less of a fork than the plan suggests. Then implement, then re-run
the TritonRoute comparison to see whether the via-ratio gap closes from
0.78× toward 1.0× and the wire-ratio drift stabilises.

**Verification command:**

```sh
cd ~/Code/gpu-pnr && uv run pytest tests/
# Expect: 43 passed
```

## Done this session

| Commit | Subject | Notes |
|---|---|---|
| `5ce4852` | docs: adopt four-document discipline (ADRs, plans, spikes, handoffs) | 27 files; +1450/−378. Created `CLAUDE.md`, 9 ADRs (0001–0009), Phase 3 plan, spike doc under `docs/spikes/`. Removed `docs/roadmap.md` (split into ADRs + plan). |
| _follow-up_ | docs: backfill reference repos + open CUDA-RAM strategic question | Pending — adds two items that the YAML-handoff fold missed first time around. |

## Open follow-ups (priority-ordered)

### 1. Per-layer preferred direction (small-to-medium)

[Plan link](../plans/phase3-detailed-routing.md#ws32--real-fixture-integration-hazard3-on-gf180mcud).
Pick option A or B (see critical context), thread a per-axis cost tensor
through `sweep_sssp_3d` and `route_nets_3d`, expose a builder helper that
takes a base `w` and per-layer multipliers, and wire a `--prefer-direction`
knob into `scripts/spike_route_many_nets.py`. Then re-run the TR comparison
at N = 50, 200, 500.

### 2. Multi-pin net router (medium)

[Plan link, deliverable 2](../plans/phase3-detailed-routing.md#ws32--real-fixture-integration-hazard3-on-gf180mcud).
~11K of 24K Hazard3 nets have 3+ pins. Sequential point-to-point with
re-rooting, or Steiner-tree-flavored. Don't bundle with (1) — separate
session.

### 3. Per-via-pair `via_cost` array (small)

[Plan link, deliverable 3](../plans/phase3-detailed-routing.md#ws32--real-fixture-integration-hazard3-on-gf180mcud).
Replace the scalar with a length-`(L-1)` array. Tiny API change; finishes
the per-via realism story. Defer until (1) lands so the TR-comparison
deltas are interpretable.

## Critical context

**The A-vs-B fork is smaller than the plan implies.** `_precompute_axis(w,
obstacle_mask, axis, seg_barrier)` in `src/gpu_pnr/sweep.py` already takes a
cost tensor per axis — `sweep_sssp_3d` happens to pass the same `w` to both
axes today, but nothing in the kernel structure requires that. So:

- **Option A** ("per-axis cost multiplier per layer") and **option B**
  ("per-cell H/V edge costs") are the *same kernel surgery* — both pass
  separate `w_h` and `w_v` tensors. The only difference is how the
  application builds those tensors: A factors as `w_h[l,r,c] = w[l,r,c] *
  h_mult[l]`, B builds them freely.
- Recommendation: implement at the kernel level as "takes `w_h, w_v`" (the
  general form), and add a small builder helper on top that produces the
  A-style factored tensors from a base `w` and per-layer multipliers. Then
  option B is free if needed later.
- The obstacle mask is also per-axis in this generalisation: an obstacle in
  `w_h` blocks horizontal entry to a cell; an obstacle in `w_v` blocks
  vertical entry. That's how you'd model "M1 is horizontal-pref means
  vertical moves on M1 are expensive but not infinite" vs "M1 is
  horizontal-only means vertical moves on M1 are blocked".

**TritonRoute baseline numbers to beat** (from spike, `m1_cost=10`):

| Sample | wire ratio vs TR | via ratio vs TR |
|--------|------------------|-----------------|
| 50     | 1.08×            | 0.76×           |
| 200    | 1.26×            | 0.78×           |
| 500    | 1.36×            | 0.80×           |

Hypothesis: with preferred direction, via ratio at 500 nets closes to ~1.0×
and wire ratio stabilises across sample sizes (no more 1.08× → 1.36× drift).
If it doesn't, the next layer of unmodelled constraints is per-via-pair cost
asymmetry or multi-pin Steiner topology (see deliverables 2 and 3).

**Fixture details that aren't in any doc:** the `m1_cost` knob is a
post-`build_grid` multiplier on Metal1 cells only — `scripts/spike_route_many_nets.py`
applies it just before the route call. The preferred-direction work should
*replace* this knob, not stack on top of it.

**Memory entry:** `~/.claude/projects/-Users-roberttaylor-Code-gpu-pnr/memory/hazard3_fixture.md`
points at the LibreLane fixture location. The pre-computed run at
`~/Code/Apitronix/hazard-test/hazard3/librelane/runs/RUN_2026-05-08_22-32-24/`
is the input — no LibreLane execution is needed.

## References

- [`../plans/phase3-detailed-routing.md`](../plans/phase3-detailed-routing.md) — current workstream state, deliverables, exit criteria.
- [`../spikes/phase32-hazard3-real-fixture.md`](../spikes/phase32-hazard3-real-fixture.md) — spike that motivates this work; M1-cost experiment numbers.
- [`../adr/0006-sequential-via-relax.md`](../adr/0006-sequential-via-relax.md) — the 3D-via decision that this work builds on; the "Walk-back options" section flags exactly the per-axis-cost generalisation we're about to do.
- [`../adr/0002-scan-based-sweeps.md`](../adr/0002-scan-based-sweeps.md) — the scan trick the per-axis form must preserve.

## Migration note

When this handoff resolves (preferred-direction landed, TR comparison re-run):

- Decision on **option A vs B** (kernel API takes separate `w_h, w_v`) → new ADR, citing this handoff's critical context.
- Updated TR-comparison numbers → `docs/results.md` (append new section) and `docs/spikes/phase32-hazard3-real-fixture.md` (update the "M1-as-pin-access" table or add a successor section).
- WS3.2 deliverable 1 → mark "Shipped commit `<sha>`" in `docs/plans/phase3-detailed-routing.md`.
- The "preferred direction is the next layer of unmodelled constraints" hypothesis from the spike doc → confirmed or amended.
- Then `git rm docs/handoffs/preferred-direction-handoff.md` in the same commit as the migration.
