# ADR 0008 — Defer `route_nets_batched`; sweep-sharing only via tile decomposition

**Status:** Accepted (2026-05-10).

## Context

Phase 2.3a delivered `sweep_sssp_multi(w, K-sources)` — a batched scan-based
sweep that computes K shortest-path maps in one fused pass. The plan was to
build `route_nets_batched` on top: detect conflicts among the K paths, ripup
the loser(s), reroute.

Throughput vs the K-sequential baseline, measured on the standard benchmark:

| Grid    | K=1   | K=10  | K=50      |
|---------|-------|-------|-----------|
| 256²    | 1.11× | 1.41× | **3.10×** |
| 512²    | 1.41× | 0.78× | 1.30×     |
| **1024²** | **0.37×** | **0.91×** | **0.97×** |

At our target scale (1024²+) the per-source kernel is already
memory-bandwidth-bound. Sharing the sweep doesn't reduce arithmetic or data
movement; it just makes each pass wider and proportionally slower. The 3.10×
at 256² K=50 is real but only useful at sub-target grid sizes.

Building `route_nets_batched` would add **conflict detection, ripup
queue, and re-routing** code complexity on top of a kernel that doesn't pay off
at the size we care about. The complexity cost is real; the throughput payoff
isn't, until tile decomposition (a separate work item) makes the per-tile
problem small enough that the multi-source kernel wins.

## Decision

1. **Keep `sweep_sssp_multi` in the codebase as-is** — it's the right primitive
   for tile decomposition and for later batched routing on real-fixture
   subblocks.
2. **Do not build `route_nets_batched` now.** Sequential `route_nets` plus
   HPWL ordering (ADR 0007) is the production path at chip scale.
3. **Tile decomposition (Phase 3.3) is the unlock condition.** Once a 1024²
   grid is split into 16 tiles of 256² each, the multi-source kernel hits its
   3.10× regime per tile. `route_nets_batched` becomes worth building **at
   that point**, on top of the tile harness.

## Consequences

- Avoids ~hundreds of lines of conflict-detection / ripup code that would not
  earn their keep at current scale.
- Documents the negative result clearly in [`../results.md`](../results.md) so
  future "let's batch the router" attempts start from the right premise.
- Defers the architectural question — sequential vs batched routing on
  real fixtures — until tile decomposition lands.

## Walk-back options

- **When tile decomposition lands and per-tile size is ≤ 256²** — build
  `route_nets_batched` on top of `sweep_sssp_multi`, with conflict resolution
  scoped to the tile + its halo.
- **If a small-grid workload appears** where the per-source kernel really is
  launch-bound — `route_nets_batched` becomes worth it sooner. Unlikely for
  ASIC routing but possible for adjacent applications.

## Links

- [`../results.md`](../results.md) — Phase 2.3a throughput measurements.
- [ADR 0007](0007-hpwl-ascending-net-ordering.md) — the cheaper lever that
  did win at scale.
- [`../plans/phase3-detailed-routing.md`](../plans/phase3-detailed-routing.md)
  — where tile decomposition is tracked.
