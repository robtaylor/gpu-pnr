# Plans

Long-lived plan documents for in-flight and upcoming work. One file per
phase or workstream. Updated in place; past states live in `git log`.

For the discipline behind these, see
[Claude Project Discipline — Plans](https://robtaylor.github.io/claude-project-discipline/plans.html).

## Active plans

| Plan | Status | One-line |
|------|--------|----------|
| [phase3-detailed-routing](phase3-detailed-routing.md) | Active | Real-fixture integration (Hazard3 on gf180mcuD), preferred-direction modelling, tile decomposition. |

## Closed plans

None yet. Phase 1 (single-net sweep + sequential router) and Phase 2 (pin
reservation, HPWL ordering, multi-source kernel) shipped before this
discipline was adopted; their outcomes are captured in
[`../results.md`](../results.md) and in the corresponding ADRs
([0002](../adr/0002-scan-based-sweeps.md), [0003](../adr/0003-async-convergence-check.md),
[0004](../adr/0004-cpu-backtrace.md), [0007](../adr/0007-hpwl-ascending-net-ordering.md),
[0008](../adr/0008-defer-route-nets-batched.md)).

## Template

See [`_template.md`](_template.md) for the skeleton.
