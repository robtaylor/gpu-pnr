# Roadmap

## Origin: where E5 came from

The literature survey (separately captured in session notes — to be promoted
into a proper survey doc here) established that **GPU-accelerated detailed
routing for ASICs is essentially unsolved in the open literature**. Almost all
published GPU routing work targets *global* routing (GAMER, FastGR, Superfast
GR, EDGE, DGR, InstantGR, HeLEM-GR). Detailed routing — the longest stage of
physical design, DRC-constrained, ripup-and-reroute-driven — remains
stubbornly CPU-bound.

Five candidate experiments emerged:

1. **E1**: cuOpt MILP for track assignment / ILP-relaxation DR.
2. **E2**: GPU pin-access analysis.
3. **E3**: End-to-end differentiable detailed routing (DREAMPlace-style).
4. **E4**: RL net ordering + GPU parallel maze.
5. **E5**: Sweep-based detailed routing — extending GAMER's sweep primitive
   from coarse global routing into fine-grain detailed routing.

E5 was picked first as the cleanest extension of an established GPU primitive
(sweep) to the detailed regime, with a plausible path to working code on Apple
Silicon today (E1 needs cuOpt = CUDA = waiting for the user's RAM).

E1 is the planned next experiment after E5 reaches a real-fixture milestone.

## Phase 1 — DONE

Single-net sweep SSSP on a 2D grid, validated against Dijkstra, with sequential
multi-net routing on top.

- ✓ Scan-based sweep (cumsum + cummin) — peak 9.5× vs CPU Dijkstra at 1024².
- ✓ Async pipelining via `check_every` — removes per-iter sync.
- ✓ Sequential multi-net router with per-net obstacle marking.
- ✓ 12 tests (correctness + multi-net invariants).
- ✓ Scaling characterised; precision wall identified at 2048².

See [results.md](results.md) for numbers.

## Phase 2 — IN PROGRESS

Three additions to make the multi-net router behave more like a real router:

### 2.1 Endpoint reservation (next)

Mark all pin cells (sources + sinks of all nets) as obstacles before any
routing starts. Each net's own pins are temporarily un-blocked while routing.
This recovers most of the 27/50 failures observed in Phase 1, since the
failure mode is mostly "later nets' pre-chosen pins land on earlier routes".

Estimated effort: <1 day. Mechanical change to `route_nets`.

### 2.2 Net ordering

Heuristic ordering of the input net list. Candidates:
- HPWL-ascending (short nets first; intuition: route the inflexible ones
  while there's still slack).
- HPWL-descending (long nets first; lay down the spine).
- Fanout-based (high-fanout nets first).
- Random + retry with seed search (gradient-free baseline).

Implement as a pluggable strategy in `route_nets` or a sibling
`order_nets()` function. Likely 1–2 days including a small benchmark suite.

Expected impact: ~10–20% wirelength improvement; modest success-rate
improvement on top of endpoint reservation.

### 2.3 Sweep-sharing (concurrent multi-source) — kernel done, integration deferred

`sweep_sssp_multi(w, K-sources)` is implemented and tested: agrees with
per-source `sweep_sssp` within float32 tolerance, and gives 3.1× speedup
at 256² K=50.

**But: no speedup at 1024² (~target scale).** The per-source sweep is
memory-bandwidth-bound at that size; sharing doesn't reduce data
movement. See `results.md` for the full table.

This means batched-router integration (`route_nets_batched` with
conflict detection + ripup) is deferred — it would add complexity
without throughput gain at our scale. The kernel is kept as-is for
the tile-decomposition path (Phase 3.4 below).

**The interaction with ordering** is the same as before: ordering moves
up to batch construction and back down to ripup queue priority. We just
won't pay that complexity cost until tile decomposition makes the
batching worthwhile.

## Phase 3 — IN PROGRESS

Phase 3.4 (multi-layer + via cost) landed first; 3.1 done; 3.2, 3.3 still planned.

### 3.1 Mask-based obstacle handling — DONE

`INF_PROXY` is gone. The sweep now runs `cumsum(w_clean)` (obstacles=0) plus a
parallel `seg_id` track via `cumsum(obstacle_mask)`, with the cummin input
offset by `seg_id * SEG_BARRIER` (`SEG_BARRIER = 2e4`) so cummin can never
pick across a segment boundary. `seg_cw[j]` (cumsum-from-current-segment-start)
is computed as `cw - cummax(cw_at_obstacle_positions)`. A polluted-mask step
catches the case where an entire segment is unreachable. See `docs/architecture.md`
for the full derivation and `docs/results.md` for before/after scaling numbers.

**Outcome:** the 2048² wall is gone; the new wall is between 4096² and 8192²
because legit distances must stay under `MAX_LEGIT_DISTANCE = SEG_BARRIER/2
= 1e4`. For unit weights that's grids up to ~5000 per side. Bumping the
wall further is mechanical (raise `SEG_BARRIER`, re-tune threshold) but
trades float32 ULP at intermediate products for max-distance headroom.

After hoisting the loop-invariant `cumsum`/`cummax`/`seg_cw`/`seg_id_barrier`
precompute out of the convergence loop (those depend only on `w`, not `d`),
per-iter cost is essentially back to Phase 1 parity: 1024² is 2.34 ms/iter
at 8.4× speedup vs CPU (Phase 1 was 2.06 ms/iter at 9.5×). The hoist also
halves per-iter cost at 4096² (67.93 → 31.34 ms/iter) — same Phase 1
convergence behavior, on a now-correct kernel.

### 3.2 Real-fixture integration (Hazard3 GF180MCU) — SPIKE LANDED

A pre-computed LibreLane run for Hazard3 level_3 on gf180mcuD already exists
at `~/Code/Apitronix/hazard-test/hazard3/librelane/runs/RUN_2026-05-08_22-32-24/`,
so the fixture work is parsing-only — no LibreLane execution needed. See
[phase32_spike.md](phase32_spike.md) for the single-net spike result.

**What landed:** ad-hoc Python parser for `after_grt.guide`, single-net
routing pipeline, two real nets routed end-to-end (one trivial M1-only,
one M1->M2->M3->M2->M1 multi-layer with 4 vias). Distances match
Manhattan + via_cost exactly. Kernel handles real GR-derived geometry
without modification.

**Key finding:** the Phase 3.1 module-constant `SEG_BARRIER=2e4` is too
high for real per-net guides because they have ~93% obstacle density per
row. Auto-tuning `SEG_BARRIER` from `max(seg_id_per_axis)` is the next
natural fix and unblocks fixture work at scale.

**Landed under 3.2 so far:**
- Single-net spike (`scripts/spike_route_one_net.py`).
- Auto-tune `SEG_BARRIER` per call (replaced the module constant; 1.5ms
  overhead per sweep).
- Multi-net spike (`scripts/spike_route_many_nets.py`): 500 small 2-pin
  nets all routed end-to-end on real LibreLane data; per-net latency
  41-50ms (kernel-launch-bound at these tiny grids).
- TritonRoute comparison: post-DR DEF NETS-section parser
  (`parse_def_nets`); aggregate per-net wire and via counts. Headline:
  our router uses ~10x fewer vias than TritonRoute because it doesn't
  model the pin-access constraint (M1 reserved for intra-cell routing).
  Wire ratio looks competitive at 1.07x for the smallest 50 nets but
  grows to 1.33x at 500 nets and is misleading anyway -- both numbers
  are dominated by the via-tax we're not paying.

**Still TODO under 3.2:**
- M1-as-pin-access-only cost model: set M1 wire cost high (~1000) to
  force routing onto M2+. Cheapest experiment to make the comparison
  honest.
- Preferred routing direction (Metal1=H, Metal2=V, ...) — needed for
  honest wirelength comparison once vias are paid correctly.
- Multi-pin net handling (Hazard3 has many; spike was 2-pin only).
- LEF parsing (only needed when pin coords aren't inferable from Metal1
  patches; `lefdef` on PyPI is macOS-broken, so build-from-source or
  ad-hoc parser are the options).
- DRC compliance.

### 3.3 Tile decomposition (the path to actual sweep-sharing value)

Split a 1024² (or larger) grid into 4×4 = 16 tiles of 256² each, with
halo overlap between tiles. Within each tile, route all its nets via
sweep-sharing (3.1× win at that grid size). At tile boundaries,
reconcile partial routes through the halo zone. Standard parallel-
routing partitioning, well-studied for FPGA routing.

This is what unlocks Phase 2.3's sweep-sharing kernel at chip scale.

### 3.4 Multi-layer + via cost — DONE

`sweep_sssp_3d` and `route_nets_3d` operate on a `(L, H, W)` cost tensor.
Edge model: horizontal arrival at `(l, r, c)` pays `w[l, r, c]`; via
arrival pays only `via_cost`. Vias respect per-layer obstacles — they
neither land on nor chain through blocked cells.

Inner loop adds two sequential per-layer min relaxations along `axis=0`
(up then down, each followed by an obstacle re-mask) on top of the
existing four intra-layer sweeps. A cumsum-cummin scan along `axis=0`
was the first design choice and would have kept the layer phase a single
parallel pass — but it lets vias chain through intermediate obstacle
layers, so the sequential form is the correct one. Cost: `2(L-1)`
min/where ops per outer iter, dwarfed by the four intra-layer scans
for typical ASIC stacks (L=4-12).

16 new tests; full suite 35/35 green. See [results.md](results.md) for
the inner-loop derivation and the negative finding on the parallel scan.

**What was deliberately deferred:** per-layer preferred routing direction
(M1=H, M2=V, ...). The current scalar `w[l, r, c]` model can't express
anisotropic edge weights; a real fixture (3.2) is the right place to
decide whether to add per-cell direction-cost or whether to model
preferred-direction via a via-cost surcharge for direction changes. The
kernel as it stands treats every layer as fully 4-connected.

## Phase 4 and beyond (sketches)

- **Sweep + DRC kernel co-iteration** — interleave OpenDRC-style design-rule
  checks with sweep iterations rather than as a separate post-pass. The
  hard problem in DR.
- **CUDA / cuOpt port for E1** when CUDA hardware is back online.
- **Tile + halo decomposition** for grids beyond the precision wall, even
  with mask-based obstacles. Standard parallel-routing partitioning.

## What's *not* on the roadmap

- A general-purpose router for any process node. The scope is "demonstrate
  GPU detailed routing techniques on a real but small workload (Hazard3 +
  GF180MCU)" — not a complete tool.
- Pin-access analysis (E2), differentiable DR (E3), RL net ordering (E4) —
  these are sibling experiments to E5, planned but separate.
- Generic graph SSSP. The kernel is grid-shaped on purpose; ASIC routing
  grids have known structure that the scan exploits.
