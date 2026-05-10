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

### 2.3 Sweep-sharing (concurrent multi-source)

Adapt the 2025 ICCAD LBR "sweep-sharing" technique: route multiple nets
concurrently in a single sweep operation by maintaining per-source distance
buffers and propagating them in parallel. K-fold throughput gain at the cost
of per-batch quality (nets in a shared sweep don't see each other's
commitments yet).

This is the **bigger lever for production scale** (10K-net designs) and
where the actual GPU throughput story lives. Sequential routing as in Phase
1 is fundamentally bottlenecked at 1 sweep per net.

The interaction with ordering: ordering moves up to *batch construction*
(which nets can share a sweep without contention?) and back down to *ripup
queue priority* (when conflicts are detected, which loser routes first?).

Estimated effort: 1–2 weeks. Requires:
- Per-source distance buffer (4D tensor: K × H × W or similar)
- Multi-source initialisation
- Conflict detection on K simultaneous backtraces
- Ripup queue + repair loop

## Phase 3 — PLANNED

### 3.1 Mask-based obstacle handling

Replace the `INF_PROXY` trick with a separate `obstacle_mask` tensor that
short-circuits propagation through blocked cells without inflating cumsum
magnitudes. Restores full float32 dynamic range for legit distances and
removes the 2048² precision wall.

Approach: split each row/column into segments at obstacle boundaries; run
the scan within each segment independently; recombine. Vectorisable via
segmented-scan primitives (PyTorch doesn't have one built-in, but it's
constructible from cumsum + reset-on-mask patterns).

### 3.2 Real-fixture integration (Hazard3 GF180MCU)

Capture LibreLane GR output for Hazard3 level_3 on gf180mcuD, extract a
small subblock (~10K nets), convert to the grid + net-list model, route via
gpu-pnr, compare against OpenROAD/drt's detailed routing on:
- Total wirelength
- Via count
- DRC violation count
- Wall-clock for the routing stage

This is the milestone that turns gpu-pnr from a kernel demo into an actual
ASIC routing experiment. Requires:
- LibreLane harness script
- OpenROAD `odb` Python API integration for LEF/DEF/guide parsing
- 3D / multi-layer extension of the sweep kernel (per-layer sweeps + via cost)
- Comparison harness

### 3.3 Multi-layer + via cost

The current kernel is 2D. Real ASIC routing is multi-layer with via
transitions. Per-layer sweeps + a cross-layer via-update step. Adds a layer
dimension to all tensors and modest complexity to the inner loop. Should
not change the asymptotic GPU efficiency story.

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
