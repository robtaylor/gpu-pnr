# Architecture

A tour of the codebase's modules and data flow. Decision rationale lives in
[`adr/`](adr/); forward-looking work in [`plans/`](plans/); measured results
in [`results.md`](results.md).

## Origin: where E5 came from

The literature survey established that **GPU-accelerated detailed routing for
ASICs is essentially unsolved in the open literature**. Almost all published
GPU routing work targets *global* routing (GAMER, FastGR, Superfast GR, EDGE,
DGR, InstantGR, HeLEM-GR). Detailed routing — the longest stage of physical
design, DRC-constrained, ripup-and-reroute-driven — remains stubbornly
CPU-bound.

Five candidate experiments emerged:

1. **E1**: cuOpt MILP for track assignment / ILP-relaxation DR.
2. **E2**: GPU pin-access analysis.
3. **E3**: End-to-end differentiable detailed routing (DREAMPlace-style).
4. **E4**: RL net ordering + GPU parallel maze.
5. **E5**: Sweep-based detailed routing — extending GAMER's sweep primitive
   from coarse global routing into fine-grain detailed routing.

E5 was picked first as the cleanest extension of an established GPU primitive
(sweep) to the detailed regime, with a plausible path to working code on
Apple Silicon today. E1 is the planned next experiment after E5 reaches a
real-fixture milestone; it gates on CUDA hardware.

**Open strategic question:** when the CUDA-equipped box is back online, do we
pivot to E1 (cuOpt MILP) immediately or continue pushing E5 (preferred
direction, tile decomposition, whole-chip integration) on the same hardware?
Both are defensible — E1 is a clean second data-point on the same problem; E5
on CUDA is the cleanest scaling test for the sweep approach. Defer until the
hardware is actually back; the answer depends partly on how close E5 is to a
defensible "competitive with TritonRoute" claim at that moment.

### External prior art worth tracking

These are the repos whose work is most directly relevant to E5 and E1. Worth a
re-read before starting any related implementation.

| Repo | Why it matters |
|------|----------------|
| [`cuhk-eda/InstantGR`](https://github.com/cuhk-eda/InstantGR) | Current SOTA GPU global router (ICCAD 2024 / TCAD 2025). Reference for what state-of-the-art GPU routing looks like in 2025. |
| [`cuhk-eda/Xplace`](https://github.com/cuhk-eda/Xplace) | Where GGR / GAMER live — the sweep primitive this project extends. |
| [`cuhk-eda/dr-cu`](https://github.com/cuhk-eda/dr-cu), [`limbo018/dr-cu-rl`](https://github.com/limbo018/dr-cu-rl) | CPU detailed routers; the RL fork is relevant to E4. |
| [`The-OpenROAD-Project/OpenROAD`](https://github.com/The-OpenROAD-Project/OpenROAD) (`src/drt`) | TritonRoute — the detailed-router baseline we compare against. |
| [`NVIDIA/cuopt`](https://github.com/NVIDIA/cuopt) | The MILP backend for E1, when CUDA hardware returns. |

## Modules

### `gpu_pnr.sweep`

The grid-SSSP kernel. Implements 4-direction Bellman-Ford via Gauss-Seidel
fast sweeping — each outer iteration runs four directional axis sweeps
(H-forward, H-backward, V-forward, V-backward), then a polluted-mask cleanup.
The 3D variant adds two per-layer relaxation passes per iteration to handle
via transitions.

**Public API (2D):**
- `sweep_sssp(w, source, max_iters=200, check_every=8, seg_barrier=None) -> (d, iters)` — distance map.
- `backtrace(d, w, source, sink, atol=1e-5) -> path | None` — path reconstruction.
- `sweep_sssp_multi(w, sources, ...) -> (d, iters)` — K-source batched.

**Public API (3D):**
- `sweep_sssp_3d(w, source, via_cost=1.0, ...) -> (d, iters)` — multi-layer SSSP.
- `backtrace_3d(d, w, source, sink, via_cost=1.0, ...) -> path | None`.

**Implementation rationale.** Each per-axis sweep is one `cumsum` + one
`cummin`, not a Python loop over rows ([ADR 0002](adr/0002-scan-based-sweeps.md)).
Obstacles are handled by a mask-based segmented scan with a per-call
autotuned `SEG_BARRIER` ([ADR 0005](adr/0005-mask-based-segmented-scan.md)).
Convergence is checked every 8 iterations to amortise CPU↔GPU sync
([ADR 0003](adr/0003-async-convergence-check.md)). Via transitions in 3D
use sequential per-layer min relaxation, not a parallel scan — vias must
not chain through obstacles ([ADR 0006](adr/0006-sequential-via-relax.md)).

### `gpu_pnr.baseline`

Reference Dijkstra on the same grid model, hand-rolled with `heapq`. Used
for correctness ground-truth in tests and the benchmark script. Always runs
on CPU (operates on a numpy view of the input tensor). `dijkstra_grid_3d`
adds the via edges to the relaxation step and skips via transitions whose
destination is an obstacle, matching the kernel's "vias respect obstacles"
invariant.

### `gpu_pnr.router`

Sequential multi-net routing on top of `sweep`.

**Public API (2D):**
- `route_nets(w, [(source, sink), ...], reserve_pins=True) -> [NetResult, ...]`
- `NetResult(source, sink, path)` with `.routed` and `.length` properties.

**Public API (3D):**
- `route_nets_3d(w, [((l,r,c), (l,r,c)), ...], via_cost=1.0, reserve_pins=True) -> [Net3DResult, ...]`
- `Net3DResult` with the same shape but `(layer, row, col)` coordinates.

For each net in order: run `sweep_sssp` (or `sweep_sssp_3d`) from its source,
`backtrace` to its sink, mark every cell of the path as an obstacle for all
subsequent nets. Endpoints are reserved up-front as obstacles too — pins are
physical, not shareable.

Default ordering is HPWL-ascending ([ADR 0007](adr/0007-hpwl-ascending-net-ordering.md)).
Backtrace runs on `.cpu()` views to exploit Apple Silicon's unified memory
([ADR 0004](adr/0004-cpu-backtrace.md)).

## Data flow

```
RTL design (Hazard3) ──┐
                       │   [Phase 4 only — not in repo yet]
PDK (gf180mcuD)    ──┐ │
                     ▼ ▼
                  LibreLane ─────────► GR guides + LEF/DEF
                                              │
                                              │  (extract subblock,
                                              │   convert to grid + nets)
                                              ▼
                            (H, W) cost grid + [(src, sink), ...]
                                              │
                                              ▼
                                       route_nets()
                                              │
                                              │  per net:
                                              ├─► sweep_sssp() ──► distance map d
                                              │       │
                                              │       ▼
                                              ├─► backtrace(d, ...) ──► path
                                              │       │
                                              │       ▼
                                              └─► mark path cells as obstacles
                                              │
                                              ▼
                                  [NetResult, ...]  (per-net path or None)
```

## Where decisions live

Don't write decision rationale here. New choices about kernel structure, host
platform, obstacle handling, or routing strategy go in [`adr/`](adr/) as a new
ADR. Forward-looking implementation work goes in [`plans/`](plans/).

## Tested behaviours

`tests/test_sweep.py`:
- Open grid, unit weights — distances match Dijkstra.
- Open grid, random weights — distances match Dijkstra.
- Grid with line obstacles — paths route around.
- Unreachable region — produces `inf` consistently.
- Backtrace produces 4-connected paths avoiding obstacles.
- MPS and CPU produce equivalent results.

`tests/test_router.py`:
- Single net on open grid.
- Two nets — second routes around first; no cell overlap.
- Net forced to detour around first.
- Blocked endpoint returns `None`.
- Endpoint collision (B's pin lands on A's path) returns `None` for B.
- An unrouteable first net doesn't corrupt state for routeable second net.

`tests/test_ordering.py`:
- HPWL-ascending vs identity vs descending across nets counts.

`tests/test_sweep_3d.py`:
- Single-layer 3D matches existing 2D `sweep_sssp` exactly.
- `via_cost=0` collapses to "best of any layer" (matches 3D Dijkstra).
- High `via_cost` keeps paths on the source layer; the cheapest off-layer
  arrival is exactly one via.
- Layer-0 row obstacle forces a path to detour up to layer 1 and back.
- Random multi-layer grid agrees with 3D Dijkstra.
- Backtrace produces 4-connected-in-layer + via-only-cross-layer paths.
- Source on a fully-walled layer-0 still reaches its sink via layer 1.
- MPS and CPU produce equivalent 3D results.

`tests/test_router_3d.py`:
- Single net across layers on an open 3D grid.
- Two disjoint 3D nets.
- Net A occupies a horizontal stripe on layer 0; net B's natural path
  crosses it. With low `via_cost`, B detours via layer 1 (uses a via).
- Blocked-endpoint and endpoint-collision parity with 2D.
- Pin reservation across layers protects nets whose paths share an
  intermediate `(r, c)` cell.
- Layer-0 wall forces a route through layer 1.

`tests/test_hazard3_io.py`:
- Guide parser and post-DR DEF NETS parser invariants on the real fixture.
