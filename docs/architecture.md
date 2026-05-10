# Architecture

## Modules

### `gpu_pnr.sweep`

The grid-SSSP kernel. Implements 4-direction Bellman-Ford via Gauss-Seidel
"fast sweeping" — each outer iteration runs four directional axis sweeps
(H-forward, H-backward, V-forward, V-backward), then masks polluted distances
back to `inf`. The 3D variant adds two inter-layer relaxation passes per
iteration to handle via transitions.

**Public API (2D):**
- `sweep_sssp(w, source, max_iters=200, check_every=8) -> (d, iters)` — distance map.
- `backtrace(d, w, source, sink, atol=1e-5) -> path | None` — path reconstruction.
- `sweep_sssp_multi(w, sources, ...) -> (d, iters)` — K-source batched (Phase 2.3a).

**Public API (3D, Phase 3.4):**
- `sweep_sssp_3d(w, source, via_cost=1.0, ...) -> (d, iters)` — multi-layer SSSP.
- `backtrace_3d(d, w, source, sink, via_cost=1.0, ...) -> path | None`.

**Key implementation notes:**

Each per-axis sweep is implemented as **one cumsum + one cummin**, not a Python
loop over rows/cols. The forward-sweep recurrence
```
d_new[j] = min over k<=j of (d[k] + sum w[k+1..j])
```
expands to
```
d_new = cumsum(w) + cummin(d - cumsum(w))
```
which dispatches as a parallel scan on GPU rather than N sequential kernel
launches. Backward sweep = flip → forward → flip.

**Obstacle handling (Phase 3.1, mask-based segmented scan; SEG_BARRIER
autotuned in Phase 3.2):** `cumsum` runs on `w_clean = where(obstacle, 0, w)`
so magnitudes stay proportional to real path weight. A separate
`seg_id = cumsum(obstacle_mask)` per axis identifies which segment (maximal
run of non-obstacle cells) each cell belongs to. The cummin input is offset
by `seg_id * seg_barrier`, making earlier-segment values larger than
current-segment values so cummin can never pick across a segment boundary.
The offset is subtracted back exactly at the output. `seg_cw[j]`
(cumsum-from-current-segment-start) is computed as
`cw - cummax(cw_at_obstacle_positions)`.

When an entire current segment is unreachable (`d=inf` everywhere), cummin
falls back to the prior segment's running minimum and the reconstruction
shifts that value by `(S-S')*seg_barrier`. The result is a finite-but-large
polluted distance, so a final `d > seg_barrier/2` mask returns it to `inf`.

**Autotuned `seg_barrier`:** the Phase 3.1 module constant `SEG_BARRIER=2e4`
worked for synthetic 5%-obstacle-density grids but broke on real per-net
guides with ~93% obstacle density (per-row obstacle counts of ~1000 push
`seg_id*SEG_BARRIER` past `1.85e7` where float32 ULP corrupts distances of
order 1000 -- see `docs/phase32_spike.md`). `_autotune_seg_barrier(w, mask)`
in `sweep.py` picks a per-call value as the geometric mean of
`[2*max_legit_distance, FLOAT32_PRECISION_BUDGET/max_seg_id]`. Cost: ~3 GPU
syncs per sweep call (~1.5ms on MPS), <2% of typical sweep time. If the
constraint range is empty (workload exceeds the float32 precision budget),
autotune falls back to the upper bound and the polluted-mask threshold
becomes incorrect -- documented behavior; this is what 8192^2 unit-weight
hits.

Float32 precision budget: legit distances must stay under `seg_barrier/2`.
With autotune covering both regimes, the new wall is at grids where
`max_legit_distance * 2 * max_seg_id > FLOAT32_PRECISION_BUDGET = 1e7`. For
unit-weight grids that's around 8000 per side at 5% obstacle density.

**Async pipelining:** Convergence is checked every `check_every=8` iterations
(via `torch.equal`, which forces a CPU↔GPU sync) rather than per-iter. K
iters can run async between syncs.

**3D extension (Phase 3.4):** `sweep_sssp_3d` operates on a `(L, H, W)` cost
tensor. Per outer iteration:

1. The four intra-layer sweeps reuse `_sweep_forward`/`_sweep_backward` with
   `axis=2` and `axis=1`. These helpers already vectorise over a leading
   batch dim, so all `L` layers scan in parallel.
2. After `_mask_polluted`, a sequential per-layer min relaxation along
   `axis=0` propagates via transitions: `d[l] = min(d[l], d[l±1] + via_cost)`,
   followed by an obstacle re-mask so vias never land on or chain through
   blocked cells.

A naive cumsum-cummin scan along `axis=0` (analogous to the horizontal
sweeps with `via_cost * arange(L)` as the offset) would be a single parallel
pass but would let vias "pass through" intermediate obstacles — the scan
adds `via_cost*|Δl|` regardless of whether intermediate cells exist. The
sequential form costs `2(L-1)` min/where ops per iter and is correct under
obstacles. For typical L=4-12 (real ASIC stacks), this is a small fraction
of total per-iter cost.

**Edge model:** horizontal arrival at `(l,r,c)` pays `w[l,r,c]`; via
arrival pays only `via_cost` (no double-charge of the destination cell's
wire cost). This keeps the model symmetric with the 2D edge semantics
(every "edge" has its own weight; cells aren't independently priced).

### `gpu_pnr.baseline`

Reference Dijkstra on the same grid model, hand-rolled with `heapq`. Used
for correctness ground-truth in tests and the benchmark script. Always runs
on CPU (operates on numpy view of the input tensor). `dijkstra_grid_3d`
(Phase 3.4) adds the via edges to the relaxation step and skips via
transitions whose destination is an obstacle, matching the kernel's
"vias respect obstacles" invariant.

### `gpu_pnr.router`

Sequential multi-net routing on top of `sweep`.

**Public API (2D):**
- `route_nets(w, [(source, sink), ...]) -> [NetResult, ...]`
- `NetResult(source, sink, path)` with `.routed` and `.length` properties.

**Public API (3D, Phase 3.4):**
- `route_nets_3d(w, [((l,r,c), (l,r,c)), ...], via_cost=1.0, reserve_pins=True) -> [Net3DResult, ...]`
- `Net3DResult` with the same shape but `(layer, row, col)` coordinates.

For each net in order: run `sweep_sssp` (or `sweep_sssp_3d`) from its
source, `backtrace` to its sink, mark every cell of the path as an
obstacle for all subsequent nets. Endpoints become obstacles too —
pins are physical, not shareable. The 3D variant treats `(l, r, c)`
triples as distinct cells; pins on different layers at the same `(r, c)`
are different pins, and pin reservation works per-cell.

**Apple-Silicon-specific:** backtrace runs on the CPU view of the tensor
(`d.cpu()`, `w.cpu()`). On unified memory this is metadata-only and avoids
the per-cell `.item()` syncs that would otherwise dominate at 1024²+ grids.

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

## Key design choices

| Choice | Rationale |
|---|---|
| **PyTorch MPS** as host | Most ecosystem; `cumsum`/`cummin` already supported on MPS; cheap CPU↔MPS swap on unified memory; brings autograd if E3 (differentiable DR) ever happens. |
| **Scan-based sweeps** | Python-loop sweeps did 1024 kernel launches per axis. Scan does 1. Got us from 3× *slower* than CPU Dijkstra to 9.5× faster. |
| **`INF_PROXY` instead of NaN-handling** | Cleanest way to keep `cumsum` finite; the precision wall is documented and bounded. Mask-based obstacles is the Phase 2 escape. |
| **Sequential routing for Phase 1** | Demonstrates the kernel works end-to-end. Sweep-sharing (parallel multi-source) is the natural next step but a much bigger build. |
| **CPU backtrace** | Apple Silicon unified memory makes `.cpu()` near-free; per-cell `.item()` sync would otherwise be the bottleneck. |

## Tested behaviours

`tests/test_sweep.py`:
- Open grid, unit weights — distances match Dijkstra.
- Open grid, random weights — distances match Dijkstra.
- Grid with line obstacles — paths route around.
- Unreachable region — produces `inf` consistently.
- Backtrace produces 4-connected paths avoiding obstacles.
- MPS and CPU produce equivalent results (within float32 sum-order tolerance).

`tests/test_router.py`:
- Single net on open grid.
- Two nets — second routes around first; no cell overlap.
- Net forced to detour around first.
- Blocked endpoint returns `None`.
- Endpoint collision (B's pin lands on A's path) returns `None` for B.
- An unrouteable first net doesn't corrupt state for routeable second net.

`tests/test_sweep_3d.py`:
- Single-layer 3D matches existing 2D `sweep_sssp` exactly.
- `via_cost=0` collapses to "best of any layer" (matches 3D Dijkstra).
- High `via_cost` keeps paths on the source layer; the cheapest off-layer
  arrival is exactly one via.
- Layer-0 row obstacle forces a path to detour up to layer 1 and back.
- Random multi-layer grid agrees with 3D Dijkstra (atol=5e-3, absorbing
  float32 sum-order drift between cumsum-based scans and edge-by-edge
  Dijkstra accumulation on longer paths).
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
