# Architecture

## Modules

### `gpu_pnr.sweep`

The 2D-grid SSSP kernel. Implements 4-direction Bellman-Ford via Gauss-Seidel
"fast sweeping" — each outer iteration runs four directional axis sweeps
(H-forward, H-backward, V-forward, V-backward), then masks polluted distances
back to `inf`.

**Public API:**
- `sweep_sssp(w, source, max_iters=200, check_every=8) -> (d, iters)` — distance map.
- `backtrace(d, w, source, sink, atol=1e-5) -> path | None` — path reconstruction.

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

**Obstacle handling:** `inf` weights would make `cumsum` produce NaN, so we
substitute `INF_PROXY = 1e4` and post-mask any cell with `d > INF_PROXY/2`
back to `inf`. The proxy magnitude is bounded by float32 ULP: MPS doesn't
support float64, and `(cumsum + cummin)` near `proxy * N` loses precision
when `N * proxy > ~4e6`. This caps Phase 1 at grids of ~2048 per side.

**Async pipelining:** Convergence is checked every `check_every=8` iterations
(via `torch.equal`, which forces a CPU↔GPU sync) rather than per-iter. K
iters can run async between syncs.

### `gpu_pnr.baseline`

Reference Dijkstra on the same grid model, hand-rolled with `heapq`. Used
for correctness ground-truth in tests and the benchmark script. Always runs
on CPU (operates on numpy view of the input tensor).

### `gpu_pnr.router`

Sequential multi-net routing on top of `sweep`.

**Public API:**
- `route_nets(w, [(source, sink), ...]) -> [NetResult, ...]`
- `NetResult(source, sink, path)` with `.routed` and `.length` properties.

For each net in order: run `sweep_sssp` from its source, `backtrace` to its
sink, mark every cell of the path as an obstacle for all subsequent nets.
Endpoints become obstacles too — pins are physical, not shareable.

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
