# gpu-pnr

GPU-accelerated place-and-route experiments, targeting the **detailed routing** stage
of ASIC physical design. Apple Silicon (Metal/MPS) for development; CUDA (cuOpt + custom
kernels) for production-scale runs once that hardware is available.

The current active experiment is **E5: sweep-based detailed routing** — extending the
GAMER sweep primitive (originally global-routing only, CUHK 2021) into the
detailed-routing regime, with PyTorch MPS as the host.

## Repo layout

```
src/gpu_pnr/
├── sweep.py        sweep-SSSP kernel + naive backtrace (PyTorch, device-agnostic)
├── baseline.py     reference Dijkstra (CPU) for ground-truth comparison
└── router.py       sequential multi-net routing on top of sweep
tests/              pytest suite (correctness vs Dijkstra; multi-net no-overlap)
scripts/
├── demo.py             single-net demo with timings
├── demo_multinet.py    N-net synthetic demo
└── bench_scaling.py    throughput + speedup across grid sizes
docs/                this directory
```

## Quick start

```sh
uv sync                                    # install deps (torch, numpy, pytest)
uv run pytest tests/                       # run the test suite
uv run python scripts/demo.py --size 1024  # single-net demo
uv run python scripts/bench_scaling.py     # scaling sweep
```

## Document index

- **[architecture.md](architecture.md)** — modules, types, key design choices.
- **[results.md](results.md)** — Phase 1 benchmark findings.
- **[roadmap.md](roadmap.md)** — E5 origin, what's done, what's next.

## Current status (2026-05-10)

Phase 1 complete: working sweep-SSSP on PyTorch MPS, correctness validated against
Dijkstra, multi-net sequential routing demonstrated. Peak 9.5× speedup vs CPU
Dijkstra at 1024². Next: endpoint reservation → net ordering → sweep-sharing.
