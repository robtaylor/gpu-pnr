# gpu-pnr

GPU-accelerated place-and-route experiments, targeting the **detailed
routing** stage of ASIC physical design. Apple Silicon (Metal / MPS) for
development today; CUDA (cuOpt + custom kernels) once that hardware is
available.

The current active experiment is **E5: sweep-based detailed routing** —
extending the GAMER sweep primitive (originally global-routing only) into
the detailed-routing regime, with PyTorch MPS as the host.

## Quick start

```sh
uv sync                                      # install deps (torch, numpy, pytest)
uv run pytest tests/                         # full test suite (43 tests)
uv run python scripts/demo.py --size 1024    # single-net demo
uv run python scripts/bench_scaling.py       # scaling sweep
```

## Repo layout

```
src/gpu_pnr/
├── sweep.py        sweep-SSSP kernel + naive backtrace (PyTorch, device-agnostic)
├── baseline.py     reference Dijkstra (CPU) for ground-truth comparison
├── router.py       sequential multi-net routing on top of sweep
└── ordering.py     net-ordering strategies for the router
tests/              pytest suite
scripts/            demo + benchmark scripts; Hazard3 spike drivers
docs/               see docs/README.md for the doc index
```

## Documentation

Start with [`docs/README.md`](docs/README.md). The project uses the
[four-document discipline](https://robtaylor.github.io/claude-project-discipline/)
— ADRs, plans, spikes, handoffs — described in [`CLAUDE.md`](CLAUDE.md).

- [Architecture overview](docs/architecture.md)
- [Results & benchmarks](docs/results.md)
- [Decisions (ADRs)](docs/adr/)
- [In-flight plans](docs/plans/)
- [Resolved spikes](docs/spikes/)

## Current status (2026-05-11)

Phases 1, 2, 3.1, 3.4 shipped; Phase 3.2 spike resolved. Next slice:
per-layer preferred routing direction. See
[`docs/plans/phase3-detailed-routing.md`](docs/plans/phase3-detailed-routing.md).
