# ADR 0001 — PyTorch MPS as host platform

**Status:** Accepted (2026-05-09).

## Context

E5 (sweep-based detailed routing) needs a development host that supports the
core scan primitives (`cumsum`, `cummin`, `cummax`) on a parallel device today,
on the hardware actually present (Apple Silicon M-series). CUDA is unavailable
until the user's dedicated GPU box is restored.

Candidate stacks considered:

- **PyTorch MPS** — `cumsum`/`cummin` already supported on MPS in 2.11; unified
  memory makes CPU↔device transfers metadata-only; large ecosystem; brings
  autograd if E3 (differentiable DR) ever happens.
- **JAX with Metal backend** — primitives present but the Metal plugin lagged
  behind CUDA on coverage and stability at the time of evaluation.
- **MLX** — Apple's native ML framework; clean API but a smaller ecosystem and
  no scan-axis primitive equivalent to `cummin` at the time.
- **Raw Metal Performance Shaders / Compute kernels** — full control but a much
  bigger build for what amounts to a research prototype; loses portability to
  CUDA later.

## Decision

1. Use **PyTorch 2.11+ with MPS** as the host for all kernel work.
2. Keep the kernel **device-agnostic** at the API level: tensors carry their own
   `.device`; the routing code does not branch on `mps` vs `cpu` vs (future)
   `cuda`. The only Apple-Silicon-specific code path is the CPU-side backtrace
   (see ADR 0004), which would still work elsewhere — just slower.
3. **Run the test suite on CPU** so correctness is reproducible on any host;
   only benchmarks select MPS.

## Consequences

- Trades raw throughput vs hand-tuned Metal for ecosystem speed: scan primitives
  are immediately usable, no kernel authoring required.
- Locks the project to Python + PyTorch as the host language for the
  foreseeable future. A CUDA port (E1, cuOpt) inherits this — fine, since cuOpt
  has Python bindings.
- `torch.compile` on MPS in 2.11 gives a modest 10–20% speedup (`inductor`
  slightly ahead of `aot_eager`); not enough to gate progress on, but a free
  gain when wired in.

## Walk-back options

- **If MPS coverage regresses or scan primitives lose performance** in a future
  PyTorch release — pin to the last good version, or write a minimal scan in
  Metal Performance Shaders called via `torch.utils.cpp_extension`.
- **If the CUDA box returns and we want pure-CUDA throughput numbers** — keep
  the kernel as-is (PyTorch CUDA is the same source) and re-bench. No
  architectural change needed.

## Links

- [`../architecture.md`](../architecture.md) — module overview.
- [ADR 0002](0002-scan-based-sweeps.md) — the scan-based sweep choice these
  primitives enable.
- [ADR 0004](0004-cpu-backtrace.md) — the unified-memory optimisation.
