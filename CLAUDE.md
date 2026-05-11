# gpu-pnr

GPU-accelerated place-and-route experiments, targeting the **detailed routing**
stage of ASIC physical design. Apple Silicon (Metal/MPS) for development; CUDA
(cuOpt + custom kernels) for production-scale runs once that hardware is
available.

The current active experiment is **E5: sweep-based detailed routing** —
extending the GAMER sweep primitive into the detailed-routing regime, with
PyTorch MPS as the host. See [`docs/adr/0001-pytorch-mps-host.md`](docs/adr/0001-pytorch-mps-host.md)
for the platform choice, [`docs/architecture.md`](docs/architecture.md) for the
overview, and [`docs/plans/`](docs/plans/) for what's next.

## Quick commands

```sh
uv sync                                      # install deps
uv run pytest tests/                         # full test suite (43 tests)
uv run python scripts/demo.py --size 1024    # single-net demo
uv run python scripts/bench_scaling.py       # scaling sweep
```

## Project memory discipline

This project uses the four-document discipline from
<https://robtaylor.github.io/claude-project-discipline/>. Four kinds of
project memory, each with a clear home and a clear lifecycle:

| Doc kind | Lives in | Lifetime | Answers |
|---|---|---|---|
| **Architecture Decision Record (ADR)** | `docs/adr/NNNN-*.md` | Forever; amended in place; never deleted | *Why* did we choose this? |
| **Plan** | `docs/plans/<topic>.md` | Long-lived; updated as work lands | *What's next, in what order?* |
| **Spike** | `docs/spikes/<topic>.md` | Forever, marked Resolved | *Did this idea work?* |
| **Handoff** | `docs/handoffs/<topic>-handoff.md` | Ephemeral — folded into the others, then deleted | *What's in flight right now?* |

The load-bearing rule: **information has exactly one home, and the
handoff is the only doc that gets deleted.** Everything that survives
a session migrates from the handoff into an ADR / plan / spike / design
doc *before* the handoff file is removed.

### Where things go

| If you're about to write… | It belongs in… |
|---|---|
| "We chose X over Y because Z" | A new ADR, or an existing one's Decision/Consequences section |
| "What needs to happen next, in what order, with exit criteria" | The relevant plan doc under `docs/plans/` |
| "I want to validate <assumption> before committing to ADR NNNN" | A new spike at `docs/spikes/<topic>.md`, time-boxed |
| "Here's what's in flight right now and what the next session should pick up" | A handoff at `docs/handoffs/<topic>-handoff.md` |
| "Here's how subsystem X works internally" | `docs/architecture.md` or a topic-specific design doc |
| "Here's the measured behaviour of the system" | `docs/results.md` |

### Smell tests

If you notice these, the discipline has been broken — fix it before
proceeding:

- **More than one file in `docs/handoffs/`** (excluding `_template.md`) → previous handoff didn't fold-and-delete. Migrate its content to the right home, then `git rm`.
- **An ADR contains step-by-step "first do X, then do Y"** → that's a plan, not a decision. Move to `docs/plans/`.
- **A handoff contains "we chose X over Y because Z"** → that's a decision, not ephemeral context. Move to an ADR.
- **A plan doc hasn't been updated in months but the work has clearly evolved** → update it in place; past states live in `git log`, not stacked inside the doc.

### Handoff write/read protocol

**At session end (work in progress, partial state):** write a handoff
at `docs/handoffs/<topic>-handoff.md` using the template at
`docs/handoffs/_template.md`. Goal, next-up, done-this-session,
open-follow-ups, critical-context, verification-command. Skip if the
session ended at a clean stopping point.

**At session start (resuming work):** read the single handoff under
`docs/handoffs/`. Run its verification command first to confirm
the claimed state matches reality. Then proceed with its
"next session should pick up" item.

**At resolution:** migrate every load-bearing piece of the handoff
into its proper home (ADR / plan / spike / design doc), then
`git rm docs/handoffs/<topic>-handoff.md` in the same commit as the
migration. Commit message: `docs: resolve <topic> handoff — fold
into <where-it-went>`.

The migration table is in [`docs/handoff-discipline.md`](docs/handoff-discipline.md).

### Override for skill toolkit defaults

Various Claude Code skill toolkits (`create_handoff`, `resume_handoff`,
etc.) default to YAML format under `thoughts/shared/handoffs/` with
database indexing. **That doesn't apply here.** Override:

- Format: markdown.
- Location: `docs/handoffs/<topic>-handoff.md`.
- No database indexing.

The skill activation is informational; this project's convention takes
precedence.

### When NOT to write a handoff

- Session ended at a clean stopping point (everything merged, all
  decisions documented in ADRs/plans, nothing surprising left over).
- Work was a single small commit with no follow-ups.
- The plan doc already captures what's next at sufficient detail.

### When NOT to write an ADR

- Mechanical changes (renames, typo fixes, dependency bumps).
- Choices the code itself makes self-evident.
- Things adequately captured by a commit message.

Test: *if someone asked "why is it like this?" six months from now,
would the answer be in the code, the commit, or this ADR?* Only write
the ADR if the answer is "the ADR".
