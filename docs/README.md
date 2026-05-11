# Documentation index

This project follows the [four-document discipline](https://robtaylor.github.io/claude-project-discipline/):
**ADRs** for why-we-chose-this, **plans** for what's-next, **spikes** for
did-this-idea-work, **handoffs** for what's-in-flight-right-now (ephemeral).
See the top-level [`CLAUDE.md`](../CLAUDE.md) for the conventions in detail.

## Design narrative

- **[architecture.md](architecture.md)** — modules, public APIs, data flow,
  links to the ADRs that justify the design.
- **[results.md](results.md)** — measured throughput, scaling, and routing
  outcomes.

## Decisions, plans, spikes, handoffs

- **[adr/](adr/)** — accepted architecture decision records.
- **[plans/](plans/)** — long-lived plan documents for in-flight workstreams.
- **[spikes/](spikes/)** — resolved time-boxed investigations.
- **[handoffs/](handoffs/)** — ephemeral session-bridge notes. Empty when no
  session has left work mid-flight.

## How the discipline works here

Quick reference; full version in [`../CLAUDE.md`](../CLAUDE.md) and
[`handoff-discipline.md`](handoff-discipline.md):

- New choice that future-you will need to remember the *why* of → new ADR
  under [`adr/NNNN-…md`](adr/).
- Forward-looking work, in-order, with exit criteria → update or add a plan
  under [`plans/`](plans/).
- "I want to validate X before committing to ADR Y" → new spike under
  [`spikes/`](spikes/), time-boxed.
- Session ends with work in flight → write a handoff under
  [`handoffs/`](handoffs/) using [`handoffs/_template.md`](handoffs/_template.md);
  delete it (fold-then-`git rm`) when the work resolves.
