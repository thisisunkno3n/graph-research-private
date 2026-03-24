# Agent instructions (interchangeable)

**Canonical project brief:** [`CLAUDE.md`](./CLAUDE.md)

Claude Code, Cursor, and other assistants should **read `CLAUDE.md` first** for:

- Research question, notebook location, plan index  
- How to use **`results/`** exports as source of truth for metrics  
- Mask conventions, dataset caveats, when **not** to run full training in-agent  

Cursor loads the same expectations via **`.cursor/rules/agent-handoff.mdc`** (`alwaysApply`).

If a tool only surfaces this file: open **`CLAUDE.md`** in the repo root — it is the single source of truth.
