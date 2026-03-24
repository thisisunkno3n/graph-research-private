# Agent handoff — Graph research project

**Single source of truth** for **Claude Code**, **Cursor**, and similar tools.  
**Cursor:** `.cursor/rules/agent-handoff.mdc` (`alwaysApply`) points here.  
**Other tools:** [`AGENTS.md`](./AGENTS.md) is a stub that points to this file.

Read this first, then pull numbers from `results/` and narrative from `plans/`.

## What this repo is

**Graph-masked self-attention** on a synthetic **k-hop reachability** task: compare **dense**, **sliding-window**, and **graph-structured** attention (plus optional GCN baseline) on the same dataset.

- **Main artifact:** `graph_research_presentation.ipynb` — full experiments (intended for **GPU**, e.g. Colab).
- **Context / write-up:** `plans/` markdown (especially `00`, `08`, `09`, `01`).

## What to read (in order)

| File | Why |
|------|-----|
| `plans/00-project-context.md` | Research question, hypothesis, setup (**tables may predate dataset fixes** — trust `results/` when present) |
| `plans/08-final-report-narrative.md` | How to frame results for the final report |
| `plans/09-dataset-analysis.md` | Dataset rules, distractor fix, viz caveats, limitations (e.g. shorter paths) |
| `plans/01-notebook-architecture.md` | Cell map, function names, data flow |
| `plans/02-bugs-and-fixes.md` | Known bugs, optimizations, remaining caveats |
| `results/README.md` | Expected CSV/JSON filenames and column meanings |

## Feeding experiment results into an agent

1. Run the notebook (Colab or local GPU) through the cells that produce **`df`** (`grid_run()`), **`rows`** (multi-seed table), and optionally **`shortcut_baselines`**.
2. Run the **“Export results”** cell at the bottom of `graph_research_presentation.ipynb` (writes under `results/`).
3. **Commit** `results/*.csv` / `results/*.json` and push, **or** paste file contents into the chat.
4. Ask the agent to update prose, slides, or `plans/08-final-report-narrative.md` using those files as **source of truth** for numbers.

Do **not** rely on notebook **outputs** embedded in the `.ipynb` JSON for authoritative metrics — prefer exported `results/`.

## Local setup (optional)

```bash
python3 -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Training is heavy; smoke-test imports only unless you have a CUDA GPU.

## Conventions

- **Notebook name:** `graph_research_presentation.ipynb` (not `*_interim*`).
- **Mask convention:** boolean mask, `True` = **blocked** in `F.scaled_dot_product_attention`-style usage (see notebook).
- **Edge order in tokens:** shuffled — never treat the first *k* decoded edges as the true path (see `query_start_and_k_from_x` in the notebook).

## If the user asks to “run the full experiment” in-agent

Prefer **not** to execute full `grid_run()` / multi-seed loops unless they explicitly request it and have GPU time. Default: use **exported `results/`** or a **small** `n`, `epochs`, and `fast=True` in `run_three()` for smoke tests.
