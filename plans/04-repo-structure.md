# Repository Structure

## Current State

```
graph-research-private/
  CLAUDE.md                             # Canonical agent handoff (Claude Code, Cursor, etc.)
  AGENTS.md                             # Points to CLAUDE.md (interchangeable entry)
  .cursor/rules/agent-handoff.mdc       # Cursor: alwaysApply → same brief as CLAUDE.md
  requirements.txt                      # pip deps for local / agent environments
  SE Interim Report FINAL FINAL.pdf     # Interim report (Jan 17, 2026)
  graph_research_presentation.ipynb     # All-in-one Colab notebook (bugfixed)
  results/                              # Exported CSV/JSON from notebook (see README)
  plans/
    00-project-context.md               # Research question, setup, key results
    01-notebook-architecture.md         # Cell map, function reference, data flow
    02-bugs-and-fixes.md                # Bugs found + fixes applied + remaining issues
    03-future-development-plan.md       # 5-phase development roadmap
    04-repo-structure.md                # This file
    08-final-report-narrative.md        # How to write the final report
    09-dataset-analysis.md              # Dataset correctness, fixes, limitations
```

## Proposed Future Structure

As the project grows beyond a single notebook, refactor into:

```
graph-research-private/
  plans/                    # Context & plans (this folder)
  data/                     # Generated datasets / cached data
  src/
    dataset.py              # build_dataset(), split_data(), collate()
    masks.py                # sliding_window_mask(), graph_token_mask(), cache
    model.py                # Block, TinyModel
    train.py                # train_one(), eval_acc(), expand_mask_for_heads()
    experiment.py           # run_three(), grid_run()
    baselines/
      gcn.py                # GCN baseline (Phase 3)
      gat.py                # GAT baseline (Phase 3)
  notebooks/
    interim_experiments.ipynb     # Original experiments (cleaned up)
    phase1_hyperparam.ipynb       # Hyperparameter sweeps
    phase2_multihop.ipynb         # Higher k experiments
  tests/
    test_dataset.py         # Dataset generation correctness
    test_masks.py           # Mask shape, convention, correctness
    test_model.py           # Forward pass, output shapes
  reports/
    SE Interim Report FINAL FINAL.pdf
  requirements.txt
```

## When to Refactor

Refactor from single notebook to modular structure when:
- Adding GNN baselines (Phase 3) — need separate model files
- Running experiments outside Colab — need importable modules
- Collaborating with others — need proper package structure
