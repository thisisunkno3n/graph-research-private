# Repository Structure

## Current State

```
graph-research-private/
  SE Interim Report FINAL FINAL.pdf    # Interim report (Jan 17, 2026)
  graph_research_presentation.ipynb     # All-in-one Colab notebook (bugfixed)
  plans/
    00-project-context.md               # Research question, setup, key results
    01-notebook-architecture.md         # Cell map, function reference, data flow
    02-bugs-and-fixes.md                # Bugs found + fixes applied + remaining issues
    03-future-development-plan.md       # 5-phase development roadmap
    04-repo-structure.md                # This file
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
