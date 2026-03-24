# Notebook Architecture: graph_research_presentation.ipynb

## Cell Map (40 cells)

| Cell | Type | Purpose |
|------|------|---------|
| 0 | markdown | Title + overview |
| 1 | code | Imports, seed, GPU, plot style |
| 2 | markdown | Dataset design |
| 3 | code | `build_dataset()` |
| 4 | markdown | Viz intro |
| 5 | code | `decode_edges_from_x`, `query_start_and_k_from_x`, sample graph plot |
| 6 | markdown | Train/val / loaders intro |
| 7 | code | `split_data()`, `collate()` |
| 8 | markdown | Attention masks intro |
| 9 | code | `sliding_window_mask()`, `graph_token_mask()`, cache |
| 10 | markdown | Multi-hop mask intro |
| 11 | code | `khop_graph_token_mask()` |
| 12 | markdown | Model intro |
| 13 | code | `Block`, `TinyModel` |
| 14 | code | `SimpleGCN` |
| 15 | markdown | Training loop intro |
| 16 | code | `expand_mask_for_heads()`, `eval_acc()`, `train_one()` |
| 17 | code | GCN infra, `run_with_gcn()`, plot helpers |
| 18 | markdown | `run_three` intro |
| 19 | code | `_DATASET_CACHE`, `get_dataset()`, `run_three()` |
| 20 | markdown | Note on graph allowed % |
| 21 | code | `plot_hist`, `plot_best_so_far`, `measure_shortcut_baselines`, `plot_accuracy_vs_shortcuts`, `grid_run` defs |
| 22 | code | Main experiment run + `plot_hist` / `plot_best_so_far` |
| 23 | markdown | HP sweep intro |
| 24 | code | `hp_sweep`, `plot_sweep` |
| 25 | markdown | Attention viz intro |
| 26 | code | `get_attention_weights`, `plot_attention_comparison` |
| 27 | markdown | Higher k intro |
| 28 | code | Higher k (commented `run_three` k=4/5) |
| 29 | code | `run_multihop_comparison`, `plot_multihop_comparison` |
| 30 | markdown | GCN comparison intro |
| 31 | code | GCN experiment (commented) |
| 32 | markdown | Note `distractors=0` |
| 33 | code | Multi-seed × distractors → `rows` |
| 34 | markdown | Shortcuts intro |
| 35 | code | `measure_shortcut_baselines` + `plot_accuracy_vs_shortcuts` |
| 36 | code | `pareto_plot`, `surface_plot`, `winner_heatmap`, `line_plot` defs |
| 37 | code | Run analysis plots (`df` from cell 22 `grid_run`) |
| 38 | markdown | **Export results** for agents (Claude Code / Cursor) / git |
| 39 | code | Write `results/*.csv`, `results/*.json` |

## Key Functions

### Data Pipeline
- **`build_dataset(n, num_nodes, k, distractors, seed, KMAX)`** — Generates synthetic k-hop graphs. Constructs path, adds distractor edges between **any** node pair (not just non-path), with BFS rejection of alternative k-hop paths. Returns list of (tensor, target) tuples + vocab size + vocab dict.
- **`split_data(data, val_frac)`** — Copies then shuffles data, returns (train, val) split.
- **`collate(batch)`** — Stacks fixed-length sequences into (B, L) tensor + label tensor.
- **`decode_edges_from_x(x, vocab)`** — Reconstructs edge list from token sequence for visualization (order matches shuffled blocks, **not** true-path order).
- **`query_start_and_k_from_x(x, vocab)`** — Reads `[SEP, S, start, Kk]` from the query tail so plots can recover the k-hop path without assuming `edges[:k]`.
- **`tokens_to_graph(x, vocab, num_nodes)`** — Converts token batch to (adj, start_node) for GCN.
- **`collate_gcn(batch, vocab, num_nodes)`** — Collate function producing GCN-format batches.

### Attention Masks
- **`sliding_window_mask(L, window, device)`** — Vectorized boolean mask, True=blocked.
- **`graph_token_mask(x, vocab, num_nodes, device)`** — Per-batch (B, L, L) 1-hop graph mask.
- **`khop_graph_token_mask(x, vocab, num_nodes, device, k_hops)`** — Per-batch k-hop reachability mask. Computes R = A + A² + ... + Aᵏ.
- **`clear_graph_mask_cache()`** — Frees cached masks between runs.
- **`expand_mask_for_heads(m, h)`** — Expands (B, L, L) → (B·H, L, L) for MultiheadAttention.

### Models
- **`Block(d_model, n_heads, dropout)`** — Pre-norm Transformer block.
- **`TinyModel(V, num_nodes, sep_id, ...)`** — Token + positional embeddings → Blocks → linear head on [SEP].
- **`SimpleGCN(num_nodes, d_hidden, n_layers, dropout)`** — Hand-coded GCN. Node embeddings → row-normalized message passing → start-node readout → linear head.

### Training
- **`train_one(method, ..., graph_mask_fn=None)`** — Transformer training loop. Accepts custom graph mask function.
- **`eval_acc(model, loader, method, ..., graph_mask_fn=None)`** — Transformer validation accuracy.
- **`train_gcn(num_nodes, train_loader, val_loader, ...)`** — GCN training loop with identical optimizer/scheduler.
- **`eval_gcn_acc(model, loader, ...)`** — GCN validation accuracy.

### Visualization
- **`get_attention_weights(model, x, method, ..., graph_mask_fn=None)`** — Extracts layer-1 attention weights for a single sample. Supports custom mask functions.
- **`plot_attention_comparison(models, sample_x, ..., graph_mask_fn=None)`** — Side-by-side heatmaps for dense/window/graph.

### Experiments
- **`run_three(k, distractors, ..., graph_mask_fn=None)`** — Runs dense/window/graph on same data. Accepts custom mask function for multi-hop experiments.
- **`run_with_gcn(..., gcn_layers=None)`** — Delegates to `run_three()` then adds GCN on same split.
- **`run_multihop_comparison(...)`** — Compares {1-hop+4-layer, 2-hop+2-layer, 4-hop+1-layer} configs.
- **`grid_run(ks, ds, ...)`** — Sweeps k × distractors grid.
- **`hp_sweep(base_config, param_name, values, ...)`** — One-variable-at-a-time hyperparameter sweep.

### Analysis
- **`measure_shortcut_baselines(num_nodes, k, distractors_list, ...)`** — Measures accuracy of non-learning heuristic shortcuts (degree-based, dst-only) on the dataset.
- **`plot_accuracy_vs_shortcuts(results_by_d, baselines, ...)`** — Bar chart comparing trained model accuracy against heuristic shortcuts per distractor level.

## Data Flow

```
build_dataset() → [(tensor, target), ...] → split_data() → train/val
    ├─ collate() → DataLoader → train_one()
    │       → sliding_window_mask() or graph_token_mask() or khop_graph_token_mask()
    │       → expand_mask_for_heads()
    │       → TinyModel.forward(x, attn_mask)
    │       → eval_acc()
    │   → run_three() collects results
    │   → run_with_gcn() delegates to run_three() + adds GCN
    │
    └─ collate_gcn() → DataLoader → train_gcn()
            → SimpleGCN.forward(adj, start_node)
            → eval_gcn_acc()
```
