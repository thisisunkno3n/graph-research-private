# Notebook Architecture: graph_research_presentation.ipynb

## Cell Map (36 cells)

| Cell | ID | Type | Purpose |
|------|----|------|---------|
| 0 | c1df530f | markdown | Title + overview |
| 1 | e7242abc | code | Imports, seed, GPU check, plot style |
| 2 | dcf1443e | markdown | Dataset design explanation |
| 3 | fc8bac06 | code | `build_dataset()` |
| 4 | 96e49ee6 | markdown | Visualization intro |
| 5 | c2391eb6 | code | `decode_edges_from_x()` + NetworkX viz |
| 6 | 31b180ee | markdown | Train/val split explanation |
| 7 | 93177cc4 | code | `split_data()`, `collate()`, loaders |
| 8 | 58e82f71 | markdown | Attention mask explanation |
| 9 | 811fd667 | code | `sliding_window_mask()`, `graph_token_mask()`, cache |
| 10 | — | markdown | Multi-hop graph mask explanation |
| 11 | — | code | `khop_graph_token_mask()` |
| 12 | e2d13f95 | markdown | Model architecture explanation |
| 13 | af6b90b2 | code | `Block`, `TinyModel` classes |
| 14 | l7y8dgk18i9 | code | `SimpleGCN` class |
| 15 | d2c086be | markdown | Training loop explanation |
| 16 | bc26897c | code | `expand_mask_for_heads()`, `eval_acc()`, `train_one()` |
| 17 | abe2d91e | code | GCN infra: `tokens_to_graph()`, `collate_gcn()`, `eval_gcn_acc()`, `train_gcn()`, `run_with_gcn()`, plot helpers |
| 18 | 6c416614 | markdown | Experiment explanation |
| 19 | 6da8877e | code | `_DATASET_CACHE`, `get_dataset()`, `run_three()` |
| 20 | 595c76a0 | markdown | Note on graph allowed % |
| 21 | 7fbb319e | code | `plot_hist()`, `plot_best_so_far()`, `grid_run()` + initial run |
| 22 | — | markdown | Hyperparameter sweep intro |
| 23 | — | code | `hp_sweep()`, `plot_sweep()`, sweep configs |
| 24 | — | markdown | Attention visualization intro |
| 25 | — | code | `get_attention_weights()`, `plot_attention_comparison()` |
| 26 | — | markdown | Higher hop counts intro |
| 27 | — | code | Higher k experiment (commented out) |
| 28 | 8e0c1a22 | code | `run_multihop_comparison()`, `plot_multihop_comparison()` |
| 29 | — | markdown | GCN baseline comparison intro |
| 30 | cf9eb0e1 | code | GCN baseline experiment (commented out) |
| 31 | bf6979df | markdown | Note on distractors=0 |
| 32 | YFdm3Aa5dezv | code | Multi-seed experiment (5 seeds × 3 distractor levels) |
| 33 | — | markdown | Accuracy vs Heuristic Shortcuts intro |
| 34 | — | code | `measure_shortcut_baselines()`, `plot_accuracy_vs_shortcuts()` execution |
| 35 | 61f428d1 | code | Extra runs + Pareto/surface/winner plots |

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
