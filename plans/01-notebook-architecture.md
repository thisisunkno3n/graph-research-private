# Notebook Architecture: graph_research_interim.ipynb

## Cell Map (21 cells)

| Cell | ID | Type | Purpose |
|------|----|------|---------|
| 0 | c1df530f | markdown | Title + overview |
| 1 | e7242abc | code | Imports, seed, GPU check |
| 2 | dcf1443e | markdown | Dataset design explanation |
| 3 | fc8bac06 | code | `build_dataset()` |
| 4 | 96e49ee6 | markdown | Visualization intro |
| 5 | c2391eb6 | code | `decode_edges_from_x()` + NetworkX viz |
| 6 | 31b180ee | markdown | Train/val split explanation |
| 7 | 93177cc4 | code | `split_data()`, `collate()`, loaders |
| 8 | 58e82f71 | markdown | Attention mask explanation |
| 9 | 811fd667 | code | `sliding_window_mask()`, `graph_token_mask()`, cache |
| 10 | e2d13f95 | markdown | Model architecture explanation |
| 11 | af6b90b2 | code | `Block`, `TinyModel` classes |
| 12 | d2c086be | markdown | Training loop explanation |
| 13 | bc26897c | code | `expand_mask_for_heads()`, `eval_acc()`, `train_one()` |
| 14 | 6c416614 | markdown | Experiment explanation |
| 15 | 6da8877e | code | `run_three()` |
| 16 | 595c76a0 | markdown | Note on graph allowed % |
| 17 | 7fbb319e | code | `plot_hist()`, `plot_best_so_far()`, `grid_run()` |
| 18 | bf6979df | markdown | Note on distractors=0 |
| 19 | YFdm3Aa5dezv | code | Multi-seed experiment (5 seeds x 3 distractor levels) |
| 20 | 61f428d1 | code | Extra runs + Pareto/surface/winner plots |

## Key Functions

### Data Pipeline
- **`build_dataset(n, num_nodes, k, distractors, seed, KMAX)`** — Generates synthetic k-hop graphs. Constructs path deterministically, adds distractor edges from non-path/non-sink -> sink nodes. Returns list of (tensor, target) tuples + vocab size + vocab dict.
- **`split_data(data, val_frac)`** — Copies then shuffles data, returns (train, val) split.
- **`collate(batch)`** — Stacks fixed-length sequences into (B, L) tensor + label tensor.
- **`decode_edges_from_x(x, vocab)`** — Reconstructs edge list from token sequence for visualization.

### Attention Masks
- **`sliding_window_mask(L, window, device)`** — Vectorized boolean mask, True=blocked. O(1) tensor op.
- **`graph_token_mask(x, vocab, num_nodes, device)`** — Per-batch (B, L, L) mask. Allows: same-block tokens, same-node-ID tokens across blocks, query<->all-node-IDs. Uses `_GRAPH_MASK_CACHE` for the static portion.
- **`clear_graph_mask_cache()`** — Frees cached masks between runs.
- **`expand_mask_for_heads(m, h)`** — Expands (B, L, L) -> (B*H, L, L) for MultiheadAttention.

### Model
- **`Block(d_model, n_heads, dropout)`** — Pre-norm Transformer block: LN -> MHA -> residual -> LN -> FFN -> residual.
- **`TinyModel(V, num_nodes, sep_id, d_model=128, n_heads=4, n_layers=2, max_len=512)`** — Token + positional embeddings -> stack of Blocks -> linear head on [SEP] hidden state.

### Training
- **`train_one(method, ...)`** — Full training loop. AdamW + linear warmup + cosine decay scheduler. Label smoothing=0.05. Early stopping with patience. Collects mask connectivity stats.
- **`eval_acc(model, loader, method, ...)`** — Validation accuracy with argmax. Accepts cached window mask.

### Experiments
- **`run_three(k, distractors, ...)`** — Runs dense/window/graph on same data. Clears mask cache between runs. Stores actual sequence length as `_L`.
- **`grid_run(ks, ds, ...)`** — Sweeps k x distractors grid, collects results into DataFrame.

## Data Flow

```
build_dataset() -> [(tensor, target), ...] -> split_data() -> train/val
    -> collate() -> DataLoader -> train_one()
        -> sliding_window_mask() or graph_token_mask()
        -> expand_mask_for_heads()
        -> TinyModel.forward(x, attn_mask)
        -> F.cross_entropy()
        -> eval_acc()
    -> run_three() collects results
    -> grid_run() sweeps parameters
```
