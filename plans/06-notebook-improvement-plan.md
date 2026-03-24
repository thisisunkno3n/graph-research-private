# Notebook Improvement Plan

**Created:** 2026-03-23
**Target:** graph_research_presentation.ipynb
**Constraint:** Google Colab T4, solo 2nd-year CS student, must be explainable

---

## Phase 0: Documentation & API Reference

**PyTorch APIs used in this notebook:**
- `torch.nn.MultiheadAttention(embed_dim, num_heads, dropout, batch_first)` — accepts `attn_mask` as `(B·num_heads, L, L)` bool or float additive mask
- `torch.nn.functional.cross_entropy(input, target, label_smoothing)` — label smoothing 0.0–1.0
- `torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)` — custom schedule
- `torch.optim.AdamW(params, lr, weight_decay)` — default weight_decay=0.01
- `networkx.has_path(G, source, target)` — BFS reachability check (needed for Phase 2)

**Current notebook structure (21 cells):**
- Cells 0-1: Imports & config
- Cells 2-3: Dataset generation (`build_dataset`)
- Cells 4-5: Visualization
- Cells 6-7: Train/val split (`split_data`, `collate`)
- Cells 8-9: Attention masks (`sliding_window_mask`, `graph_token_mask`)
- Cells 10-11: Model (`Block`, `TinyModel`)
- Cells 12-13: Training (`train_one`, `eval_acc`, `expand_mask_for_heads`)
- Cells 14-15: Experiment runner (`run_three`)
- Cells 16-17: Grid sweep & plotting (`grid_run`, `plot_hist`, `plot_best_so_far`)
- Cells 18-19: Multi-seed experiment
- Cell 20: Analysis plots

**Current hyperparameters (never tuned):**
- d_model=128, n_heads=4, n_layers=2, dropout=0.1
- lr=1e-3, epochs=20, batch_size=64, patience=5
- label_smoothing=0.05, warmup_epochs=3
- n=4000, num_nodes=30, k=3, distractors=40, window=32

---

## Phase 1: Re-run with Bug Fixes (Baseline)

**Goal:** Get clean baseline numbers with all bug fixes applied.

**What to do:**
1. Run the multi-seed experiment (Cell 19) as-is — no code changes needed
2. Run `grid_run()` (Cell 17) as-is
3. Record results — these replace the interim report numbers

**Verification:**
- [ ] Multi-seed experiment completes without OOM (cache clearing works)
- [ ] Results dict contains `_L` key (grid_run fix works)
- [ ] Compare to interim numbers — note differences in the notebook markdown

**Runtime:** ~30 minutes on T4

**Anti-patterns:** Do NOT change any hyperparameters yet. This is a clean re-run only.

---

## Phase 2: Fix Distractor Distribution

**Goal:** Remove structural confound where distractors always go non-sink → sink.

**Why this is critical:** If the model wins because it learned "distractors look different structurally," the core hypothesis isn't validated. This is the most important experiment.

**What to change in `build_dataset()` (Cell 3):**

Current code (approximate):
```python
sinks = [n for n in range(num_nodes) if n not in non_sinks]
cand = [(u, s) for u in non_sinks for s in sinks if (u, s) not in true_edges]
```

New code:
```python
non_path = [n for n in range(num_nodes) if n not in path_nodes]
cand = [(u, v) for u in non_path for v in non_path if u != v and (u, v) not in true_edges]
```

**Additional safety check — no distractor creates an alternative k-hop path:**
```python
import networkx as nx
G = nx.DiGraph()
G.add_edges_from(true_edges + selected_distractors)
# Verify: only one simple path of length k from start to target
# Use nx.has_path or BFS to confirm no shortcut exists
```

**Verification:**
- [ ] Distractors connect arbitrary non-path nodes (not just non-sink → sink)
- [ ] No distractor creates alternative k-hop path (BFS check passes)
- [ ] Run multi-seed experiment with fixed distractors
- [ ] Compare graph-masked accuracy vs dense — does graph still win?
- [ ] Document whether accuracy drops for all methods (expected)

**Runtime:** ~2-3 hours implementation + debugging, ~30 min run

**Watch out for:**
- Candidate pool may be smaller for low num_nodes — check `len(cand) >= distractors`
- BFS check adds generation time but is fast for 30 nodes

---

## Phase 3: Focused Hyperparameter Sweep

**Goal:** Find better hyperparameters. One variable at a time, not full grid.

**PREREQUISITE — Thread `d_model`, `n_heads`, `dropout` through the call chain:**

Currently these are hardcoded as TinyModel defaults and not exposed as parameters of `train_one()` or `run_three()`. Before sweeping, modify:

1. **`train_one()` (Cell 13)** — add `d_model=128, n_heads=4, dropout=0.1` params, pass to TinyModel:
   ```python
   def train_one(method, num_nodes, vocab, V, train_loader, val_loader,
                 n_layers=2, d_model=128, n_heads=4, dropout=0.1,  # ← ADD THESE
                 window=32, epochs=20, lr=1e-3, ...):
       ...
       model = TinyModel(V, num_nodes, sep_id=vocab["SEP"],
                         d_model=d_model, n_heads=n_heads,  # ← PASS THROUGH
                         n_layers=n_layers, dropout=dropout).to(device)
   ```

2. **`run_three()` (Cell 15)** — add same params, forward to `train_one()`:
   ```python
   def run_three(k=3, distractors=40, num_nodes=30, n=4000,
                 n_layers=2, d_model=128, n_heads=4, dropout=0.1,  # ← ADD
                 epochs=20, lr=1e-3, window=32, ...):
       ...
       model, best_va, final_full, allowed_pct, hist = train_one(
           method, num_nodes, vocab, V, train_loader, val_loader,
           n_layers=n_layers, d_model=d_model, n_heads=n_heads,  # ← FORWARD
           dropout=dropout, window=window, epochs=epochs, lr=lr, ...)
   ```

**Then implement the sweep — new cell after Cell 17:**

```python
def hp_sweep(base_config, param_name, values, seeds=[42]):
    """Sweep one hyperparameter while holding others fixed."""
    results = []
    for val in values:
        config = {**base_config, param_name: val}
        for seed in seeds:
            r = run_three(**config, seed=seed)
            for method in ["dense", "window", "graph"]:
                results.append({
                    param_name: val, "seed": seed, "method": method,
                    "best_acc": r[method]["best"], "allowed_pct": r[method]["allowed_pct"]
                })
    return pd.DataFrame(results)
```

**Sweep order (each with k=3, distractors=40, 1 seed first):**
1. `d_model`: {64, 128, 256} — keep n_heads=4 (all divide evenly)
2. `n_layers`: {2, 3, 4}
3. `dropout`: {0.0, 0.1, 0.2}
4. `n_heads`: {2, 4, 8} — must divide d_model

**After finding best combo:** Confirm with 5 seeds.

**Verification:**
- [ ] Each sweep produces a DataFrame with results per value
- [ ] Plot accuracy vs parameter value for each method
- [ ] Best config confirmed with 5-seed run
- [ ] GPU memory stays within T4 limits (watch d_model=256)

**Runtime:** ~25 min for sweeps + ~30 min for 5-seed confirmation

**Watch out for:**
- d_model=256 with n_heads=8 may be tight on T4 memory
- If d_model=256 OOMs, reduce batch_size to 32

---

## Phase 4: Dataset Size Scaling

**Goal:** Test whether accuracy scales with more data.

**What to do — run with best config from Phase 3:**
- n = {4000, 10000, 20000}
- k=3, distractors=40, fixed hyperparams, 1 seed

**What to implement — use `hp_sweep` from Phase 3:**
```python
df_size = hp_sweep(best_config, "n", [4000, 10000, 20000])
```

**Verification:**
- [ ] Accuracy plotted vs dataset size for each method
- [ ] Document whether graph-masked advantage grows, shrinks, or stays constant
- [ ] n=20000 completes without OOM (L=424, ~34MB)

**Runtime:** ~10-15 min total

---

## Phase 5: Attention Visualization (for presentation)

**Goal:** Generate attention heatmaps showing where each method focuses.

**What to implement — new cell after Cell 20:**

```python
def get_attention_weights(model, x, method, vocab, num_nodes, window=32):
    """Extract attention weights from first layer for visualization."""
    model.eval()
    with torch.no_grad():
        h = model.tok(x) + model.pos(torch.arange(x.size(1), device=x.device))
        h = model.drop(h)
        # Build mask same way as training
        if method == "dense":
            mask = None
        elif method == "window":
            mask = sliding_window_mask(x.size(1), window, x.device)
        else:  # graph
            mask = graph_token_mask(x, vocab, num_nodes, x.device)
            mask = expand_mask_for_heads(mask, model.n_heads)
        # Get attention weights from first block
        block = model.blocks[0]
        attn_out, attn_weights = block.attn(h, h, h, attn_mask=mask, need_weights=True)
    return attn_weights  # shape: (B, L, L) averaged over heads
```

Then plot 3 heatmaps side-by-side: dense attention, window attention, graph attention — on the same input graph. Overlay the true path edges to show graph mask focuses there.

**Verification:**
- [ ] Heatmaps render for all 3 methods
- [ ] True path edges are visibly highlighted in graph-masked attention
- [ ] Export as PNG for presentation slides

**Runtime:** Minutes (single forward pass)

**Watch out for:**
- `need_weights=True` returns averaged weights by default; use `average_attn_weights=False` for per-head view
- Heatmap may be large (L×L) — crop to relevant tokens or subsample

---

## Phase 6 (Stretch): Scale to k=4, k=5

**Goal:** Test whether graph masking helps for deeper reasoning.

**What to change:**
- `run_three(k=4, n_layers=4, num_nodes=50)` — match layers to hops
- `run_three(k=5, n_layers=5, num_nodes=50)`

**Key insight:** With only 2 layers, the model can't propagate info across a 5-hop path. Need n_layers >= k.

**Verification:**
- [ ] k=4 graph-masked accuracy > random chance (>2%)
- [ ] Training loss decreasing (model is learning something)
- [ ] Document where performance degrades

**Runtime:** ~30 min per k value

---

## Phase 7 (Stretch): Multi-Hop Masks

**Goal:** Let attention see k-hop neighborhoods instead of just 1-hop.

**What to implement — new function in Cell 9:**

```python
def khop_graph_token_mask(x, vocab, num_nodes, device, k_hops=2):
    """Graph mask allowing attention within k hops (not just direct neighbors)."""
    # Build adjacency matrix A from edge tokens
    # Compute A + A^2 + ... + A^k_hops (reachability within k hops)
    # Convert to boolean mask
    ...
```

**Experiments to run:**
- Compare: {1-hop mask, 4 layers} vs {2-hop mask, 2 layers} vs {4-hop mask, 1 layer}
- Report mask density for each — multi-hop masks approach dense as k grows

**Verification:**
- [ ] Mask density increases with k_hops (expected)
- [ ] Compare accuracy across mask/layer combos
- [ ] Document whether multi-hop mask helps or if stacking layers is sufficient

**Runtime:** ~2-3 hours implementation, ~1 hour experiments

---

## Execution Order

```
Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 5
                                              ↓
                                    Phase 6 and/or 7 (if time)
```

Phase 1 must come first (clean baseline). Phase 2 is most scientifically important. Phase 5 can be done anytime after Phase 1 (independent). Phases 6-7 are stretch goals.

---

## What NOT to Do

- Do not implement AMP/mixed precision — marginal gain, adds complexity
- Do not implement FlashAttention — requires custom CUDA, not worth it for this project
- Do not attempt real-world benchmarks (FB15k-237, CLRS) — scope too large
- Do not add virtual nodes or recursive architectures — hard to explain in Q&A
- Do not tune more than one hyperparameter at a time — confounds results
