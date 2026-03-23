# Bugs Found and Fixes Applied

**Date:** March 23, 2026

## Bugs Fixed

### Bug 1: `split_data()` mutates input list (Cell 7)

**Problem:** `random.shuffle(data)` shuffles the original list in-place. If `data` is reused after calling `split_data()`, the order is silently corrupted.

**Fix:** Added `data = list(data)` to copy before shuffling.

```python
# Before
def split_data(data, val_frac=0.2):
    random.shuffle(data)  # mutates original!
    ...

# After
def split_data(data, val_frac=0.2):
    data = list(data)  # copy to avoid mutating the original list
    random.shuffle(data)
    ...
```

---

### Bug 2: `grid_run()` dead config key (Cell 17)

**Problem:** `run_three()` returns `{"dense": {...}, "window": {...}, "graph": {...}}` but `grid_run` tried `r.get("config", {}).get("L", ...)`. The `"config"` key never existed, so `L` always fell back to the formula `5*(k+d)+4`.

**Fix:** `run_three()` now stores `"_L": L` in results dict. `grid_run` reads `r.get("_L", ...)`.

```python
# run_three() now returns:
results = {"_L": L}  # actual sequence length
results["dense"] = {...}
...

# grid_run() now reads:
L = r.get("_L", 5*(k + d) + 4)  # use actual L, formula as fallback
```

---

### Bug 3: `_GRAPH_MASK_CACHE` never cleared (Cell 9 + 15)

**Problem:** Global dict accumulates cached masks across different configurations. On Colab with limited RAM, this causes OOM during grid sweeps.

**Fix:** Added `clear_graph_mask_cache()` function. Called at start of each `run_three()`.

```python
def clear_graph_mask_cache():
    """Clear the mask cache to free memory between runs."""
    _GRAPH_MASK_CACHE.clear()

def run_three(...):
    clear_graph_mask_cache()  # free memory from previous runs
    ...
```

## Optimizations Applied

### Opt 1: Vectorized `sliding_window_mask()` (Cell 9)

**Before:** Python for-loop over L positions (~10-50x slower)
**After:** Single tensor broadcast operation

```python
# Before
for i in range(L):
    lo, hi = max(0, i-window), min(L, i+window+1)
    m[i, lo:hi] = False

# After
idx = torch.arange(L, device=device)
return (idx.unsqueeze(0) - idx.unsqueeze(1)).abs() > window
```

### Opt 2: `.expand()` instead of `.repeat()` (Cell 13)

**Before:** `.repeat()` copies memory for each head
**After:** `.expand().reshape()` uses views (no allocation)

```python
# Before
m.unsqueeze(1).repeat(1, h, 1, 1).view(B*h, L, L)

# After
m.unsqueeze(1).expand(B, h, L, L).reshape(B * h, L, L)
```

### Opt 3: Cached window mask in `eval_acc()` (Cell 13)

**Before:** Recreated `sliding_window_mask()` on every `eval_acc()` call
**After:** Accepts `cached_win` parameter; `train_one` passes pre-built mask

### Opt 4: LR warmup + cosine decay schedule (Cell 13)

**Before:** Flat learning rate throughout training
**After:** 3-epoch linear warmup -> cosine decay via `LambdaLR`

### Opt 5: Label smoothing enabled (Cell 13)

**Before:** `label_smoothing=0.0` with TODO comment
**After:** Default `label_smoothing=0.05` as function parameter

## Known Issues Not Yet Fixed

| Issue | Severity | Notes |
|-------|----------|-------|
| Distractor structure leak | Medium | Distractors always go non-sink -> sink; model may learn structural shortcut |
| No mixed precision (AMP) | Low | ~2x throughput gain possible on T4 with float16 |
| `max_len=512` not validated | Low | Current configs fit (max L=429) but no bounds check if params increase |
