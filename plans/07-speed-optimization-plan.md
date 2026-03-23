# Speed Optimization Plan

**Created:** 2026-03-23
**Goal:** ~3.5-4x speedup on T4 GPU without changing experimental results

---

## Phase 1: AMP (Automatic Mixed Precision)

**What:** Wrap training forward pass in `torch.cuda.amp.autocast()` with GradScaler.

**Where:** `train_one()` in Cell 15 (index 15 in notebook)

**Changes:**
1. Create `scaler = torch.cuda.amp.GradScaler()` before training loop
2. Wrap forward+loss in `with torch.cuda.amp.autocast():`
3. Use `scaler.scale(loss).backward()` instead of `loss.backward()`
4. Use `scaler.step(opt)` instead of `opt.step()`
5. Add `scaler.update()` after step
6. Also wrap `eval_acc()` forward pass in autocast

**Verification:**
- [ ] Training runs without NaN losses
- [ ] Results comparable to non-AMP baseline

---

## Phase 2: Fast Sweep Mode

**What:** Add `fast=True` parameter to `run_three()` that uses aggressive early stopping and reduced eval.

**Where:** `run_three()` in Cell 17 (index 17)

**Changes:**
- Add `fast=False` parameter to `run_three()`
- When `fast=True`: override epochs=10, patience=3, eval_every=3, eval_val_batches=5
- `hp_sweep()` and `grid_run()` should use `fast=True` by default
- Multi-seed experiment stays `fast=False` (final numbers need full precision)

**Verification:**
- [ ] `run_three(fast=True)` completes in ~40% of normal time
- [ ] `run_three(fast=False)` behavior unchanged

---

## Phase 3: Data Pipeline

**What:** Pin memory, add workers, and cache datasets.

**Where:** `run_three()` Cell 17, DataLoader creation

**Changes:**
1. `DataLoader(pin_memory=True, num_workers=2)` for both train and val loaders
2. Add dataset caching: `_DATASET_CACHE = {}` keyed by `(n, num_nodes, k, distractors, seed)`
3. `build_dataset()` checks cache before generating

**Verification:**
- [ ] DataLoaders use pin_memory and num_workers
- [ ] Second call with same params returns cached data instantly

---

## Phase 4: torch.compile

**What:** Compile model for faster inference on PyTorch 2.x.

**Where:** `train_one()` in Cell 15, after model creation

**Changes:**
- Add `model = torch.compile(model)` after `.to(device)`, guarded by version check
- Only compile if `torch.__version__ >= "2.0"` and device is cuda

**Verification:**
- [ ] Model compiles without error on Colab
- [ ] Falls back gracefully on CPU or older PyTorch
