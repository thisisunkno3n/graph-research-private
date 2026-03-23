# Feasible Future Directions: Prioritized Plan

**Date:** March 23, 2026
**Constraints:** Solo 2nd-year CS student, Google Colab T4, course project (not a thesis), must understand and explain everything in a 10-minute oral presentation.

---

## Guiding Principles

Before diving in, three rules for scoping decisions:

1. **The bug fixes come first.** The current numbers (6-29% accuracy) were produced with known bugs (split_data mutation, dead config key, memory leak). Re-running the existing experiments with fixes applied is prerequisite zero. Do this before any new direction.
2. **Depth over breadth.** One well-understood improvement with clear analysis beats five half-baked extensions. The presentation rubric rewards "forethought regarding potential extension" -- you can *discuss* ambitious ideas without having implemented all of them.
3. **Every experiment must have a hypothesis.** "I tried X" is weak. "I predicted X would help because Y, and the result was Z, which tells us W" is what makes a course project strong.

---

## Tier 1: Definitely Do

These are high-impact, feasible within days, and directly strengthen the core argument of the project.

### 1.1 Re-run Existing Experiments with Bug Fixes

- **What to implement:** Nothing new. Just re-execute the multi-seed experiment (Cell 19) with the patched notebook. Record new numbers for k=3, distractors={20,40,80}, 5 seeds.
- **Why it matters:** The current published numbers are unreliable. The split_data mutation bug means seeds were not truly independent. The LR warmup and label smoothing changes will also affect results. You need a clean baseline before anything else.
- **Complexity:** Trivial. Run the cell.
- **Watch out for:** Results may change significantly. If graph-masked accuracy drops, that is still a valid finding -- document it honestly. If it improves, great. Either way, the new numbers are your real baseline.
- **Time estimate:** ~30 minutes of Colab runtime.

### 1.2 Hyperparameter Sweep (Focused, Not Exhaustive)

- **What to implement:** Modify `run_three()` calls to test a small grid. Do NOT do a full combinatorial sweep (3x3x3x3x4 = 324 configs is way too many). Instead, test one variable at a time while holding others fixed:
  - d_model: {64, 128, 256} (3 runs)
  - n_layers: {2, 3, 4} (3 runs)
  - dropout: {0.0, 0.1, 0.2} (3 runs)
  - n_heads: {2, 4, 8} (3 runs, must divide d_model)
  - Keep k=3, distractors=40, n=4000, 1 seed for the sweep, then confirm best config with 5 seeds
- **Why it matters:** The current model (d=128, 4 heads, 2 layers, dropout=0.1) was never tuned. Even a modest sweep could reveal that the model is under-capacity or over-regularized. This is the lowest-effort path to better accuracy.
- **Complexity:** Easy. You already have `run_three()`. Just call it in a loop with different kwargs. Add the new kwargs (d_model, n_layers, dropout) as parameters if they are not already passed through.
- **Watch out for:**
  - d_model=256 with n=4000 should still fit on T4, but monitor GPU memory.
  - n_layers=4 with k=3 gives the model more depth than needed. If it helps, that is interesting (more capacity). If it hurts, that is also interesting (overfitting with small data).
  - Record training curves, not just final accuracy. A model that reaches 40% then crashes to 10% tells a different story than one that plateaus at 15%.
- **Time estimate:** ~12 one-at-a-time runs x ~2 min each = ~25 minutes, plus 5-seed confirmation.

### 1.3 Increase Dataset Size

- **What to implement:** Run the best config from 1.2 at n={4000, 10000, 20000}. 50k is likely unnecessary and slow on Colab.
  - `build_dataset(n=10000, ...)` and `build_dataset(n=20000, ...)`
  - Keep everything else fixed: k=3, distractors=40, same hyperparams
- **Why it matters:** With 4000 samples, 80/20 split gives only 3200 train / 800 val. For 30-class classification with noisy input, this is very small. If accuracy scales with data, it means the model architecture works but was data-starved. If it plateaus, the bottleneck is elsewhere.
- **Complexity:** Easy. One-line change to `build_dataset()`. Main cost is training time.
- **Watch out for:**
  - n=20000 with distractors=80 means L=424, batch_size=64, so each batch is 64x424 tokens. This fits on T4 but will be slower per epoch. Consider reducing epochs if validation loss plateaus early.
  - Memory: 20000 samples x 424 tokens x 4 bytes ~ 34 MB for data alone. Fine.
  - Use the same val_frac=0.2 so validation set also grows. More validation samples = more reliable accuracy estimates.
- **Time estimate:** ~10 minutes for 3 dataset sizes x 3 methods.

### 1.4 Fix Distractor Distribution

- **What to implement:** Modify `build_dataset()` so distractors are edges between arbitrary non-path nodes (not just non-sink -> sink). The constraint is: no new k-hop path from start to any node should be created (or at minimum, no alternative k-hop path to the correct target).
  - Current code: `cand = [(u, s) for u in non_sinks for s in sinks]` -- always points at sink nodes
  - New code: `cand = [(u, v) for u in non_path for v in non_path if u != v]` -- edges between any non-path nodes
  - Must verify no distractor creates an alternative k-hop path from v0. Simplest approach: after adding distractors, run BFS from v0 and confirm vk is the unique node reachable in exactly k hops.
- **Why it matters:** This is arguably the most important experiment in the project. If graph-masked attention only wins because distractor edges have a distinguishable structural pattern (always pointing to sink nodes), then the result is an artifact. If graph-masked attention still wins with realistic distractors, the core hypothesis is validated.
- **Complexity:** Medium. The generation logic needs rewriting and careful testing. You need to ensure:
  - No distractor edge is on the true path
  - No alternative k-hop path exists from start to target
  - The graph is still connected enough to be interesting
- **Watch out for:**
  - Generating valid graphs becomes harder. With unrestricted distractors, you might accidentally create alternative paths. The BFS verification step is essential.
  - Accuracy for ALL methods will likely drop (distractors are harder to distinguish). That is expected. The question is whether graph-masked still wins by a margin.
  - Run this comparison at the same settings (k=3, distractors=40, n=4000, same hyperparams) so it is directly comparable to your fixed baseline from 1.1.
- **Time estimate:** ~2-3 hours to implement and debug, ~30 minutes to run experiments.

---

## Tier 2: Stretch Goals (Do If Time Permits)

These are meaningful extensions that would strengthen the project but are not essential. Each one is a self-contained experiment you could add if your Tier 1 results are solid.

### 2.1 Scale to k=4 and k=5

- **What to implement:** Run `run_three(k=4, ...)` and `run_three(k=5, ...)` with the best config from Tier 1. Use n_layers >= k (e.g., 4 layers for k=4) since each Transformer layer can propagate information one hop.
- **Why it matters:** k=3 is the only hop count tested so far. Showing that graph-masked attention scales to deeper reasoning (or documenting exactly where it breaks) is a strong contribution. Even negative results ("graph masking helps at k=3 but fails at k=5") are valuable if analyzed.
- **Complexity:** Easy to run, medium to analyze. The code already supports arbitrary k. The challenge is that accuracy will drop and you need to determine whether the model is fundamentally limited or just needs tuning.
- **Watch out for:**
  - Sequence length grows: k=5 with 80 distractors = 5*(5+80)+4 = 429 tokens. Still within max_len=512.
  - With 2 layers and k=5, the model cannot propagate information across the full path in a single forward pass (each layer only sees 1-hop neighbors with graph masking). You MUST increase n_layers to at least k. This is a key insight to discuss.
  - Random chance is 1/30 = 3.3%. If your best method is at 4-5%, you are barely above chance. Report confidence intervals.
- **Time estimate:** ~30 minutes to run, significant time to analyze and interpret.

### 2.2 Multi-Hop Masks (k-Hop Attention)

- **What to implement:** Create a new mask function `khop_graph_token_mask(x, vocab, num_nodes, k, device)` that computes A + A^2 + ... + A^k (where A is the adjacency matrix), then uses the resulting matrix as the attention mask. Tokens attend to each other if their corresponding nodes are within k hops.
  - Compute adjacency matrix A from the edge list in x
  - Compute A_k = sum of A^i for i=1..k (matrix power, then binarize)
  - Build the token-level mask from A_k the same way graph_token_mask uses A
- **Why it matters:** This is the natural evolution of the core idea. 1-hop masking with 2 layers can theoretically reach 2-hop paths, but multi-hop masking lets a single layer see the full k-hop neighborhood. If this dramatically improves accuracy, it suggests the bottleneck is information propagation, not model capacity.
- **Complexity:** Medium. Matrix power on 30x30 is trivial computationally. The tricky part is correctly mapping the k-hop adjacency back to token-level masks using the existing `graph_token_mask` infrastructure.
- **Watch out for:**
  - As k grows, the k-hop mask approaches dense attention (every node is within k hops of every other node in a well-connected graph). Measure and report mask density at each k.
  - This gives the model an "unfair" advantage: the mask encodes exactly the path length needed. Discuss this explicitly -- it is a controlled experiment to measure information propagation, not a claim of general superiority.
  - Compare: {1-hop mask + 4 layers} vs {4-hop mask + 1 layer} vs {2-hop mask + 2 layers}. This is a clean experiment about depth vs. receptive field.
- **Time estimate:** ~2-3 hours to implement, ~1 hour to run experiments.

### 2.3 Distance-Aware Attention Biases (Graphormer-Style)

- **What to implement:** Instead of a binary mask (attend/block), add a learnable bias to attention logits based on shortest-path distance between nodes. Specifically:
  - Compute shortest-path distance matrix D (30x30) using BFS for each graph
  - Create a learnable embedding table: `dist_embed = nn.Embedding(max_dist+1, n_heads)`
  - In the attention computation, add `dist_embed[D[i,j]]` to the attention logits for tokens representing nodes i and j
  - Clip distances beyond a max (e.g., 8) to a single "far" bucket
- **Why it matters:** This is a softer version of graph masking. Rather than hard blocking, it biases attention toward nearby nodes. This tests whether the binary mask is too aggressive (blocking useful long-range signals) or whether the hard boundary is actually helpful.
- **Complexity:** Hard for a 2nd-year student. Requires modifying the attention computation inside `Block`, which currently uses `nn.MultiheadAttention` as a black box. You would need to either:
  - (a) Rewrite attention manually (educational but time-consuming), or
  - (b) Pre-compute the bias and pass it as `attn_mask` (hacky but possible since PyTorch MHA accepts float masks as additive biases)
  - Option (b) is feasible: instead of a boolean mask, pass a float tensor where 0.0 = neutral, large negative = blocked, and intermediate values encode distance.
- **Watch out for:**
  - This adds learnable parameters (the distance embedding). With n=4000 samples, overfitting is a real risk. Keep the embedding small.
  - Computing shortest-path distances per sample adds to data generation time but is fast for 30-node graphs (BFS is O(V+E)).
  - If this works well, it is a strong result connecting to the Graphormer literature. If it does not help over binary masking, that is also interesting (hard sparsity > soft bias for this task).
- **Time estimate:** ~3-4 hours to implement, ~1 hour to run.

### 2.4 GCN Baseline (Simple Version Only)

- **What to implement:** A basic 2-layer GCN using PyTorch Geometric (or hand-coded, which is simpler than it sounds for a GCN):
  - Node features: one-hot encoding of node ID (30-dim)
  - Message passing: H' = ReLU(A_norm @ H @ W) for each layer
  - Readout: take the embedding of the start node, project to 30 classes
  - Train with same loss, same data split
  - You do NOT need PyG. A GCN layer is literally one line: `H = relu(A_norm @ H @ W)`
- **Why it matters:** Without a GNN baseline, the paper cannot claim graph-masked Transformers are "better" or "different" from GNNs. Even a simple GCN comparison tells a story: if GCN wins easily, the Transformer approach needs more work. If GCN also struggles, the task is genuinely hard. If graph-masked Transformer wins, it validates the approach.
- **Complexity:** Medium. A from-scratch GCN is ~30 lines of PyTorch. The harder part is reformatting your data: currently stored as token sequences, you need to extract adjacency matrices and node features. Write a conversion function.
- **Watch out for:**
  - The data encoding is designed for Transformers (edge list as token sequence). For a GCN, you need the raw adjacency matrix. Either modify `build_dataset()` to also return the adjacency matrix, or write a decoder (you already have `decode_edges_from_x`).
  - A GCN with k layers theoretically has k-hop receptive field, matching the task. Use n_layers=k for a fair comparison.
  - Normalize the adjacency matrix (add self-loops, degree normalization) -- this is standard for GCN and takes 3 lines.
- **Time estimate:** ~3-4 hours to implement and debug, ~30 minutes to run.

### 2.5 Scale Graph Size to 50 Nodes

- **What to implement:** Run experiments with `num_nodes=50` instead of 30. Keep k=3, scale distractors proportionally (e.g., 40, 80, 160).
- **Why it matters:** With 30 nodes, random chance is 3.3%. With 50 nodes, it drops to 2%. If graph-masked attention maintains its absolute accuracy while random chance drops, the relative advantage grows. This tests whether graph sparsity becomes more valuable as graphs get larger.
- **Complexity:** Easy. Change one parameter. But sequences get longer: k=3, distractors=160, num_nodes=50 gives L = 5*(3+160)+4 = 819, which exceeds max_len=512. Either increase max_len or keep distractors at 80 (L=419, fits).
- **Watch out for:**
  - max_len=512 is a hard limit in TinyModel (positional embedding size). If L > 512, you get an index error. Either increase max_len or cap your distractor count.
  - More nodes = more classes = harder classification. Accuracy will drop for all methods.
  - Memory: graph_token_mask produces (B, L, L) boolean tensors. At L=400, B=64: 64x400x400 = 10M booleans = 10 MB per batch. Fine for T4.
- **Time estimate:** ~30 minutes to run, minimal code changes.

---

## Tier 3: Out of Scope (Discuss in Presentation, Do Not Implement)

These are ideas to mention in your "future work" slide (presentation criterion 6) but should NOT be attempted for this project.

### 3.1 Hybrid GNN + Transformer

- **Why out of scope:** Requires designing a two-stage architecture, implementing GNN encoding, handling the interface between GNN node embeddings and Transformer token embeddings, and tuning two sets of hyperparameters. This is a research paper's worth of work, not a course project extension.
- **What to say in presentation:** "A natural extension would be to use a GNN encoder to produce node embeddings, then feed those into the Transformer. This could combine the GNN's local message passing with the Transformer's global reasoning."

### 3.2 Graph-of-Thought / Recursive Models

- **Why out of scope:** These are complex inference-time strategies from recent papers (2023-2025). Implementing them correctly requires deep understanding of the original papers, careful engineering, and extensive debugging. The concepts are also not well-established enough to be "standard" implementations.
- **What to say in presentation:** "Iterative refinement approaches like Graph-of-Thought could potentially allow the model to reason over multiple passes, which might help for higher hop counts."

### 3.3 FlashAttention with Sparse Masks

- **Why out of scope:** FlashAttention's speed advantage comes from fused CUDA kernels that avoid materializing the full attention matrix. Integrating custom sparse masks with FlashAttention requires either (a) using block-sparse patterns that FlashAttention supports or (b) writing custom CUDA code. Neither is reasonable for a 2nd-year student on Colab.
- **What to say in presentation:** "For practical deployment at scale, integrating graph-structured masks with FlashAttention-style kernels would be necessary to maintain the wall-clock speedup that sparsity theoretically provides."

### 3.4 Real-World Benchmarks (FB15k-237, CLRS)

- **Why out of scope:** These benchmarks have complex data pipelines, established evaluation protocols, and strong existing baselines. Getting competitive results requires weeks of engineering and tuning. The synthetic task is the right testbed for the core hypothesis; real-world benchmarks are for a follow-up project.
- **What to say in presentation:** "Validating these findings on established benchmarks like FB15k-237 for knowledge graph reasoning, or the CLRS benchmark for algorithmic reasoning, would be important future work to establish practical relevance."

### 3.5 Virtual Nodes (Exphormer-Style)

- **Why out of scope:** Adding virtual nodes requires modifying the data encoding (add special "virtual" tokens), the masking logic (virtual nodes attend to everything), and the model architecture (handle virtual vs. real node outputs differently). While conceptually simple, the implementation touches every part of the pipeline.
- **What to say in presentation:** "Exphormer-style virtual nodes could provide a middle ground between full graph masking and dense attention, allowing global information flow through a small number of hub nodes."

---

## Recommended Execution Order

```
Week 1 (immediate):
  [1] Re-run with bug fixes (1.1)               -- 30 min
  [2] Hyperparameter sweep (1.2)                 -- 2-3 hours
  [3] Dataset size scaling (1.3)                 -- 1 hour
  [4] Fix distractor distribution (1.4)          -- 3-4 hours

Week 2 (if time permits, pick 1-2):
  [5] Scale to k=4, k=5 (2.1)                   -- 1-2 hours
  [6] Multi-hop masks (2.2)                      -- 3-4 hours
  [7] GCN baseline (2.4)                         -- 3-4 hours

Skip unless everything else is done:
  [8] Distance-aware biases (2.3)                -- 4-5 hours
  [9] Scale to 50 nodes (2.5)                    -- 1 hour
```

The most impactful pair from Tier 2 is **2.1 (higher k) + 2.2 (multi-hop masks)** because together they directly test the multi-hop reasoning claim. The GCN baseline (2.4) is most impactful for the presentation narrative ("how does this compare to existing approaches?").

---

## What to Present

For the 10-minute oral presentation, structure around these results:

1. **Baseline results** (1.1): Clean numbers with bug fixes. This is your foundation.
2. **Distractor fix** (1.4): "We identified and corrected a confound in the distractor distribution." This shows scientific rigor.
3. **Best result** from hyperparameter/data tuning (1.2 + 1.3): Your strongest accuracy numbers.
4. **One Tier 2 result** (whichever is most interesting): This shows depth.
5. **Future work slide** listing Tier 2 items you did not get to + Tier 3 items. This satisfies criterion 6 and shows you understand the broader landscape.

The presentation does NOT need to show every experiment. Pick the 3-4 most compelling results and explain them well.
