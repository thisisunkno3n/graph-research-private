# Final Report: Narrative Guide

**Purpose:** Notes on how to frame the notebook results in the final report so they tie back to the hypothesis clearly.

**Numbers:** Prefer metrics from **`results/`** (export cell at end of `graph_research_presentation.ipynb`) over stale tables in this repo or embedded notebook outputs. See **`CLAUDE.md`**, **`AGENTS.md`**, and **`results/README.md`**.

---

## Hypothesis (restate in report intro)

A Transformer whose self-attention is restricted to the edges of a known graph will:
1. Achieve higher accuracy than dense and window-mask attention on k-hop reachability
2. Filter out distractors more effectively (advantage grows with noise)
3. Use attention capacity more efficiently — achieving results with far less connectivity

---

## Core Results Section

After presenting the baseline `run_three()` numbers (k=3, distractors={20,40,80}, 5 seeds), interpret them against each hypothesis claim:

### Claim 1: Accuracy
- Graph-masked attention outperforms dense and window at every distractor level
- The margin widens as distractors increase (from ~1.3pp at d=20 to ~7pp at d=80 in the pre-fix numbers)
- Re-run with bug fixes may change absolute numbers — report whatever you get honestly
- If graph still wins: hypothesis confirmed. If not: analyze why (e.g., model too small, distractor fix changed difficulty)

### Claim 2: Distractor Filtering
- Key insight: as distractors increase, dense attention drowns in noise (100% connectivity means the model must learn to ignore irrelevant tokens through training alone)
- Window attention fails because sequence locality ≠ graph locality — nearby tokens in the edge list are not necessarily nearby in the graph
- Graph mask hard-codes the relevant structure, so the model only needs to learn *what* to compute, not *where* to look
- The scaling trend (graph advantage grows with noise) directly validates claim 2

### Claim 3: Efficiency
- Report the `allowed_pct` numbers alongside accuracy
- Graph mask uses ~2.5-7% connectivity vs 100% for dense
- This means ~93-97.5% of attention computations are eliminated
- Frame this as: "Graph masking achieves the best accuracy while using the least attention capacity, suggesting that structural sparsity is not just cheaper but actively beneficial"

---

## GCN Baseline: Why It Matters

Frame the GCN comparison around this question: **Does graph-masked attention offer anything beyond what a native GNN already provides?**

Key framing points:
- The GCN gets a structural advantage: it receives the raw adjacency matrix directly, while the Transformer must extract graph structure from a flat token sequence
- Despite this, if graph-masked Transformer matches or beats GCN, it suggests that the Transformer's global attention mechanism (even when sparse) adds value beyond simple message passing
- If GCN wins easily: the Transformer approach needs more work, but graph masking is still validated as the right *direction* (it makes the Transformer more GNN-like)
- If GCN also struggles: the task itself is genuinely hard, and both architectures face the same fundamental challenge (multi-hop information propagation)
- Discuss over-smoothing: GCN with k layers for a k-hop task may suffer from over-smoothing (all node representations converge). The Transformer avoids this via residual connections and layer norms within each block

---

## Multi-Hop Masks: What You're Testing

Frame around: **Is the bottleneck model depth or receptive field?**

The three configs compared by `run_multihop_comparison`:

| Config | Mask Hops | Layers | What It Tests |
|--------|-----------|--------|---------------|
| 1-hop + 4 layers | 1 | 4 | Standard approach: each layer propagates one hop |
| 2-hop + 2 layers | 2 | 2 | Trade depth for wider receptive field |
| 4-hop + 1 layer | 4 | 1 | Extreme: single layer sees full k-hop neighborhood |

Predictions to state:
- If 4-hop+1L ≈ 1-hop+4L: the bottleneck is information propagation, not computation depth. One layer with the right receptive field is enough.
- If 1-hop+4L >> 4-hop+1L: depth matters — the model needs multiple rounds of computation to transform information, not just access to it.
- If 2-hop+2L is best: there's a sweet spot between depth and receptive field.

Important caveat to discuss: k-hop masking "gives away" the answer structure by encoding exactly the path length needed. This is a controlled experiment about information propagation, not a claim of practical superiority. Acknowledge this explicitly.

Also note: as k_hops grows, the mask approaches dense attention (every node is within k hops in a well-connected graph). Report mask density at each k_hops value to show this convergence.

---

## Distractor Distribution Fix

This is a methodological contribution worth highlighting:

- **Original bug:** distractors were restricted to **non-path × non-path**, so path nodes had distinctive degree patterns (e.g. target appears only once as a destination). Heuristic baselines could beat random without multi-hop reasoning.
- **Fix:** distractors are now sampled from **any** directed node pair except true path edges (and self-loops), with verification that no **other** simple path of exactly *k* edges exists from start to the labeled target.
- **Why it matters:** this eliminates a potential confound and makes the results more credible. If graph masking still wins with realistic distractors, the core hypothesis is validated more strongly.
- Frame as: "We identified and corrected a confound in the distractor generation process that could have inflated accuracy for all methods via structural shortcuts."

---

## Attention Visualization

Use `plot_attention_comparison()` to show:
- Dense: diffuse attention everywhere (the model must learn what to ignore)
- Window: band-diagonal pattern (captures only local context, misses graph structure)
- Graph: sparse, structured pattern following the actual graph edges

This is a qualitative complement to the quantitative accuracy results. It directly illustrates *how* graph masking helps: by concentrating attention on structurally relevant tokens.

---

## Limitations to Acknowledge

1. **Synthetic task only.** Results on k-hop reachability may not transfer to real-world graph reasoning (knowledge graphs, molecular graphs, etc.)
2. **Small model.** TinyModel (d=128, 2 layers, 4 heads) is far from production scale. Larger models might close the gap between dense and graph-masked attention through sheer capacity.
3. **Known graph structure.** The graph mask requires knowing the graph at inference time. In many real applications, the graph structure is unknown or approximate.
4. **Computational overhead.** While graph masking reduces attention FLOPs, constructing the per-sample mask has its own cost. For small graphs this is negligible; for large graphs it could offset the savings.
5. **k-hop mask caveat.** The multi-hop mask encodes the exact path length needed, giving it an "unfair" advantage. It's a diagnostic tool, not a practical method.
6. **Shorter routes to the target.** The dataset generator forbids a *second* simple path of exactly *k* edges to the labeled target, but it does **not** forbid a shorter simple path (fewer than *k* edges) from start to that same target. The label is always the endpoint of the planted *k*-edge chain, not necessarily the unique node at graph distance *k*. State this clearly in the task definition / limitations (see `plans/09-dataset-analysis.md`).

---

## Suggested Report Structure

1. **Introduction** — Research question, motivation (GNN bottlenecks + Transformer quadratic cost), hypothesis
2. **Related Work** — Graphormer, Exphormer, GAT, FlashAttention (already in references)
3. **Method** — Synthetic dataset, encoding scheme, three attention strategies, model architecture
4. **Experiments & Results**
   - 4.1 Baseline comparison (dense/window/graph, multiple seeds)
   - 4.2 Distractor scaling (how advantage grows with noise)
   - 4.3 Hyperparameter sensitivity (one-at-a-time sweep)
   - 4.4 GCN baseline comparison
   - 4.5 Multi-hop mask analysis (optional, if results are interesting)
   - 4.6 Attention visualization (qualitative)
5. **Discussion** — Interpret results against hypothesis, limitations
6. **Future Work** — Higher k, Graphormer-style distance biases, real-world benchmarks, hybrid GNN+Transformer
7. **Conclusion** — One paragraph summarizing the key finding
