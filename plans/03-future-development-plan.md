# Future Development Plan

Based on the interim report's "Future Direction" section and issues identified during code review.

---

## Phase 1: Improve Baseline Accuracy (Current Model)

**Goal:** Push k=3 accuracy significantly higher before scaling to harder tasks.

### Tasks

1. **Hyperparameter sweep**
   - Try d_model in {64, 128, 256}
   - Try n_heads in {2, 4, 8}
   - Try n_layers in {2, 3, 4}
   - Try dropout in {0.0, 0.1, 0.2}
   - Try label_smoothing in {0.0, 0.05, 0.1, 0.15}

2. **Increase dataset size**
   - Current: n=4000. Try n=10000, 20000, 50000
   - Monitor if validation accuracy scales with data

3. **Fix distractor distribution**
   - Current: distractors always go non-sink -> sink (structurally distinguishable)
   - New: allow distractors between any non-path nodes, ensuring no alternative k-hop path exists
   - This tests whether graph masking truly helps or just exploits structure

### Verification
- k=3, distractors=40: target >30% graph-masked accuracy (currently ~9.73%)
- Graph advantage should persist with improved distractor distribution

---

## Phase 2: Scale to Higher Hop Counts (k=4, 5+)

**Goal:** Make graph-masked attention work for deeper reasoning.

### Tasks

1. **Multi-hop masks**
   - Current mask allows 1-hop attention (direct neighbors only)
   - Implement k-hop mask: allow attention to nodes within k hops in the graph
   - This lets a single attention layer "see" multi-hop neighborhoods

2. **Deeper models**
   - Increase n_layers to match k (e.g., 4 layers for k=4)
   - Each layer can propagate info one hop, so L layers covers L-hop paths

3. **Distance-aware attention biases**
   - Instead of binary mask (attend/block), add soft bias based on graph distance
   - Closer nodes get stronger attention, distant nodes get weaker (not blocked)
   - Reference: Graphormer (Ying et al., 2021) shortest-path distance bias

4. **Scale graph size**
   - Increase num_nodes from 30 to 50, 100
   - Test whether graph masking's efficiency advantage grows with graph size

### Verification
- k=4: graph-masked accuracy significantly above random (>1/N = 2-3.3%)
- k=5: meaningful learning signal (training loss decreasing)

---

## Phase 3: Add GNN Baselines

**Goal:** Compare against proper GNN models to understand where graph-masked Transformers fit.

### Tasks

1. **Implement GCN baseline**
   - Standard Graph Convolutional Network with k message-passing layers
   - Same graph data, same train/val split

2. **Implement GAT baseline**
   - Graph Attention Network (Velickovic et al., 2017)
   - Compare learned attention patterns vs. our fixed graph mask

3. **Hybrid GNN + Transformer**
   - GNN encoder for local structure -> Transformer for global reasoning
   - Or: use GNN-derived attention weights as soft mask for Transformer

### Verification
- Side-by-side accuracy comparison: GCN vs GAT vs Dense Transformer vs Graph-Masked Transformer
- Compare convergence speed and parameter efficiency

---

## Phase 4: Advanced Architectures

**Goal:** Explore cutting-edge ideas from the literature.

### Tasks

1. **Graph-of-Thought style inference** (Besta et al., 2023)
   - Multi-step reasoning with intermediate graph states
   - Could graph masking help at each reasoning step?

2. **Recursive Language Models** (Zhang et al., 2025)
   - Iterative refinement for multi-hop reasoning
   - Test if graph-masked attention + recursion handles higher k

3. **Virtual nodes** (Shirzad et al., 2023 — Exphormer)
   - Add global virtual nodes that all tokens can attend to
   - Preserves sparsity while allowing long-range info flow

4. **FlashAttention integration** (Dao et al., 2022)
   - Implement graph-masked attention with FlashAttention kernels
   - Measure actual wall-clock speedup vs. dense FlashAttention

### Verification
- At least one architecture should handle k=5+ above random chance
- Document which approaches fail and why (negative results are valuable)

---

## Phase 5: Real-World Evaluation

**Goal:** Move beyond synthetic data.

### Tasks

1. **Knowledge graph reasoning**
   - Benchmark on FB15k-237 or WN18RR
   - Test if graph-masked attention helps for link prediction

2. **Algorithmic reasoning**
   - CLRS benchmark (Velickovic et al., 2022)
   - Graph algorithm tasks where structure is critical

3. **Network analysis**
   - Community detection, influence propagation
   - Tasks where attention should follow network structure

### Verification
- Competitive with existing methods on at least one benchmark
- Efficiency advantage (FLOPs, memory) documented

---

## Priority Order

| Phase | Priority | Estimated Effort | Depends On |
|-------|----------|-----------------|------------|
| 1 | **High** | 1-2 weeks | Nothing (start here) |
| 2 | **High** | 2-3 weeks | Phase 1 |
| 3 | **Medium** | 1-2 weeks | Phase 1 |
| 4 | **Medium** | 3-4 weeks | Phase 2 + 3 |
| 5 | **Low** | 4+ weeks | Phase 4 |
