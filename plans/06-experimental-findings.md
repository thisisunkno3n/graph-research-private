# Experimental Findings Log

## Key Finding: Graph Masking Works for 1-Hop, Fails for Multi-Hop

### The Sparsity Paradox

| Mask Type | % Connections Used | k=1 Accuracy | k=3 Accuracy |
|-----------|-------------------|--------------|--------------|
| Dense (no mask) | 100% | 11.9% | ~4.5% |
| Window (w=32) | ~28-50% | 10.9% | ~4.0% |
| Graph (1-hop) | ~5-8% | **69.7%** | ~5.5% |
| Graph (3-hop) | ~80%+ | not tested | ~5.0% |
| Random baseline | — | 3.3% | 3.3% |

### Why 1-Hop Graph Masking Works (k=1)

For 1-hop reasoning, the task is: "which node does the start node point to?"

The 1-hop graph mask restricts attention to tokens encoding directly connected edges. This means:
- The [SEP]/query token can only attend to edge tokens touching the start node
- The model doesn't waste attention on irrelevant distractor edges
- The answer (the V of the edge where U=start) is directly visible in one attention step
- **Sparsity IS the feature** — fewer connections = less noise = stronger signal

Result: 70% accuracy (21x random). Train ~80%, val ~70% — healthy generalization.

### Why Multi-Hop Fails (k=3)

For 3-hop reasoning, the task is: "follow start→A→B→target through the graph."

**Attempt 1: 1-hop mask, 2 layers** → ~5.5%
- The model can only see direct neighbors per layer
- Needs to chain: find start→A edges, then find A→B edges, then find B→target edges
- 2 layers isn't enough depth for 3 compositional steps from shuffled tokens
- The model can't learn to "relay" information through intermediate nodes

**Attempt 2: 4 layers (more depth)** → ~3.7%
- More layers should allow more hops of information flow
- But the model still can't learn to chain sparse attention steps
- Deeper model + sparse gradients = harder to train, not easier

**Attempt 3: 3-hop mask** → ~5.0%
- Expands the mask so tokens within 3 edges can attend to each other
- **Problem: on a 30-node graph with 43 edges, most nodes are within 3 hops**
- The 3-hop mask is ~80%+ dense — it collapses back to nearly the dense condition
- Loses the sparsity advantage that made 1-hop work

### The Fundamental Tension

```
Narrow mask (1-hop):  Sparse, focused, BUT can't see multi-hop paths
Wide mask (k-hop):    Can see paths, BUT loses sparsity → becomes dense
More layers:          Should chain hops, BUT can't learn the pattern from shuffled tokens
```

This is a **Goldilocks problem** — no mask setting is "just right" for multi-hop:
- Too narrow → can't reach the answer
- Too wide → too much noise (same as no mask)
- The "correct" mask would highlight only the specific path edges, but that requires knowing the answer in advance (circular)

### Why the Model Was "Working" Before (Pre-Augmentation)

Before edge-order augmentation:

| Condition | Train Acc | Val Acc |
|-----------|-----------|---------|
| k=1, graph | 88.6% | 10.2% |
| k=3, graph | 85.0% | ~7.0% |

The model was **memorizing specific token sequences**, not learning graph reasoning:
- Same graph always had the same token order
- Model memorized "when I see token pattern X, output Y"
- 85%+ train accuracy with ~7% val accuracy = pure memorization
- Adding edge-order shuffling (augmentation) destroyed this shortcut
- After augmentation: train and val converge, but at ~5% (no actual learning for k≥2)

### After Augmentation (Corrected Results)

| Condition | Train Acc | Val Acc | Interpretation |
|-----------|-----------|---------|----------------|
| k=1, graph | ~80% | **69.7%** | Genuine learning, healthy generalization |
| k=1, dense | ~28% | 11.9% | Some learning, but weak without structure |
| k=3, any condition | 3-7% | 3-5% | No learning for any method |

### Conclusions

1. **Graph-masked attention provides a powerful structural inductive bias** — 70% vs 12% at k=1 is definitive
2. **The benefit comes from sparsity** — forcing attention onto relevant edges eliminates noise from distractors
3. **Multi-hop reasoning requires compositional attention** that small transformers cannot learn from shuffled token sequences
4. **The encoding matters** — shuffled edge tokens make multi-hop compositionally hard because the model must first reconstruct graph topology from linear sequence before reasoning over it
5. **Overfitting was masking a deeper issue** — pre-augmentation results showed apparent learning that was actually memorization

### Implications for Future Work

- **Positional graph encodings** (Graphormer-style distance biases) may help by explicitly encoding hop distance
- **Hierarchical attention** — 1-hop mask in early layers, wider mask in later layers
- **Different tokenization** — adjacency matrix input instead of edge list might help multi-hop
- **Larger models** — may be needed for compositional reasoning (current: d=128, 2 layers)
- **GNN comparison** — GNNs handle multi-hop via message passing natively; comparing would show whether the bottleneck is the transformer or the task

### Experimental Conditions

All experiments run on Google Colab T4 GPU with:
- num_nodes=30, n=20000 (with edge-order augmentation)
- d_model=128, n_heads=4, dropout=0.2
- AdamW, LR warmup + cosine decay, label smoothing=0.05
- Early stopping with patience=10-15
