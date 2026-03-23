# Project Context: Graph-Masked Self Attention For Multi-Hop Reasoning

**Author:** Felix Jeong
**Affiliation:** Western University, Scholar's Elective 2200E
**Date:** January 17, 2026
**Keywords:** Transformers, Graph Neural Networks (GNNs), Sparse Attention

## Core Research Question

Can a Transformer with a graph-structured attention mask improve multi-hop relational reasoning on graph data, compared to dense or window attention?

## Approach

Combine GNNs and Transformers by restricting self-attention to follow graph edges (adjacency-derived mask). This aims to:
- Get the Transformer's long-range capability
- Avoid GNN bottlenecks (over-smoothing, over-squashing)
- Reduce the Transformer's quadratic attention cost via graph-like sparsity

## Hypothesis

A Transformer whose self-attention is restricted to the edges of a known graph will:
1. Achieve higher accuracy than dense and window-mask attention
2. Filter out distractors more effectively
3. Use attention capacity more efficiently with less connectivity

## Experimental Setup

- **Synthetic dataset:** Directed graphs, 30 nodes, unique k-hop path (v0 -> v1 -> ... -> vk), plus distractor edges
- **Encoding:** Each edge (u,v) as `[U, u, V, v, E]`; query as `[SEP, S, start, Kk]`; label = target node vk
- **Model:** TinyModel — token + positional embeddings, L=2 Transformer blocks (d=128, 4 heads), linear head on [SEP] token
- **Training:** AdamW, lr=1e-3, cross-entropy loss, 80/20 train/val split, batch_size=64
- **Hardware:** Google Colab, NVIDIA T4 GPU

## Three Attention Conditions

| Condition | Description | Connectivity |
|-----------|-------------|-------------|
| Dense (baseline) | Full attention, every token attends to every token | 100% |
| Window (w=32) | Each token attends to +/- 32 positions | ~15-47% |
| Graph-masked | Adjacency-based boolean mask; only graph-neighbor tokens attend | ~2.5-7.2% |

## Key Results (k=3, averaged over 5 seeds)

### Validation Accuracy

| # Distractor Edges | Dense | Window | Graph |
|---------------------|-------|--------|-------|
| 20 | 4.80% | 5.47% | **6.07%** |
| 40 | 7.27% | 6.07% | **9.73%** |
| 80 | 21.80% | 5.60% | **28.73%** |

### Attention Connectivity

| # Distractors | Dense Allowed % | Window Allowed % | Graph Allowed % |
|---------------|-----------------|------------------|-----------------|
| 20 | 100.0 | 47.16 | 7.20 |
| 40 | 100.0 | 27.48 | 4.21 |
| 80 | 100.0 | 14.91 | 2.49 |

### Key Findings

1. Graph-masked attention wins at every distractor level; advantage grows with noise
2. Achieves this with only ~2.5-7.2% connectivity (massive efficiency gain)
3. Converges faster than dense (~0.305 peak vs ~0.237 at 40 distractors)
4. Window attention is consistently weak — sequence locality != graph locality

## References (from interim report)

- Alon & Yahav, 2020 — GNN bottleneck
- Topping et al., 2021 — Over-squashing via curvature
- Velickovic et al., 2017 — Graph Attention Networks (GAT)
- Ying et al., 2021 — Graphormer
- Mialon et al., 2021 — GraphiT
- Shirzad et al., 2023 — Exphormer (sparse graph transformers)
- Yuan et al., 2025 — Survey of graph transformers
- Dao et al., 2022 — FlashAttention
- Sun et al., 2025 — Efficient attention survey
- Besta et al., 2023 — Graph of Thoughts
- Duan et al., 2025 — Directed SSSP sorting barrier
- Zhang et al., 2025 — Recursive Language Models
