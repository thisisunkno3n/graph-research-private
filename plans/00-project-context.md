# Project Context: Graph-Structured Attention Masking as an Inductive Bias for Transformer Reasoning

**Author:** Felix Jeong
**Affiliation:** Western University, Scholar's Elective 2200E
**Date:** January 17, 2026 (updated March 24, 2026)
**Keywords:** Transformers, Graph Neural Networks (GNNs), Sparse Attention, Inductive Bias

## Revised Title

"Graph-Structured Attention Masking as an Inductive Bias for Transformer Reasoning"

## Core Research Question

Does restricting Transformer self-attention to graph-adjacent elements improve reasoning accuracy on graph-structured tasks? And can this principle generalize beyond graph-specific datasets?

## Approach

Use graph adjacency as an attention mask within a standard Transformer architecture. This tests whether known relational structure can serve as an effective inductive bias — forcing the model to attend only along structurally relevant connections. This aims to:
- Demonstrate that structural sparsity improves Transformer reasoning
- Provide a proof-of-concept for graph-derived attention masking
- Motivate application to any domain with known relational structure

## Long-Term Vision

While current experiments use synthetic graph datasets, the underlying principle is general: any data with known relational structure can provide a graph-derived attention mask for Transformers. Two future directions:

1. **Impose graph masks on existing data:** Take structured data (knowledge bases, protein interaction networks, document citation graphs) and use the relational structure as an attention mask on a standard Transformer.
2. **Infer graph structure from flat data:** Take sequential data (text, code) and infer a graph (dependency parsing, co-reference, call graphs), then apply graph-masked attention.

These extensions are out of scope for the current study but motivate the research direction.

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

## Key Results (Updated March 2026 — with edge-order augmentation)

**Note:** Earlier interim results (Jan 2026) showed modest improvements but were confounded by memorization. Edge-order augmentation revealed the true picture below.

### Headline Result: k=1 (single-hop reasoning)

| Method | Val Accuracy | vs Random (3.3%) |
|--------|-------------|-------------------|
| Dense | ~12% | 3.6x |
| Window | ~11% | 3.3x |
| **Graph-masked** | **~70%** | **21x** |

### Multi-Hop Results: k=2, k=3

| k | Dense | Window | Graph |
|---|-------|--------|-------|
| 1 | 11.9% | 10.9% | **69.7%** |
| 2 | 6.8% | 5.9% | 8.6% |
| 3 | ~4.5% | ~4.0% | ~5.5% |

All methods degrade to near-random at k≥2. Multi-hop compositional reasoning remains an open challenge.

### Attention Connectivity

| Mask Type | % Connections Used |
|-----------|--------------------|
| Dense | 100% |
| Window (w=32) | ~28-50% |
| Graph (1-hop) | ~5-8% |

### Key Findings

1. **Graph-masked attention achieves 6x the accuracy of dense at k=1** — the strongest result
2. Achieves this with only ~5-8% connectivity (massive efficiency gain)
3. Converges faster than dense or window conditions
4. **Multi-hop reasoning fails for ALL methods** — the bottleneck is compositional, not informational
5. The "sparsity paradox": expanding mask radius to k-hop approaches dense connectivity, losing the sparsity advantage
6. **Methodological finding:** without edge-order augmentation, models memorize token sequences (85% train, 7% val) rather than learning graph structure. Augmentation was necessary for valid generalization measurement.

### Implications for Broader Application

The core finding — that graph-derived attention sparsity acts as a powerful inductive bias — suggests potential application beyond graph-specific tasks. Any domain with known relational structure (knowledge graphs, molecular interaction networks, document discourse graphs) could benefit from graph-structured attention masking in Transformers.

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
