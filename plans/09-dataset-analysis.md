# Dataset Analysis: Correctness and Fitness for Purpose

**Date:** March 23, 2026

---

## Summary Verdict

The dataset is **mechanically correct** (encoding works, decoding roundtrips, labels are accurate, distributions are uniform). A **historical** design flaw — distractors restricted to non-path nodes — created exploitable degree shortcuts; that flaw is **fixed** in the current notebook (`cand` is all node pairs minus true edges). Post-fix, heuristic shortcuts sit near random and the BFS “alternative k-hop path to target” check fires often enough to be meaningful.

**Viz note:** Any code that highlights the “true path” must **not** use the first *k* edges from `decode_edges_from_x()` (order is shuffled). Recover start and *k* from the query tail and trace the unique k-edge simple path in the decoded graph.

---

## What's Correct

### Encoding integrity
- Edge encoding `[U, src, V, dst, E]` roundtrips perfectly — `decode_edges_from_x` recovers all edges
- Query tail `[SEP, S, start, Kk]` correctly encodes start node and hop count
- Token IDs don't collide (node IDs 0-29, special tokens 30+)

### Label distribution
- Target nodes are uniformly distributed across all 30 nodes (~333 per node per 10,000 samples)
- Start-target pairs are also uniform (all 870 possible pairs appear)
- No positional bias: path edges are uniformly shuffled across sequence positions

### Graph mask alignment
- `graph_token_mask` correctly connects tokens that share the same node ID
- This creates the right message-passing chain: v0-src ↔ v1-dst ↔ v1-src ↔ v2-dst ↔ ... ↔ vk-dst
- Query tokens attend to all node-ID tokens bidirectionally
- Within-block tokens (same edge) attend to each other

---

## Historical issue (pre-fix): path-node isolation created structural shortcuts

*The following applied before distractors were allowed to touch path nodes. Kept for report context.*

### The problem

Distractors are drawn exclusively from `non_path × non_path`:

```python
non_path = [v for v in nodes if v not in path_set]
cand = [(u, v) for u in non_path for v in non_path if u != v]
```

This means **no distractor edge touches any path node**. Path nodes {v0, v1, ..., vk} only appear in the k true path edges.

### Why it matters: degree-based shortcuts

Since path nodes are structurally isolated, they have distinctive degree signatures:

| Node | In-degree (src count) | Out-degree (dst count) | Total appearances |
|------|----------------------|------------------------|-------------------|
| v0 (start) | 1 | 0 | 1 |
| v1, ..., v(k-1) | 1 | 1 | 2 |
| vk (target) | 0 | 1 | 1 |
| Non-path (avg) | ~1.5 | ~1.5 | ~3.1 (at d=40) |

The target node appears **exactly once** in the entire edge list (as a destination), while non-path nodes appear ~3x on average. A model can exploit this without any multi-hop reasoning.

### Measured shortcut accuracy (no reasoning needed)

| Distractors | Random | "Pick dst-only node" | "Pick lowest degree" | Reported graph-masked |
|-------------|--------|---------------------|---------------------|----------------------|
| 20 | 3.3% | 13.8% | 0.0%* | 6.07% |
| 40 | 3.3% | 21.3% | 8.3% | 9.73% |
| 80 | 3.3% | 64.8% | 87.0% | 28.73% |

*At d=20, many non-path nodes also have low degree, so the shortcut is weak.

Key observation: **the shortcut gets stronger as distractors increase**. This is because more distractors push non-path node degrees higher, making path nodes (especially the target) stand out more. This exactly mirrors the observed trend of "graph advantage grows with noise" — but the cause may be structural shortcuts, not multi-hop reasoning.

### The BFS safety check is vacuous

The code checks for alternative k-hop paths from start to target:

```python
for p in nx.all_simple_paths(G, start, target, cutoff=k):
    if len(p) == k + 1 and p != path:
        has_alt = True
```

Over 1,000 samples, this triggered **0 times** (0.0%). Since distractors only connect non-path nodes, they cannot create any path involving the start node (v0's only outgoing edge is the true path edge). The check is doing nothing.

### The target is always the only k-hop-reachable node

Since no distractor touches the start node, the only outgoing edge from v0 is v0→v1. The entire reachability tree from v0 is exactly the true path. No other node is reachable in k hops (or any number of hops). This means the task isn't "which of several reachable nodes is the right one" — it's "can you trace the one and only path."

---

## Moderate Issues

### Directed edges treated as undirected in masks and GCN

Both `graph_token_mask` and `khop_graph_token_mask` treat the graph as undirected (connecting tokens by shared node ID regardless of src/dst role). `tokens_to_graph` for GCN explicitly symmetrizes: `adj[src,dst] = adj[dst,src] = 1.0`.

This is defensible for the Transformer (directionality is encoded in token position — src at offset +1, dst at offset +3, and the model can learn this). But for the GCN, undirected edges let information flow backward along the path, which could help or obscure things.

**Impact:** Moderate. The model can still learn directionality from the token encoding, but this should be documented as a design choice.

### No self-loops or isolated nodes in encoding

The encoding only represents edges that exist. Nodes with no edges at all never appear in the token sequence. This is fine currently (all non-path nodes appear in at least one distractor edge at d≥20 for 30 nodes), but could matter at low distractor counts.

### Shorter start→target routes are not forbidden

The generator’s safety loop only rejects samples where there exists **another simple path of exactly *k* edges** from start to the **labeled target** (same length as the planted path, different node sequence):

```python
for p in nx.all_simple_paths(G, start, target, cutoff=k):
    if len(p) == k + 1 and p != path:
        return None  # reject
```

It does **not** check for a path to the target with **fewer than *k* edges**. So the graph can contain a “shortcut” route start → … → target in fewer than *k* hops while the label remains the endpoint of the planted *k*-edge path. The learning objective is still well-defined (predict the given target node), but the task is not “shortest path” or “unique *k*-hop endpoint.” **Worth one sentence in the final report** under limitations / task definition.

---

## Fix Applied

### Fix 1: Allow distractors to touch path nodes — APPLIED

Changed the candidate pool in `build_dataset()`:

```python
# Old (flawed):
non_path = [v for v in nodes if v not in path_set]
cand = [(u, v) for u in non_path for v in non_path if u != v]

# New (fixed):
cand = [(u, v) for u in nodes for v in nodes
        if u != v and (u, v) not in true_set]
```

Also refactored the duplicated main-loop/retry-loop into a single `_try_one_sample()` helper.

**Verified results after fix:**

| Distractors | Safety rejections | Shortcut 1 (was) | Shortcut 2 (was) |
|-------------|-------------------|-------------------|-------------------|
| 20 | 3.8% | 7.4% (13.8%) | 0.0% (0.0%) |
| 40 | 16.2% | 5.3% (21.3%) | 0.3% (8.3%) |
| 80 | 56.2% | 2.9% (64.8%) | 1.3% (87.0%) |

Shortcuts are reduced to near-random (3.3%) levels. Generation speed remains fast (<7s for n=8000, d=80).

### Fix 2 and Fix 3: NOT applied (naturally addressed)

Fix 1 implicitly addresses Fix 2: since distractors can now touch any node, the target and start nodes naturally accumulate distractor edges, equalizing degree distributions. Fix 3 (competing k-hop endpoints) happens organically — distractors touching path nodes can create k-hop paths from start to non-target nodes, which the safety check does NOT reject (it only rejects alternative paths to the target).

---

## Impact after the fix

1. **Accuracies dropped** vs the old shortcut-heavy regime (expected).
2. **Graph vs dense/window** should be interpreted with shortcut baselines (`measure_shortcut_baselines` / bar chart) — gains above heuristic ceilings support genuine reasoning.
3. **Trends vs distractor count** may differ from pre-fix runs; re-run sweeps for the final report.

---

## What's NOT Wrong

- Label distribution: uniform ✓
- Encoding roundtrip: correct ✓
- Edge shuffle: uniform ✓
- Mask-encoding alignment: correct ✓
- Sequence length: consistent ✓
- Vocab construction: no collisions ✓
