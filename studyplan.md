# HyperGraph Sparse Attention - Study Plan

## Research Focus
Long context modeling by partitioning timeline into K shorter timelines with learned routing.

---

## Benchmark Comparison Plan

### Method Properties Comparison

| Property | HyperGraph (Ours) | MoSA | Fixed Block-Sparse | Longformer |
|----------|-------------------|------|-------------------|------------|
| **Mechanism** | Partition into K timelines | Select k tokens | Fixed chunking | Sliding window + global |
| **Routing** | Learned MLP | Learned linear | None (positional) | None (fixed pattern) |
| **Coverage** | Full (all tokens) | Partial (k tokens) | Full | Full |
| **Position encoding** | Local RoPE (resets) | Local RoPE | Global | Global |
| **Complexity** | O(N²/K) | O(k²) = O(N²/K²) | O(N²/K) | O(N) |

### Priority Benchmarks

#### 1. Dense Baseline ⭐⭐⭐ (Required)
- **Purpose**: Quality ceiling, speedup baseline
- **Implementation**: Standard multi-head attention
- **Metrics**: Perplexity, throughput, memory

#### 2. Fixed Block-Sparse ⭐⭐⭐ (Critical Ablation)
- **Purpose**: Demonstrate value of learned routing
- **Implementation**: Same as HyperGraph but tokens assigned by position (tokens 0 to N/K → timeline 0, etc.)
- **Hypothesis**: Learned routing should show better perplexity due to semantic clustering
- **Metrics**: Perplexity gap vs HyperGraph, routing entropy comparison

#### 3. MoSA ⭐⭐⭐ (Primary Comparison)
- **Paper**: [Mixture of Sparse Attention](https://arxiv.org/abs/2505.00315)
- **Code**: https://github.com/piotrpiekos/MoSA
- **Purpose**: Compare partition (full coverage) vs selection (partial coverage)
- **Key differences**:
  - MoSA: Each head selects k tokens → O(k²) per head, partial coverage
  - Ours: Each head partitions all N into K groups → O(N²/K) per head, full coverage
- **Hypothesis**: HyperGraph should be more stable (guaranteed coverage) but MoSA more efficient
- **Metrics**: Perplexity, FLOPs, memory, needle-in-haystack retrieval

#### 4. Longformer ⭐⭐⭐ (Standard Baseline)
- **Paper**: [Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150)
- **Purpose**: Compare learned global partitioning vs fixed local patterns
- **Key differences**:
  - Longformer: Sliding window (local) + global tokens → O(N)
  - Ours: Learned timeline partitioning → O(N²/K)
- **Hypothesis**: HyperGraph may capture longer-range dependencies better
- **Metrics**: Perplexity, long-range retrieval tasks

### Secondary Benchmarks

#### 5. Routing Transformer ⭐⭐
- **Paper**: [Routing Transformer](https://arxiv.org/abs/2003.05997)
- **Purpose**: Compare routing mechanisms (MLP vs online k-means)
- **Key differences**:
  - Routing Transformer: Online k-means clustering
  - Ours: Learned MLP router with Gumbel-softmax
- **Metrics**: Routing stability, convergence speed, perplexity

#### 6. BigBird ⭐⭐
- **Paper**: [Big Bird: Transformers for Longer Sequences](https://arxiv.org/abs/2007.14062)
- **Purpose**: Compare with random + local + global attention patterns
- **Metrics**: Perplexity on long documents

#### 7. Sparse Transformer ⭐
- **Paper**: [Generating Long Sequences with Sparse Transformers](https://arxiv.org/abs/1904.10509)
- **Purpose**: Compare strided attention patterns
- **Complexity**: O(N√N)

---

## Complexity Landscape

| Method | Attention Complexity | Coverage | Routing |
|--------|---------------------|----------|---------|
| Dense | O(N²) | Full | None |
| **HyperGraph (Ours)** | **O(N²/K)** | **Full** | **Learned** |
| Fixed Block-Sparse | O(N²/K) | Full | Fixed |
| MoSA | O(N²/K²) | Partial | Learned |
| Longformer | O(N) | Full | Fixed |
| BigBird | O(N) | Full | Fixed + Random |
| Sparse Transformer | O(N√N) | Full | Fixed |
| Linear Attention | O(N) | Full | None |

---

## Evaluation Tasks

### Language Modeling
- [ ] WikiText-103 perplexity
- [ ] PG-19 (long documents)
- [ ] Perplexity vs sequence length curves

### Long-Range Retrieval
- [ ] Needle-in-haystack (passkey retrieval)
- [ ] Long-range arena benchmark
- [ ] Document QA (NarrativeQA, Qasper)

### Efficiency Metrics
- [ ] Training throughput (tokens/sec)
- [ ] Inference latency vs sequence length
- [ ] Peak memory usage
- [ ] FLOPs measurement

### Routing Analysis
- [ ] Routing entropy over training
- [ ] Timeline load balance
- [ ] Semantic clustering visualization (t-SNE of routed tokens)
- [ ] Routing pattern stability across layers

---

## Key Research Questions

1. **Does learned routing outperform fixed partitioning?**
   - Compare: HyperGraph vs Fixed Block-Sparse
   - Metric: Perplexity gap

2. **Is full coverage worth the extra compute vs token selection?**
   - Compare: HyperGraph vs MoSA
   - Metric: Quality per FLOP

3. **Can timeline partitioning capture long-range dependencies better than local attention?**
   - Compare: HyperGraph vs Longformer
   - Metric: Needle-in-haystack, long-range retrieval

4. **What is the optimal K (number of timelines)?**
   - Ablation: K = {2, 4, 8, 16, 32}
   - Trade-off: Quality vs efficiency

5. **Does local RoPE (position reset) help or hurt?**
   - Ablation: Local RoPE vs global RoPE within timelines
   - Hypothesis: Local RoPE enables better generalization to longer sequences

---

## Implementation Checklist

### Baselines to Implement
- [ ] Dense attention (reference)
- [ ] Fixed block-sparse (ablation)
- [ ] Integrate MoSA from https://github.com/piotrpiekos/MoSA
- [ ] Longformer attention pattern

### Evaluation Infrastructure
- [ ] Perplexity evaluation script
- [ ] Throughput benchmarking
- [ ] Memory profiling
- [ ] Needle-in-haystack test

### Visualization
- [ ] Routing pattern heatmaps
- [ ] Timeline load balance plots
- [ ] Attention pattern comparison figures

---

## Timeline

| Week | Focus |
|------|-------|
| 1 | Implement Fixed Block-Sparse baseline, run ablation |
| 2 | Integrate MoSA, compare on WikiText-103 |
| 3 | Add Longformer baseline, long-range retrieval tasks |
| 4 | Scaling experiments (vary K, sequence length) |
| 5 | Analysis: routing patterns, visualizations |
| 6 | Paper writing, final benchmarks |

