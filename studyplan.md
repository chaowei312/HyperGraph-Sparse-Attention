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

## Datasets

### Language Modeling

| Dataset | Context Length | Purpose |
|---------|---------------|---------|
| **WikiText-103** | ~3K tokens | Standard baseline, easy comparison |
| **PG-19** | 10K-100K tokens | Long documents, tests timeline benefits |
| **OpenWebText** | Variable | Web text diversity |
| **ArXiv** | 8K-32K tokens | Technical documents, structured content |

### Long Context Retrieval

| Dataset/Task | Description | Why Important |
|--------------|-------------|---------------|
| **Needle-in-Haystack** | Find passkey in long context | Tests if timelines preserve information |
| **RULER** | Multi-hop retrieval benchmark | Tests complex dependencies across timelines |
| **LongBench** | Diverse long-context tasks | Comprehensive evaluation |
| **ScrollS** | Summarization, QA over long docs | Practical long-context applications |

---

## Experiments

### Experiment 1: Iso-FLOP Quality Comparison ⭐⭐⭐

**Goal**: Same FLOPs budget, compare perplexity

```
Fix total FLOPs budget = F

Dense baseline:     H heads × N² × d = F
HyperGraph:         H heads × K × (N/K)² × d = F/K  → K× compute savings
Fixed Block-Sparse: H heads × K × (N/K)² × d = F/K  → ablation (no learned routing)
Longformer:         H heads × N × w × d ≈ F/N      → linear complexity
```

**Configurations**:
| Config | Heads | Seq Len | K | Layers | Notes |
|--------|-------|---------|---|--------|-------|
| Dense-small | 8 | 2048 | - | 12 | Baseline |
| HyperGraph-K4 | 8 | 2048 | 4 | 12 | 4× fewer FLOPs |
| HyperGraph-K8 | 8 | 2048 | 8 | 12 | 8× fewer FLOPs |
| Fixed-Block-K4 | 8 | 2048 | 4 | 12 | Ablation |
| Longformer-w512 | 8 | 2048 | - | 12 | Window=512 |

**Metrics**: WikiText-103 PPL, PG-19 PPL, Training loss curves

### Experiment 2: Iso-FLOP with Longer Context ⭐⭐⭐

**Goal**: Use compute savings for longer context instead of speed

```
Same FLOPs as Dense-2K:
- Dense-2K:       seq_len=2048, FLOPs=F
- HyperGraph-4K:  seq_len=4096, K=4  → FLOPs≈F (K=4 offsets 4× longer seq)
- HyperGraph-8K:  seq_len=8192, K=8  → FLOPs≈F
- HyperGraph-16K: seq_len=16384, K=16 → FLOPs≈F
```

**Configurations**:
| Config | Seq Len | K | Effective FLOPs |
|--------|---------|---|-----------------|
| Dense-2K | 2048 | - | F (baseline) |
| HyperGraph-4K | 4096 | 4 | ≈F |
| HyperGraph-8K | 8192 | 8 | ≈F |
| HyperGraph-16K | 16384 | 16 | ≈F |

**Metrics**: PG-19 PPL (benefits from long context), Needle-in-Haystack accuracy

**Hypothesis**: Longer context + timeline partitioning should improve performance on long documents

### Experiment 3: Needle-in-Haystack (Long Context Retrieval) ⭐⭐⭐

**Goal**: Test if timelines preserve information across long context

```python
def needle_in_haystack_experiment():
    context_lengths = [4096, 8192, 16384, 32768, 65536]
    needle_positions = [0.1, 0.3, 0.5, 0.7, 0.9]  # relative position
    
    for ctx_len in context_lengths:
        for pos in needle_positions:
            # Insert random passkey at position
            # Ask model to retrieve it
            # Measure accuracy
```

**Compare**:
- Dense (may OOM at long lengths)
- HyperGraph-K8 (should handle longer context)
- Longformer (local window may miss distant needles)
- Fixed-Block (ablation - does routing help retrieval?)

**Expected Result**: HyperGraph should maintain retrieval accuracy because timelines can learn to route related information together

### Experiment 4: AR Generation Benchmark ⭐⭐⭐

**Goal**: Show HyperGraph works for real inference (unlike MoSA)

```python
def generation_benchmark():
    prompt_lengths = [512, 1024, 2048, 4096]
    generate_tokens = 100
    
    for prompt_len in prompt_lengths:
        measure:
            - time_to_first_token  # prefill time
            - tokens_per_second    # generation speed
            - peak_memory          # KV cache size
            - total_time           # end-to-end
```

**Compare**:
| Method | Prefill | Generation | KV Cache | Notes |
|--------|---------|------------|----------|-------|
| Dense | ✅ | ✅ | 100% | Baseline |
| HyperGraph | ✅ | ✅ | 100% | Works incrementally |
| MoSA | ✅ | ❌ | Invalid | Must recompute every step |
| Longformer | ✅ | ✅ | ~w/N | Efficient but local |

**Key Point**: This experiment exposes MoSA's practical limitation

### Experiment 5: Ablation Studies ⭐⭐

| Ablation | What it Tests |
|----------|---------------|
| **K variation** | K ∈ {2, 4, 8, 16, 32} - optimal timeline count |
| **Local vs Global RoPE** | Position reset within timeline vs global positions |
| **Router architecture** | Linear vs MLP router |
| **Load balance loss weight** | α ∈ {0.001, 0.01, 0.1} |
| **Top-k routing** | k=1 vs k=2 (current uses k=1 for attention) |

---

## Expected Results Tables

### Table 1: Iso-FLOP Perplexity (Same Compute)

| Method | FLOPs | WikiText-103 ↓ | PG-19 ↓ | Params |
|--------|-------|----------------|---------|--------|
| Dense | F | X.XX | X.XX | 125M |
| Fixed-Block K=4 | F/4 | X.XX | X.XX | 125M |
| **HyperGraph K=4** | F/4 | **X.XX** | **X.XX** | 125M |
| Longformer | ~F/4 | X.XX | X.XX | 125M |

### Table 2: Long Context (Same FLOPs as Dense-2K)

| Method | Seq Len | PG-19 PPL ↓ | Needle Acc ↑ |
|--------|---------|-------------|--------------|
| Dense | 2K | X.XX | OOM |
| HyperGraph K=4 | 4K | X.XX | XX% |
| HyperGraph K=8 | 8K | X.XX | XX% |
| **HyperGraph K=16** | 16K | **X.XX** | **XX%** |

### Table 3: AR Generation Efficiency

| Method | Prefill 4K (ms) | Tokens/sec ↑ | KV Memory |
|--------|-----------------|--------------|-----------|
| Dense | XXX | XXX | 100% |
| **HyperGraph K=8** | XXX | XXX | 100% |
| MoSA | XXX | N/A* | N/A* |
| Longformer | XXX | XXX | ~12.5% |

*MoSA cannot do incremental generation - must recompute selection every step

### Table 4: Ablation - Number of Timelines K

| K | Attention FLOPs | WikiText-103 PPL | PG-19 PPL |
|---|-----------------|------------------|-----------|
| 1 (Dense) | N² | X.XX | X.XX |
| 2 | N²/2 | X.XX | X.XX |
| 4 | N²/4 | X.XX | X.XX |
| 8 | N²/8 | X.XX | X.XX |
| 16 | N²/16 | X.XX | X.XX |

---

## Evaluation Metrics Summary

| Metric | What it Shows | Priority |
|--------|---------------|----------|
| **Perplexity** | Model quality | ⭐⭐⭐ |
| **FLOPs/token** | Theoretical efficiency | ⭐⭐⭐ |
| **Needle-in-Haystack** | Long-range retrieval | ⭐⭐⭐ |
| **Tokens/second (gen)** | AR inference speed | ⭐⭐⭐ |
| **Peak memory** | Hardware requirements | ⭐⭐ |
| **Wall-clock time** | Practical speed | ⭐⭐ |
| **Routing entropy** | Routing diversity | ⭐ |

---

## MoSA Comparison Notes

**Critical finding from code analysis**: MoSA has fundamental AR generation issues:

1. **No KV cache**: Their code has no caching mechanism
2. **Selection changes**: Adding token t+1 changes which tokens are selected
3. **Must recompute**: Every generation step requires full recomputation

**Your advantage**: HyperGraph timeline routing is AR-friendly:
- Each token's timeline assignment is independent
- KV cache per timeline remains valid
- O(1) routing decision per new token

**Paper contribution point**: 
> "Unlike prior token-selection methods (e.g., MoSA), our timeline partitioning enables efficient KV caching for autoregressive generation while maintaining learned content-based sparsity."

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

| Week | Focus | Experiments |
|------|-------|-------------|
| 1 | Setup & Baselines | Dense baseline, Fixed Block-Sparse ablation |
| 2 | Core Experiments | Exp 1: Iso-FLOP on WikiText-103, PG-19 |
| 3 | Long Context | Exp 2: Iso-FLOP longer context, Exp 3: Needle-in-Haystack |
| 4 | Inference | Exp 4: AR generation benchmark (expose MoSA flaw) |
| 5 | Ablations | Exp 5: K variation, RoPE ablation, router architecture |
| 6 | Analysis | Routing visualizations, attention patterns |
| 7 | Longformer comparison | Add Longformer baseline, compare all methods |
| 8 | Paper writing | Final benchmarks, figures, write-up |

### Weekly Deliverables

**Week 1-2: Foundation**
- [ ] WikiText-103 dataloader
- [ ] PG-19 dataloader  
- [ ] Dense attention baseline
- [ ] Fixed Block-Sparse baseline
- [ ] Training pipeline verified

**Week 3-4: Core Results**
- [ ] Iso-FLOP perplexity table (Table 1)
- [ ] Long context perplexity table (Table 2)
- [ ] Needle-in-Haystack accuracy plot
- [ ] AR generation benchmark (Table 3)

**Week 5-6: Analysis**
- [ ] K ablation table (Table 4)
- [ ] RoPE ablation results
- [ ] Routing visualization figures
- [ ] Timeline load balance analysis

**Week 7-8: Polish**
- [ ] Longformer comparison
- [ ] All figures finalized
- [ ] Paper draft complete

