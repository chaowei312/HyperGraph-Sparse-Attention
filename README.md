# HyperGraph Sparse Attention

A decoder-only Transformer with **learned timeline partitioning** for efficient long-context modeling. Instead of attending to all N tokens, each attention head routes tokens into K independent timelines via a learned router, then computes causal attention within each timeline — reducing attention complexity from O(N²) to **O(N²/K)** while preserving full token coverage.

## Key Idea

```
Input (N tokens) → Learned Router → K timelines (each ~N/K tokens)
                                   → Causal attention per timeline (local RoPE)
                                   → Scatter back → Output
```

Each head independently partitions all N tokens into K groups ("timelines") using a Gumbel-Softmax router. Attention is computed only within each timeline, with positions resetting to 0 (timeline-local RoPE). All timelines are processed in a **single xformers kernel call** via `BlockDiagonalCausalMask`.

### Comparison with Related Methods

| Property | HyperGraph (Ours) | MoSA | Fixed Block-Sparse | Longformer |
|----------|-------------------|------|---------------------|------------|
| **Mechanism** | Partition into K timelines | Select k tokens | Fixed chunking | Sliding window + global |
| **Routing** | Learned MLP/linear | Learned linear | None (positional) | None (fixed pattern) |
| **Coverage** | Full (all tokens) | Partial (k tokens) | Full | Full |
| **Position encoding** | Local RoPE (resets) | Local RoPE | Global | Global |
| **Complexity** | O(N²/K) | O(N²/K²) | O(N²/K) | O(N) |

## Architecture

The model uses a **hybrid block pattern** of Full (F) and Sparse (S) decoder layers:

- **Full blocks (`F`)**: Standard multi-head causal attention
- **Sparse blocks (`S`)**: HyperGraph sparse attention with learned routing

Blocks are composed via a flexible pattern string (e.g., `"FFFSSSSSSSSFFF"` for the bookend layout). The default experimental configuration uses 14 layers with 6 Full + 8 Sparse (57% sparse).

### Pre-defined Architectures

| Name | Pattern | Description |
|------|---------|-------------|
| `baseline` | `FFFFFFFFFFFFFF` | All full attention (quality reference) |
| `early_full` | `FFFFFFSSSSSSSS` | Global context first, sparse late |
| `late_full` | `SSSSSSSSFFFFFF` | Sparse encoding, full decoding |
| `bookend` | `FFFSSSSSSSSFFF` | Full I/O layers, sparse middle |
| `interlaced_sf` | `SFSFSFSFSFSFSS` | Alternating sparse/full |
| `reverse_bookend` | `SSSFFFFFFSSSSS` | Sparse I/O, full middle |

### HyperGraph Sparse Attention Details

- **Router**: Linear or 2-layer MLP projecting each token to `num_heads × K` logits
- **Gumbel-Softmax**: Stochastic routing during training prevents collapse; deterministic argmax at inference
- **Top-K routing**: Each token can attend in multiple timelines (bridge tokens for cross-timeline information flow)
- **Load balance loss**: Switch Transformer-style auxiliary loss + entropy regularization + z-loss
- **Confidence gate** (optional): Learned per-head gate to suppress uninformative attention outputs
- **RoPE modes**: Timeline-local (default), global, or mixed (full layers global, sparse layers local)

## Project Structure

```
HyperGraph-Sparse-Attention/
├── model/
│   ├── model.py                          # CausalLM: main model with ModelConfig
│   └── module/
│       ├── hypergraph_attention.py        # Core HyperGraph sparse attention
│       ├── block.py                       # Standard decoder block
│       ├── sparse_block.py               # Sparse decoder block
│       ├── rope.py                        # Rotary position embeddings
│       ├── flash_attention.py             # Flash attention wrapper
│       ├── swiglu.py                      # SwiGLU FFN
│       └── mixture_of_heads_attention.py  # MoH baseline for comparison
├── train/
│   ├── parallel_train.py                  # Multi-GPU parallel training
│   ├── architectures.py                   # Architecture definitions (single source of truth)
│   ├── benchmark.py                       # Throughput & memory benchmarking
│   ├── benchmark_comparison.py            # Cross-method comparison
│   └── training_utils.py                  # LR schedules, logging, etc.
├── data/
│   ├── dataset.py                         # Dataset loaders (Gutenberg, WikiText, OpenWebText, SlimPajama)
│   └── fast_tokenizer.py                  # GPT-2 tokenizer wrapper
├── scripts/
│   ├── run_ablation.py                    # Ablation study runner
│   ├── run_fixed_capacity_benchmark.py    # Fixed-capacity baseline benchmark
│   ├── run_mosa_benchmark.py              # MoSA comparison benchmark
│   ├── visualize_routing.py               # Routing pattern visualizations
│   ├── analyze_load_balance.py            # Load balance analysis
│   ├── plot_loss_curves.py                # Training curve plots
│   └── benchmark_inference.py             # Inference latency benchmarks
├── configs/
│   ├── base.yaml                          # Base configuration
│   ├── baseline.yaml                      # All-full-attention baseline
│   ├── hybrid_2_6.yaml / hybrid_4_4.yaml  # Hybrid configs
│   └── sparse_only.yaml                   # All-sparse config
├── results/                               # Experiment results (JSON + figures)
├── report.ipynb                           # Analysis notebook
├── studyplan.md                           # Research plan & benchmark design
└── utils/
    └── config.py                          # YAML config loader with inheritance
```

## Quick Start

### Requirements

- Python 3.10+
- PyTorch 2.0+
- [xformers](https://github.com/facebookresearch/xformers) (required for block-sparse attention kernel)
- CUDA GPU

```bash
pip install torch xformers
pip install pyyaml nltk tiktoken datasets  # for data loading
```

### Training

Train multiple architectures in parallel across GPUs:

```bash
# Train all architectures on Gutenberg corpus (default)
python train/parallel_train.py --num_steps 20000 --output_dir results/my_experiment

# Train specific architectures on specific GPUs
python train/parallel_train.py \
    --configs baseline bookend early_full \
    --gpus 0 1 2 \
    --num_steps 20000 \
    --seq_len 1024 \
    --dim 512 \
    --num_heads 8 \
    --num_hyper_nodes 4 \
    --lr 3e-4 \
    --lr_schedule cosine \
    --dataset wikitext-103

# Sequential training (single GPU)
python train/parallel_train.py --configs bookend --sequential --gpus 0
```

### Model Usage

```python
from model import ModelConfig, CausalLM

# Hybrid model: bookend pattern (full I/O, sparse middle)
config = ModelConfig(
    dim=512,
    num_heads=8,
    num_hyper_nodes=4,           # K=4 timelines per head
    block_pattern="FFFSSSSSSSSFFF",
    max_seq_len=2048,
)

model = CausalLM(config).cuda()
print(model)  # Shows architecture summary

# Forward pass
input_ids = torch.randint(0, 50257, (1, 1024)).cuda()
logits, aux_loss = model(input_ids)
# logits: (1, 1024, 50257)
# aux_loss: load balance loss (add to CE loss with weight ~0.01)
```

### Configuration

Configs can be defined via YAML or directly in Python:

```python
# Simple mode: standard blocks first, then sparse
config = ModelConfig(n_standard_blocks=4, n_sparse_blocks=4)

# Flexible mode: explicit pattern string
config = ModelConfig(block_pattern="SSSSSSSFFF")

# Flexible mode: list composition
config = ModelConfig(block_pattern=['S']*6 + ['F']*3)
```

Key hyperparameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_hyper_nodes` | 4 | K timelines per head (higher = more speedup, less quality) |
| `top_k` | 1 | Timelines each token routes to (2 = bridge tokens) |
| `router_temperature` | 1.0 | Gumbel-Softmax temperature (higher = more exploration) |
| `entropy_weight` | 0.01 | Entropy regularization strength |
| `router_type` | `"linear"` | Router type: `"linear"` or `"mlp"` |
| `use_local_rope` | `True` | Timeline-local RoPE (positions reset per timeline) |

## Datasets

| Dataset | Size | Source |
|---------|------|--------|
| Gutenberg | ~2.4M tokens | NLTK (built-in, no download) |
| WikiText-2 | ~2M tokens | HuggingFace |
| WikiText-103 | ~100M tokens | HuggingFace |
| OpenWebText | ~8B tokens | HuggingFace |
| SlimPajama | ~627B tokens | HuggingFace |

## Experimental Results

All experiments use 14-layer models with d=512, 8 heads, cosine LR schedule, and PG19 dataset unless noted. Full training logs and figures are in `results/` and `report.ipynb`.

### Ablation Study — Design Choices (Phase 1)

Pure sparse models (all 14 layers sparse) to isolate individual design decisions:

| Ablation | Configuration | Best Val Loss | Steps |
|----------|---------------|:------------:|:-----:|
| **Timelines (K)** | K=4, local RoPE | **3.856** | 56K |
| | K=6, local RoPE | 4.072 | 42K |
| **RoPE mode** | Local (positions reset) | 3.856 | 56K |
| | Global (original positions) | **3.657** | 94K |
| **Router type** | Linear | **3.978** | 48K |
| | MLP (2-layer) | 4.242 | 26K |
| **Temperature** | τ=1.0 | **3.956** | 64K |
| | τ=0.5 (sharper) | 4.005 | 56K |

**Findings:**
- **K=4 consistently outperforms K=6** — fewer partitions preserve more intra-sequence dependencies
- **Global RoPE unexpectedly beats local RoPE** for pure-sparse models, though local RoPE converges faster early
- **Linear router > MLP router** — the simpler router trains more stably and reaches lower loss
- **Temperature τ=1.0 slightly preferred** over τ=0.5, suggesting exploration helps

### Hybrid Architectures (Phase 2)

Interlaced hybrid models (6 Full + 8 Sparse, `FSSFSSFSSFSSFF`) vs dense baseline, with context length extrapolation:

| Model | RoPE Mode | Val Loss (1024) | Extrap 2× | Extrap 4× | Extrap 8× |
|-------|-----------|:---------------:|:---------:|:---------:|:---------:|
| **Dense baseline** | Global | **3.393** | 4.14 | 4.88 | 5.30 |
| Hybrid (global) | Global | 3.449 | 4.21 | 4.89 | 5.30 |
| Hybrid (mixed) | Full=global, Sparse=local | 3.448 | 4.22 | 4.96 | — |
| Hybrid + freq explore | Mixed + random pos scaling | 3.453 | 4.33 | 5.17 | 5.38 |

**Findings:**
- Dense baseline still achieves the best perplexity at this scale — expected since sparse routing has an inherent information bottleneck
- Hybrid models close to within **~1.6% of the dense baseline** while replacing 57% of attention layers with O(N²/K) sparse variants
- **Context length extrapolation remains challenging** — local RoPE does not provide the hoped-for length generalization advantage; all models degrade similarly beyond training length
- RoPE frequency exploration hurts rather than helps extrapolation

### IsoFLOP Comparison — Reinvesting Compute Savings

When K=6 attention is cheaper (O(N²/6) vs O(N²/4)), we can reinvest the saved FLOPs into more capacity:

| Configuration | K | Layers | Dim | Seq Len | Best Val Loss |
|---------------|:-:|:------:|:---:|:-------:|:------------:|
| **K=4 baseline** | 4 | 14 | 512 | 1024 | **3.685** |
| K=6 + wider | 6 | 14 | 640 | 1024 | 3.736 |
| K=6 + deeper | 6 | 21 | 512 | 1024 | 3.752 |
| K=6 + longer context | 6 | 14 | 512 | 1536 | 3.817 |

**Finding:** Under matched compute budgets, **K=4 still wins** — the quality loss from more aggressive partitioning is not recovered by adding width, depth, or context length. This suggests K=4 is near the optimal partition count for this model scale.

### Block Placement Comparison

Architecture placement sweep (dim=768, 12 heads, WikiText-103):

| Architecture | Pattern | Best Val Loss |
|--------------|---------|:------------:|
| Bookend | `FFFSSSSSSSSFFF` | Best among hybrids |
| Interlaced FSS | `FSSFSSFSSFSSFF` | Close second |
| Reverse bookend | `SSSFFFFFFSSSSS` | Competitive |
| Early full | `FFFFFFSSSSSSSS` | Moderate |
| Late full | `SSSSSSSSFFFFFF` | Worst hybrid |
| Dense baseline | `FFFFFFFFFFFFFF` | Quality ceiling |

**Finding:** **Bookend and interlaced patterns** perform best — having full attention at the first and last layers appears important for the model's ability to form and consolidate representations.

### Summary & Takeaways

1. **The mechanism works**: Learned routing successfully partitions tokens into timelines with meaningful load balance, achieving near-dense quality at reduced attention cost
2. **K=4 is the sweet spot** for this model scale — more partitions hurt quality without sufficient compute recovery
3. **Block placement matters**: Bookend/interlaced patterns significantly outperform naive stacking
4. **Honest limitation**: At small scale (512d/14L), the dense baseline still wins on raw perplexity. The value proposition is for **longer sequences** where O(N²/K) savings become dominant
5. **Length extrapolation remains open**: Local RoPE did not provide the expected generalization benefit in our experiments

See `report.ipynb` for full training curves, routing visualizations, and detailed analysis.

## License

Research project — see repository for details.

