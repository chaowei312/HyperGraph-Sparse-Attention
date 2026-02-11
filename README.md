# HyperGraph Sparse Attention

A research project exploring **sparse attention mechanisms via hypergraph partitioning** for efficient long-context modeling in transformers.

## Overview

Traditional self-attention has O(N²) complexity, making it prohibitive for long sequences. This project investigates partitioning token sequences into K shorter timelines using learned routing, achieving O(N²/K) complexity while maintaining full token coverage.

## Key Ideas

- **Hypergraph Partitioning**: Tokens are routed to K parallel timelines via a learned MLP router
- **Local Position Encoding**: RoPE resets within each timeline for better local modeling
- **Learned vs Fixed Routing**: Demonstrates that semantic-aware routing outperforms positional block-sparse baselines

## Benchmark Comparisons

| Method | Mechanism | Routing | Complexity |
|--------|-----------|---------|------------|
| HyperGraph (Ours) | K timeline partition | Learned MLP | O(N²/K) |
| MoSA | Select k tokens | Learned linear | O(N²/K²) |
| Fixed Block-Sparse | Fixed chunking | None | O(N²/K) |
| Longformer | Sliding window + global | None | O(N) |

## Project Structure



## Tech Stack

- Python, PyTorch
- Custom sparse attention kernels
- RoPE positional encoding
