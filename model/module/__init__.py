"""
Core building blocks for transformer architectures.

This package provides efficient implementations of:
- Flash Attention: IO-aware exact attention for faster training/inference
- SwiGLU: Gated activation function for improved FFN performance
- RoPE: Rotary Position Embeddings for relative position encoding
- Block: Single decoder block with standard softmax attention
- HyperGraph Sparse Attention: Sparse attention with K parallel timelines

Note: Full model architectures should be composed separately to enable
hybrid designs mixing different attention types (standard, sparse, etc.)
"""

from .flash_attention import (
    FlashAttention,
    FlashCrossAttention,
    FlashMultiHeadAttention,
)

from .swiglu import (
    SwiGLU,
    SwiGLUFFN,
    GeGLU,
    ReGLU,
)

from .rope import (
    RotaryEmbedding,
    RoPEAttention,
    YaRNRoPE,
    ALiBiPositionalBias,
    apply_rotary_pos_emb,
    rotate_half,
)

from .block import (
    BlockConfig,
    DecoderBlock,
    RMSNorm,
)

from .hypergraph_attention import (
    HyperGraphRoPE,
    HyperGraphSparseAttention,
)

from .sparse_block import (
    SparseBlockConfig,
    SparseDecoderBlock,
)


__all__ = [
    # Flash Attention
    "FlashAttention",
    "FlashCrossAttention",
    "FlashMultiHeadAttention",
    # SwiGLU
    "SwiGLU",
    "SwiGLUFFN",
    "GeGLU",
    "ReGLU",
    # RoPE
    "RotaryEmbedding",
    "RoPEAttention",
    "YaRNRoPE",
    "ALiBiPositionalBias",
    "apply_rotary_pos_emb",
    "rotate_half",
    # Standard Decoder Block
    "BlockConfig",
    "DecoderBlock",
    "RMSNorm",
    # HyperGraph Sparse Attention
    "HyperGraphRoPE",
    "HyperGraphSparseAttention",
    "SparseBlockConfig",
    "SparseDecoderBlock",
]

