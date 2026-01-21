"""
Sparse Attention Model Package.

Provides:
- CausalLM: Decoder-only language model with hybrid attention support
- ModelConfig: Configuration for full model architecture
- Building blocks in model.module for custom architectures
"""

from .model import (
    ModelConfig,
    CausalLM,
)

from .module import (
    # Configs
    BlockConfig,
    SparseBlockConfig,
    # Standard blocks
    DecoderBlock,
    RMSNorm,
    # Sparse blocks
    SparseDecoderBlock,
    HyperGraphSparseAttention,
    # Primitives
    FlashAttention,
    SwiGLU,
    RotaryEmbedding,
    RoPEAttention,
)


__all__ = [
    # Model
    "ModelConfig",
    "CausalLM",
    # Configs
    "BlockConfig",
    "SparseBlockConfig",
    # Blocks
    "DecoderBlock",
    "SparseDecoderBlock",
    # Components
    "RMSNorm",
    "HyperGraphSparseAttention",
    "FlashAttention",
    "SwiGLU",
    "RotaryEmbedding",
    "RoPEAttention",
]

