"""
Decoder Block with standard softmax attention.

Provides a single causal transformer block combining:
- Flash Attention with RoPE and causal masking
- SwiGLU Feed-Forward Network
- Pre-normalization (RMSNorm/LayerNorm)

This module only defines the building block, not full model architectures.
Full models should be composed in a separate module to enable hybrid designs.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Literal

from .swiglu import SwiGLU
from .rope import RoPEAttention


@dataclass
class BlockConfig:
    """
    Configuration for a single transformer block.
    
    This only contains block-level hyperparameters, not model-level
    settings like depth or vocab_size.
    
    Attributes:
        dim: Model dimension / embedding size
        num_heads: Number of attention heads
        num_kv_heads: Number of key-value heads for GQA (None = MHA)
        head_dim: Dimension per attention head (default: dim // num_heads)
        ffn_dim: Feed-forward hidden dimension (default: 4 * dim)
        ffn_multiplier: Multiplier for FFN hidden dim (alternative to ffn_dim)
        max_seq_len: Maximum sequence length
        dropout: Dropout probability
        attn_dropout: Attention dropout probability
        bias: Whether to use bias in linear layers
        norm_type: Type of normalization ('rmsnorm' or 'layernorm')
        norm_eps: Epsilon for normalization layers
        rope_base: Base frequency for RoPE
        causal: Whether to use causal attention masking
    """
    # Core dimensions
    dim: int = 512
    num_heads: int = 8
    num_kv_heads: Optional[int] = None
    head_dim: Optional[int] = None
    
    # FFN dimensions
    ffn_dim: Optional[int] = None
    ffn_multiplier: float = 4.0
    
    # Sequence
    max_seq_len: int = 8192
    
    # Regularization
    dropout: float = 0.0
    attn_dropout: float = 0.0
    bias: bool = False
    
    # Normalization
    norm_type: Literal["rmsnorm", "layernorm"] = "rmsnorm"
    norm_eps: float = 1e-6
    
    # Position encoding
    rope_base: float = 10000.0
    
    # Attention
    causal: bool = True
    
    def __post_init__(self):
        """Compute derived values."""
        if self.head_dim is None:
            self.head_dim = self.dim // self.num_heads
        if self.num_kv_heads is None:
            self.num_kv_heads = self.num_heads
        if self.ffn_dim is None:
            self.ffn_dim = int(self.ffn_multiplier * self.dim)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x * norm).type_as(x) * self.weight


class DecoderBlock(nn.Module):
    """
    Single Decoder Block with standard softmax attention.
    
    Architecture (pre-norm with causal self-attention):
        x -> RMSNorm -> CausalAttention -> + -> RMSNorm -> SwiGLU -> +
        |__________________________________|   |_____________________|
    
    This is a building block that can be stacked or mixed with other
    block types (e.g., sparse attention) to create hybrid architectures.
    
    Args:
        config: BlockConfig with block hyperparameters
        layer_idx: Index of this layer (for potential layer-specific configs)
    """
    
    def __init__(self, config: BlockConfig, layer_idx: int = 0):
        super().__init__()
        
        self.config = config
        self.layer_idx = layer_idx
        
        # Normalization layers
        norm_cls = RMSNorm if config.norm_type == "rmsnorm" else nn.LayerNorm
        self.attn_norm = norm_cls(config.dim, eps=config.norm_eps)
        self.ffn_norm = norm_cls(config.dim, eps=config.norm_eps)
        
        # Causal self-attention with RoPE
        self.attention = RoPEAttention(
            embed_dim=config.dim,
            num_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
            dropout=config.attn_dropout,
            bias=config.bias,
            max_seq_len=config.max_seq_len,
            rope_base=config.rope_base,
            causal=config.causal,
        )
        
        # SwiGLU feed-forward network
        self.ffn = SwiGLU(
            in_features=config.dim,
            hidden_features=config.ffn_dim,
            out_features=config.dim,
            bias=config.bias,
            dropout=config.dropout,
        )
        
        # Dropout for residual connections
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()
    
    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        position_offset: int = 0,
    ) -> torch.Tensor:
        """
        Forward pass through the decoder block.
        
        Args:
            x: Input tensor of shape (batch, seq_len, dim)
            attn_mask: Optional attention mask
            position_offset: Position offset for KV cache during inference
        
        Returns:
            Output tensor of shape (batch, seq_len, dim)
        """
        # Self-attention with residual
        residual = x
        x = self.attn_norm(x)
        x = self.attention(x, attn_mask=attn_mask, position_offset=position_offset)
        x = self.dropout(x)
        x = residual + x
        
        # FFN with residual
        residual = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = self.dropout(x)
        x = residual + x
        
        return x
