"""
Flash Attention implementation using PyTorch's native scaled_dot_product_attention.

Flash Attention is an IO-aware exact attention algorithm that uses tiling to reduce
memory reads/writes between GPU high bandwidth memory (HBM) and GPU on-chip SRAM.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FlashAttention(nn.Module):
    """
    Flash Attention module leveraging PyTorch's optimized SDPA implementation.
    
    Supports:
    - Flash Attention 2 (via torch.backends.cuda.sdp_kernel)
    - Memory efficient attention
    - Math fallback for compatibility
    
    Args:
        embed_dim: Total dimension of the model
        num_heads: Number of attention heads
        dropout: Dropout probability (default: 0.0)
        bias: Whether to use bias in projections (default: False)
        causal: Whether to apply causal masking (default: False)
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = False,
        causal: bool = False,
    ):
        super().__init__()
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.causal = causal
        self.scale = self.head_dim ** -0.5
        
        # QKV projection
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        
        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for Flash Attention.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)
            attn_mask: Optional attention mask of shape (batch_size, seq_len) or
                      (batch_size, 1, seq_len, seq_len)
        
        Returns:
            Output tensor of shape (batch_size, seq_len, embed_dim)
        """
        batch_size, seq_len, _ = x.shape
        
        # Compute QKV projections
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, S, D)
        q, k, v = qkv.unbind(0)  # Each: (B, H, S, D)
        
        # Use PyTorch's scaled_dot_product_attention (Flash Attention under the hood)
        dropout_p = self.dropout if self.training else 0.0
        
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=self.causal if attn_mask is None else False,
            scale=self.scale,
        )
        
        # Reshape and project output
        out = out.transpose(1, 2).contiguous()  # (B, S, H, D)
        out = out.reshape(batch_size, seq_len, self.embed_dim)
        out = self.out_proj(out)
        
        return out


class FlashCrossAttention(nn.Module):
    """
    Flash Cross-Attention module for encoder-decoder architectures.
    
    Args:
        embed_dim: Total dimension of the model
        num_heads: Number of attention heads
        dropout: Dropout probability (default: 0.0)
        bias: Whether to use bias in projections (default: False)
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.scale = self.head_dim ** -0.5
        
        # Separate projections for Q and KV
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.kv_proj = nn.Linear(embed_dim, 2 * embed_dim, bias=bias)
        
        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for Flash Cross-Attention.
        
        Args:
            x: Query tensor of shape (batch_size, tgt_len, embed_dim)
            context: Key/Value tensor of shape (batch_size, src_len, embed_dim)
            attn_mask: Optional attention mask
        
        Returns:
            Output tensor of shape (batch_size, tgt_len, embed_dim)
        """
        batch_size, tgt_len, _ = x.shape
        src_len = context.shape[1]
        
        # Compute Q from x, KV from context
        q = self.q_proj(x)
        q = q.reshape(batch_size, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        kv = self.kv_proj(context)
        kv = kv.reshape(batch_size, src_len, 2, self.num_heads, self.head_dim)
        kv = kv.permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)
        
        # Flash attention
        dropout_p = self.dropout if self.training else 0.0
        
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            scale=self.scale,
        )
        
        # Reshape and project output
        out = out.transpose(1, 2).contiguous()
        out = out.reshape(batch_size, tgt_len, self.embed_dim)
        out = self.out_proj(out)
        
        return out


class FlashMultiHeadAttention(nn.Module):
    """
    Multi-Head Attention with Flash Attention backend, supporting both
    self-attention and cross-attention in a unified interface.
    
    Args:
        embed_dim: Total dimension of the model
        num_heads: Number of attention heads
        num_kv_heads: Number of key-value heads for GQA (default: None = MHA)
        dropout: Dropout probability (default: 0.0)
        bias: Whether to use bias in projections (default: False)
        causal: Whether to apply causal masking (default: False)
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        dropout: float = 0.0,
        bias: bool = False,
        causal: bool = False,
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.causal = causal
        self.scale = self.head_dim ** -0.5
        
        assert embed_dim % num_heads == 0
        assert num_heads % self.num_kv_heads == 0
        
        self.num_kv_groups = num_heads // self.num_kv_heads
        
        # Projections
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, self.num_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, self.num_kv_heads * self.head_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass supporting both self and cross attention.
        
        Args:
            x: Query tensor of shape (batch_size, seq_len, embed_dim)
            context: Optional context for cross-attention
            attn_mask: Optional attention mask
        
        Returns:
            Output tensor of shape (batch_size, seq_len, embed_dim)
        """
        if context is None:
            context = x
            
        batch_size, tgt_len, _ = x.shape
        src_len = context.shape[1]
        
        # Project Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(context)
        v = self.v_proj(context)
        
        # Reshape for attention
        q = q.view(batch_size, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, src_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, src_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # Expand KV for Grouped Query Attention
        if self.num_kv_groups > 1:
            k = k.unsqueeze(2).expand(-1, -1, self.num_kv_groups, -1, -1)
            k = k.reshape(batch_size, self.num_heads, src_len, self.head_dim)
            v = v.unsqueeze(2).expand(-1, -1, self.num_kv_groups, -1, -1)
            v = v.reshape(batch_size, self.num_heads, src_len, self.head_dim)
        
        # Flash attention
        dropout_p = self.dropout if self.training else 0.0
        is_causal = self.causal and (context is x) and (attn_mask is None)
        
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=self.scale,
        )
        
        # Reshape and project output
        out = out.transpose(1, 2).contiguous()
        out = out.reshape(batch_size, tgt_len, self.embed_dim)
        out = self.out_proj(out)
        
        return out

