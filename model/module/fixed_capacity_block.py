"""
Fixed-Capacity HyperGraph Decoder Block and Model Components.

Provides complete building blocks for training hybrid models with
Fixed-Capacity HyperGraph sparse attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, List, Tuple

from .hypergraph_fixed_capacity import HyperGraphFixedCapacityBatched
from .swiglu import SwiGLU


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x * norm).type_as(x) * self.weight


class FullCausalAttention(nn.Module):
    """Standard causal self-attention with RoPE."""
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        head_dim: Optional[int] = None,
        num_kv_heads: Optional[int] = None,
        max_seq_len: int = 8192,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim or dim // num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.num_kv_groups = num_heads // self.num_kv_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(dim, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(num_heads * self.head_dim, dim, bias=False)
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # RoPE
        self._build_rope_cache(max_seq_len)
    
    def _build_rope_cache(self, max_seq_len: int, base: float = 10000.0):
        inv_freq = 1.0 / (base ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        t = torch.arange(max_seq_len)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
    
    def _apply_rope(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[2]
        cos = self.cos_cached[:seq_len].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cached[:seq_len].unsqueeze(0).unsqueeze(0)
        
        x1, x2 = x[..., :self.head_dim//2], x[..., self.head_dim//2:]
        rotated = torch.cat([-x2, x1], dim=-1)
        return x * cos + rotated * sin
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, D = x.shape
        H = self.num_heads
        d = self.head_dim
        
        q = self.q_proj(x).view(B, N, H, d).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.num_kv_heads, d).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.num_kv_heads, d).transpose(1, 2)
        
        # GQA expansion
        if self.num_kv_groups > 1:
            k = k.unsqueeze(2).expand(-1, -1, self.num_kv_groups, -1, -1)
            k = k.reshape(B, H, N, d)
            v = v.unsqueeze(2).expand(-1, -1, self.num_kv_groups, -1, -1)
            v = v.reshape(B, H, N, d)
        
        # Apply RoPE
        q = self._apply_rope(q)
        k = self._apply_rope(k)
        
        # Attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        mask = torch.triu(torch.ones(N, N, device=x.device, dtype=torch.bool), diagonal=1)
        attn = attn.masked_fill(mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, N, H * d)
        
        return self.out_proj(out), torch.tensor(0.0, device=x.device)


@dataclass
class FixedCapacityBlockConfig:
    """Configuration for Fixed-Capacity decoder block."""
    dim: int = 512
    num_heads: int = 8
    num_kv_heads: Optional[int] = None
    head_dim: Optional[int] = None
    num_timelines: int = 4
    ffn_multiplier: float = 4.0
    max_seq_len: int = 8192
    dropout: float = 0.0
    is_sparse: bool = True  # True = HyperGraph sparse, False = Full attention


class FixedCapacityDecoderBlock(nn.Module):
    """
    Decoder block with either Fixed-Capacity HyperGraph or Full attention.
    
    Can be used to build hybrid models with pattern like "FSSFSSFSSFSSFF".
    """
    
    def __init__(self, config: FixedCapacityBlockConfig, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        self.norm1 = RMSNorm(config.dim)
        self.norm2 = RMSNorm(config.dim)
        
        if config.is_sparse:
            self.attn = HyperGraphFixedCapacityBatched(
                embed_dim=config.dim,
                num_heads=config.num_heads,
                num_kv_heads=config.num_kv_heads,
                num_timelines=config.num_timelines,
                max_seq_len=config.max_seq_len,
                dropout=config.dropout,
            )
        else:
            self.attn = FullCausalAttention(
                dim=config.dim,
                num_heads=config.num_heads,
                num_kv_heads=config.num_kv_heads,
                max_seq_len=config.max_seq_len,
                dropout=config.dropout,
            )
        
        ffn_dim = int(config.dim * config.ffn_multiplier)
        self.ffn = SwiGLU(config.dim, ffn_dim, config.dim)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        attn_out, aux_loss = self.attn(self.norm1(x))
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x, aux_loss


@dataclass
class FixedCapacityModelConfig:
    """Configuration for complete Fixed-Capacity HyperGraph model."""
    dim: int = 512
    num_heads: int = 8
    num_kv_heads: Optional[int] = None
    head_dim: Optional[int] = None
    num_layers: int = 14
    num_timelines: int = 4
    vocab_size: int = 50257
    max_seq_len: int = 8192
    ffn_multiplier: float = 4.0
    dropout: float = 0.0
    block_pattern: str = "FSSFSSFSSFSSFF"  # F=Full, S=Sparse
    aux_loss_weight: float = 0.01
    tie_embeddings: bool = True


class FixedCapacityCausalLM(nn.Module):
    """
    Causal Language Model with Fixed-Capacity HyperGraph attention.
    
    Supports hybrid patterns mixing Full (F) and Sparse (S) attention layers.
    Default pattern: FSSFSSFSSFSSFF (optimal hybrid from ablations)
    """
    
    def __init__(self, config: FixedCapacityModelConfig):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.embed = nn.Embedding(config.vocab_size, config.dim)
        
        # Decoder layers
        self.layers = nn.ModuleList()
        pattern = config.block_pattern
        
        for i in range(config.num_layers):
            is_sparse = pattern[i % len(pattern)] == 'S'
            
            block_config = FixedCapacityBlockConfig(
                dim=config.dim,
                num_heads=config.num_heads,
                num_kv_heads=config.num_kv_heads,
                head_dim=config.head_dim,
                num_timelines=config.num_timelines,
                ffn_multiplier=config.ffn_multiplier,
                max_seq_len=config.max_seq_len,
                dropout=config.dropout,
                is_sparse=is_sparse,
            )
            self.layers.append(FixedCapacityDecoderBlock(block_config, i))
        
        # Output
        self.norm = RMSNorm(config.dim)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        
        # Tie embeddings
        if config.tie_embeddings:
            self.lm_head.weight = self.embed.weight
        
        self._init_weights()
        
        # Log architecture
        sparse_count = sum(1 for i in range(config.num_layers) if pattern[i % len(pattern)] == 'S')
        full_count = config.num_layers - sparse_count
        print(f"FixedCapacityCausalLM: {config.num_layers}L, {sparse_count}S/{full_count}F, K={config.num_timelines}")
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            input_ids: (batch, seq_len) token IDs
            labels: (batch, seq_len) target IDs for loss computation
        
        Returns:
            logits: (batch, seq_len, vocab_size)
            aux_loss: Auxiliary load balance loss
            loss: Cross-entropy loss if labels provided
        """
        x = self.embed(input_ids)
        
        total_aux = 0.0
        for layer in self.layers:
            x, aux = layer(x)
            total_aux = total_aux + aux
        
        x = self.norm(x)
        logits = self.lm_head(x)
        
        avg_aux = total_aux / len(self.layers)
        
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                labels.view(-1),
            )
            loss = loss + self.config.aux_loss_weight * avg_aux
        
        return logits, avg_aux, loss
    
    def num_parameters(self, exclude_embeddings: bool = False) -> int:
        """Count model parameters."""
        total = sum(p.numel() for p in self.parameters())
        if exclude_embeddings:
            total -= self.embed.weight.numel()
            if not self.config.tie_embeddings:
                total -= self.lm_head.weight.numel()
        return total
    
    @classmethod
    def from_config(cls, **kwargs) -> "FixedCapacityCausalLM":
        """Create model from keyword arguments."""
        config = FixedCapacityModelConfig(**kwargs)
        return cls(config)


# Convenience functions

def create_hybrid_model(
    dim: int = 512,
    num_heads: int = 8,
    num_layers: int = 14,
    num_timelines: int = 4,
    vocab_size: int = 50257,
    pattern: str = "FSSFSSFSSFSSFF",
) -> FixedCapacityCausalLM:
    """
    Create a hybrid Fixed-Capacity HyperGraph model.
    
    Default pattern FSSFSSFSSFSSFF provides good balance between
    full attention quality and sparse attention efficiency.
    """
    config = FixedCapacityModelConfig(
        dim=dim,
        num_heads=num_heads,
        num_layers=num_layers,
        num_timelines=num_timelines,
        vocab_size=vocab_size,
        block_pattern=pattern,
    )
    return FixedCapacityCausalLM(config)


def create_pure_sparse_model(
    dim: int = 512,
    num_heads: int = 8,
    num_layers: int = 14,
    num_timelines: int = 4,
    vocab_size: int = 50257,
) -> FixedCapacityCausalLM:
    """Create a pure sparse Fixed-Capacity HyperGraph model (all S layers)."""
    pattern = "S" * num_layers
    return create_hybrid_model(dim, num_heads, num_layers, num_timelines, vocab_size, pattern)

