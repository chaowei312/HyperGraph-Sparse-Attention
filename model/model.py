"""
Causal Language Model with Hybrid Attention Architecture.

Composes standard and sparse decoder blocks into a complete decoder-only
language model. Architecture defined by explicit block counts for research.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Tuple, Union

from .module import (
    BlockConfig,
    DecoderBlock,
    SparseBlockConfig,
    SparseDecoderBlock,
    RMSNorm,
)


@dataclass
class ModelConfig:
    """
    Configuration for the full CausalLM model.
    
    Architecture is defined by explicit block counts:
        [n_standard_blocks x DecoderBlock] -> [n_sparse_blocks x SparseDecoderBlock]
    
    This explicit control is preferred for research/ablation studies.
    
    Attributes:
        # Model dimensions
        dim: Model dimension / embedding size
        num_heads: Number of attention heads
        num_kv_heads: Number of KV heads for GQA (None = MHA)
        
        # Block composition (explicit)
        n_standard_blocks: Number of standard causal attention blocks
        n_sparse_blocks: Number of HyperGraph sparse attention blocks
        
        # Sparse attention
        num_hyper_nodes: Number of hyper nodes (K timelines) per head
        
        # Other
        vocab_size: Vocabulary size
        max_seq_len: Maximum sequence length
        ffn_multiplier: Multiplier for FFN hidden dim
        dropout: Dropout probability
        attn_dropout: Attention dropout probability
        bias: Whether to use bias in linear layers
        norm_type: Type of normalization
        norm_eps: Epsilon for normalization
        rope_base: Base frequency for RoPE
        tie_embeddings: Whether to tie input/output embeddings
    """
    # Model dimensions
    dim: int = 512
    num_heads: int = 8
    num_kv_heads: Optional[int] = None
    
    # Block composition (explicit control for research)
    n_standard_blocks: int = 4
    n_sparse_blocks: int = 4
    
    # Sparse attention
    num_hyper_nodes: int = 4
    
    # Vocabulary & sequence
    vocab_size: int = 50257  # GPT-2 tokenizer
    max_seq_len: int = 2048
    
    # FFN
    ffn_multiplier: float = 4.0
    
    # Regularization
    dropout: float = 0.0
    attn_dropout: float = 0.0
    bias: bool = False
    
    # Normalization
    norm_type: str = "rmsnorm"  # rmsnorm or layernorm
    norm_eps: float = 1e-6
    
    # Position encoding
    rope_base: float = 10000.0
    
    # Embeddings
    tie_embeddings: bool = True
    
    @property
    def depth(self) -> int:
        """Total number of layers."""
        return self.n_standard_blocks + self.n_sparse_blocks
    
    def __post_init__(self):
        """Compute derived values and validate."""
        if self.num_kv_heads is None:
            self.num_kv_heads = self.num_heads
    
    def get_block_config(self) -> BlockConfig:
        """Get BlockConfig for standard layers."""
        return BlockConfig(
            dim=self.dim,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            ffn_multiplier=self.ffn_multiplier,
            max_seq_len=self.max_seq_len,
            dropout=self.dropout,
            attn_dropout=self.attn_dropout,
            bias=self.bias,
            norm_type=self.norm_type,
            norm_eps=self.norm_eps,
            rope_base=self.rope_base,
            causal=True,
        )
    
    def get_sparse_block_config(self) -> SparseBlockConfig:
        """Get SparseBlockConfig for sparse layers."""
        return SparseBlockConfig(
            dim=self.dim,
            num_heads=self.num_heads,
            num_hyper_nodes=self.num_hyper_nodes,
            num_kv_heads=self.num_kv_heads,
            ffn_multiplier=self.ffn_multiplier,
            max_seq_len=self.max_seq_len,
            dropout=self.dropout,
            attn_dropout=self.attn_dropout,
            bias=self.bias,
            norm_type=self.norm_type,
            norm_eps=self.norm_eps,
            rope_base=self.rope_base,
        )


class CausalLM(nn.Module):
    """
    Decoder-only Causal Language Model with hybrid attention.
    
    Architecture:
        Embedding -> [n_standard x DecoderBlock] -> [n_sparse x SparseDecoderBlock] -> Norm -> LM Head
    
    Standard blocks provide full causal attention for global context.
    Sparse blocks use HyperGraph attention for efficient local processing.
    
    Args:
        config: ModelConfig with model hyperparameters
        
    Example:
        ```python
        # Baseline: all standard attention
        config = ModelConfig(n_standard_blocks=8, n_sparse_blocks=0)
        
        # Hybrid: 4 standard + 4 sparse
        config = ModelConfig(n_standard_blocks=4, n_sparse_blocks=4)
        
        # Sparse only
        config = ModelConfig(n_standard_blocks=0, n_sparse_blocks=8)
        
        model = CausalLM(config)
        logits = model(input_ids)
        ```
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        
        self.config = config
        
        # Token embedding
        self.embedding = nn.Embedding(config.vocab_size, config.dim)
        
        # Embedding dropout
        self.embed_dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()
        
        # Build layers: standard first, then sparse
        block_config = config.get_block_config()
        sparse_config = config.get_sparse_block_config()
        
        self.layers = nn.ModuleList()
        
        # Standard blocks
        for i in range(config.n_standard_blocks):
            self.layers.append(DecoderBlock(block_config, layer_idx=i))
        
        # Sparse blocks
        for i in range(config.n_sparse_blocks):
            layer_idx = config.n_standard_blocks + i
            self.layers.append(SparseDecoderBlock(sparse_config, layer_idx=layer_idx))
        
        # Final normalization
        norm_cls = RMSNorm if config.norm_type == "rmsnorm" else nn.LayerNorm
        self.norm = norm_cls(config.dim, eps=config.norm_eps)
        
        # LM head
        self.lm_head = nn.Linear(config.vocab_size, config.dim, bias=False)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        
        # Tie embeddings
        if config.tie_embeddings:
            self.lm_head.weight = self.embedding.weight
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with scaled normal distribution."""
        std = self.config.dim ** -0.5
        
        nn.init.normal_(self.embedding.weight, std=std)
        
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        position_offset: int = 0,
    ) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            input_ids: Token IDs of shape (batch, seq_len)
            attn_mask: Optional attention mask
            position_offset: Position offset for KV cache (future: autoregressive)
        
        Returns:
            logits: Output logits of shape (batch, seq_len, vocab_size)
        """
        # Embed tokens
        x = self.embedding(input_ids)
        x = self.embed_dropout(x)
        
        # Pass through layers
        # NOTE: node_counts is NOT shared between layers - each sparse layer
        # computes its own independent node assignments and positions.
        # node_counts is only used for KV cache continuation across forward passes.
        
        for i, layer in enumerate(self.layers):
            is_sparse = i >= self.config.n_standard_blocks
            
            if is_sparse:
                # Each layer gets fresh node_counts (None) for this sequence
                # Only pass node_counts for KV cache continuation (autoregressive inference)
                x, _ = layer(
                    x,
                    attn_mask=attn_mask,
                    position_offset=position_offset,
                    node_counts=None,  # Each layer starts fresh per forward pass
                )
            else:
                x = layer(x, attn_mask=attn_mask, position_offset=position_offset)
        
        # Final norm and LM head
        x = self.norm(x)
        logits = self.lm_head(x)
        
        return logits
    
    def num_parameters(self, exclude_embeddings: bool = False) -> int:
        """Count total number of parameters."""
        total = sum(p.numel() for p in self.parameters())
        if exclude_embeddings:
            total -= self.embedding.weight.numel()
            if not self.config.tie_embeddings:
                total -= self.lm_head.weight.numel()
        return total
    
    def __repr__(self) -> str:
        return (
            f"CausalLM(\n"
            f"  dim={self.config.dim}, heads={self.config.num_heads},\n"
            f"  blocks: {self.config.n_standard_blocks} standard + {self.config.n_sparse_blocks} sparse,\n"
            f"  num_hyper_nodes={self.config.num_hyper_nodes},\n"
            f"  params={self.num_parameters():,}\n"
            f")"
        )
