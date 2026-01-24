"""
Causal Language Model with Hybrid Attention Architecture.

Composes standard and sparse decoder blocks into a complete decoder-only
language model. Architecture defined by flexible block patterns for research.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Optional, Tuple, Union, List

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
    
    Architecture can be defined in two ways:
    
    1. Simple mode (block counts):
        n_standard_blocks=4, n_sparse_blocks=4
        -> [4 standard] -> [4 sparse]
    
    2. Flexible mode (block pattern):
        block_pattern="SSSSSSFF" or ['S']*6 + ['F']*3
        -> [6 sparse] -> [3 full/standard]
        
        'S' = Sparse (HyperGraph attention)
        'F' = Full/Standard (causal attention)
    
    Note: If block_pattern is provided, it overrides n_standard/n_sparse_blocks.
    
    Attributes:
        # Model dimensions
        dim: Model dimension / embedding size
        num_heads: Number of attention heads
        num_kv_heads: Number of KV heads for GQA (None = MHA)
        
        # Block composition
        n_standard_blocks: Number of standard causal attention blocks (simple mode)
        n_sparse_blocks: Number of HyperGraph sparse attention blocks (simple mode)
        block_pattern: String/list defining block order (flexible mode)
                      'S'=Sparse, 'F'=Full. Example: "SSSSSSSFFF" or "SFSFSFSF"
        
        # Sparse attention
        num_hyper_nodes: Number of hyper nodes (K timelines) per head
        router_temperature: Gumbel-Softmax temperature for routing exploration
        entropy_weight: Weight for entropy regularization in router aux loss
        
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
    
    # Block composition (simple mode)
    n_standard_blocks: int = 4
    n_sparse_blocks: int = 4
    
    # Block composition (flexible mode) - overrides above if provided
    block_pattern: Optional[Union[str, List[str]]] = None
    
    # Sparse attention
    num_hyper_nodes: int = 4
    top_k: int = 1  # Number of timelines each token routes to (1=hard routing, 2=soft routing)
    router_temperature: float = 1.0  # Gumbel-Softmax temperature (higher = more exploration)
    entropy_weight: float = 0.01  # Weight for entropy regularization in router aux loss
    
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
        return len(self.get_block_pattern())
    
    def __post_init__(self):
        """Compute derived values and validate."""
        if self.num_kv_heads is None:
            self.num_kv_heads = self.num_heads
    
    def get_block_pattern(self) -> List[str]:
        """
        Get the block pattern as a list of 'S' (sparse) or 'F' (full/standard).
        
        Supports multiple input formats:
            - String: "SSSSFF" → ['S', 'S', 'S', 'S', 'F', 'F']
            - List of chars: ['S', 'S', 'F'] → ['S', 'S', 'F']
            - List of strings: ['SS', 'FF'] → ['S', 'S', 'F', 'F'] (flattened!)
            - Mixed: ['FF'] + 2*['SSFF'] → ['F', 'F', 'S', 'S', 'F', 'F', 'S', 'S', 'F', 'F']
        
        Returns:
            List of block types, e.g., ['S', 'S', 'S', 'F', 'F', 'F']
        """
        if self.block_pattern is not None:
            # Flexible mode: use provided pattern
            if isinstance(self.block_pattern, str):
                # String input: "SSSSFF"
                pattern = list(self.block_pattern)
            else:
                # List input: flatten multi-char strings
                # ['FF', 'SSSS'] → ['F', 'F', 'S', 'S', 'S', 'S']
                pattern = []
                for item in self.block_pattern:
                    pattern.extend(list(item))
            
            # Validate
            for i, b in enumerate(pattern):
                if b.upper() not in ('S', 'F'):
                    raise ValueError(f"Invalid block type '{b}' at position {i}. Use 'S' (sparse) or 'F' (full).")
            return [b.upper() for b in pattern]
        else:
            # Simple mode: standard blocks first, then sparse
            return ['F'] * self.n_standard_blocks + ['S'] * self.n_sparse_blocks
    
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
            router_temperature=self.router_temperature,
            entropy_weight=self.entropy_weight,
            top_k=self.top_k,
        )
    
    def describe_architecture(self) -> str:
        """Return a human-readable description of the architecture."""
        pattern = self.get_block_pattern()
        n_sparse = sum(1 for b in pattern if b == 'S')
        n_full = sum(1 for b in pattern if b == 'F')
        pattern_str = ''.join(pattern)
        return f"{len(pattern)} layers ({n_sparse} sparse, {n_full} full): {pattern_str}"


class CausalLM(nn.Module):
    """
    Decoder-only Causal Language Model with hybrid attention.
    
    Architecture:
        Embedding -> [blocks defined by pattern] -> Norm -> LM Head
    
    Block types:
        'F' (Full): Standard causal attention for global context
        'S' (Sparse): HyperGraph attention for efficient O(N²/K) processing
    
    Args:
        config: ModelConfig with model hyperparameters
        
    Example:
        ```python
        # Baseline: all standard attention
        config = ModelConfig(n_standard_blocks=8, n_sparse_blocks=0)
        
        # Hybrid: 4 standard + 4 sparse (simple mode)
        config = ModelConfig(n_standard_blocks=4, n_sparse_blocks=4)
        
        # Flexible: 6 sparse first, then 3 full (recommended for long context)
        config = ModelConfig(block_pattern="SSSSSSSFFF")
        
        # Interleaved: alternating pattern
        config = ModelConfig(block_pattern="SFSFSFSF")
        
        # Custom list
        config = ModelConfig(block_pattern=['S']*6 + ['F']*3)
        
        model = CausalLM(config)
        logits, aux_loss = model(input_ids)  # Returns (logits, load_balance_aux_loss)
        ```
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        
        self.config = config
        self.block_pattern = config.get_block_pattern()
        
        # Token embedding
        self.embedding = nn.Embedding(config.vocab_size, config.dim)
        
        # Embedding dropout
        self.embed_dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()
        
        # Build layers according to block pattern
        block_config = config.get_block_config()
        sparse_config = config.get_sparse_block_config()
        
        self.layers = nn.ModuleList()
        
        for i, block_type in enumerate(self.block_pattern):
            if block_type == 'F':
                self.layers.append(DecoderBlock(block_config, layer_idx=i))
            else:  # 'S'
                self.layers.append(SparseDecoderBlock(sparse_config, layer_idx=i))
        
        # Final normalization
        norm_cls = RMSNorm if config.norm_type == "rmsnorm" else nn.LayerNorm
        self.norm = norm_cls(config.dim, eps=config.norm_eps)
        
        # LM head
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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            input_ids: Token IDs of shape (batch, seq_len)
            attn_mask: Optional attention mask
            position_offset: Position offset for KV cache (future: autoregressive)
        
        Returns:
            logits: Output logits of shape (batch, seq_len, vocab_size)
            aux_loss: Load balance auxiliary loss (accumulated across sparse layers)
        """
        # Embed tokens
        x = self.embedding(input_ids)
        x = self.embed_dropout(x)
        
        # Accumulate aux_loss across sparse layers
        aux_loss = torch.tensor(0.0, device=input_ids.device)
        n_sparse_layers = 0
        
        # Pass through layers according to block pattern
        # NOTE: node_counts is NOT shared between layers - each sparse layer
        # computes its own independent node assignments and positions.
        
        for i, (layer, block_type) in enumerate(zip(self.layers, self.block_pattern)):
            if block_type == 'S':
                # Sparse block: HyperGraph attention
                x, _, layer_aux_loss = layer(
                    x,
                    attn_mask=attn_mask,
                    position_offset=position_offset,
                    node_counts=None,  # Each layer starts fresh per forward pass
                )
                aux_loss = aux_loss + layer_aux_loss
                n_sparse_layers += 1
            else:
                # Full block: standard causal attention
                x = layer(x, attn_mask=attn_mask, position_offset=position_offset)
        
        # Average aux_loss over sparse layers (if any)
        if n_sparse_layers > 0:
            aux_loss = aux_loss / n_sparse_layers
        
        # Final norm and LM head
        x = self.norm(x)
        logits = self.lm_head(x)
        
        return logits, aux_loss
    
    def num_parameters(self, exclude_embeddings: bool = False) -> int:
        """Count total number of parameters."""
        total = sum(p.numel() for p in self.parameters())
        if exclude_embeddings:
            total -= self.embedding.weight.numel()
            if not self.config.tie_embeddings:
                total -= self.lm_head.weight.numel()
        return total
    
    def __repr__(self) -> str:
        pattern = ''.join(self.block_pattern)
        n_sparse = sum(1 for b in self.block_pattern if b == 'S')
        n_full = sum(1 for b in self.block_pattern if b == 'F')
        return (
            f"CausalLM(\n"
            f"  dim={self.config.dim}, heads={self.config.num_heads},\n"
            f"  layers: {len(self.block_pattern)} ({n_full} full, {n_sparse} sparse)\n"
            f"  pattern: {pattern}\n"
            f"  num_hyper_nodes={self.config.num_hyper_nodes} (K),\n"
            f"  params={self.num_parameters():,}\n"
            f")"
        )
