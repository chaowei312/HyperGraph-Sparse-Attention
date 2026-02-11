"""
Sparse Decoder Block with HyperGraph Attention.

Provides a decoder block using HyperGraph Sparse Attention where tokens
are routed to K hyper nodes with separate timelines. This is a drop-in
replacement for DecoderBlock, enabling hybrid architectures.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Tuple, Literal

from .hypergraph_attention import HyperGraphSparseAttention
from .swiglu import SwiGLU
from .block import BlockConfig, RMSNorm


@dataclass
class SparseBlockConfig:
    """
    Configuration for sparse decoder blocks with HyperGraph attention.
    
    Contains all standard block parameters plus sparse-specific settings.
    
    Attributes:
        dim: Model dimension / embedding size
        num_heads: Number of attention heads
        num_hyper_nodes: Number of hyper nodes (K parallel timelines)
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
        router_temperature: Gumbel-Softmax temperature (higher = more exploration)
        entropy_weight: Weight for entropy regularization in aux loss
    """
    # Core dimensions
    dim: int = 512
    num_heads: int = 8
    num_hyper_nodes: int = 4  # Sparse-specific: K parallel timelines
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
    
    # Router parameters (Gumbel-Softmax for preventing collapse)
    router_temperature: float = 1.0
    entropy_weight: float = 0.01
    top_k: int = 1  # Number of timelines each token routes to
    router_type: str = "linear"  # "linear" or "mlp" (2-layer with feature extraction)
    use_local_rope: bool = True  # Timeline-local (True) or global (False) RoPE
    use_rope_freq_exploration: bool = False  # Random position scaling per timeline (for length generalization)
    rope_freq_range: tuple = (1.0, 4.0)  # Range of position multipliers when freq_exploration is enabled
    use_confidence_gate: bool = False  # Gated attention (prevents attention sink problem)
    
    def __post_init__(self):
        """Compute derived values."""
        if self.head_dim is None:
            self.head_dim = self.dim // self.num_heads
        if self.num_kv_heads is None:
            self.num_kv_heads = self.num_heads
        if self.ffn_dim is None:
            self.ffn_dim = int(self.ffn_multiplier * self.dim)
    
    @classmethod
    def from_block_config(
        cls, 
        config: BlockConfig, 
        num_hyper_nodes: int = 4,
        router_temperature: float = 1.0,
        entropy_weight: float = 0.01,
    ) -> "SparseBlockConfig":
        """Create SparseBlockConfig from a standard BlockConfig."""
        return cls(
            dim=config.dim,
            num_heads=config.num_heads,
            num_hyper_nodes=num_hyper_nodes,
            num_kv_heads=config.num_kv_heads,
            head_dim=config.head_dim,
            ffn_dim=config.ffn_dim,
            ffn_multiplier=config.ffn_multiplier,
            max_seq_len=config.max_seq_len,
            dropout=config.dropout,
            attn_dropout=config.attn_dropout,
            bias=config.bias,
            norm_type=config.norm_type,
            norm_eps=config.norm_eps,
            rope_base=config.rope_base,
            router_temperature=router_temperature,
            entropy_weight=entropy_weight,
        )


class SparseDecoderBlock(nn.Module):
    """
    Decoder Block with HyperGraph Sparse Attention.
    
    Architecture (pre-norm with sparse causal self-attention):
        x -> RMSNorm -> HyperGraphSparseAttn -> + -> RMSNorm -> SwiGLU -> +
        |_____________________________________|   |_____________________|
    
    This is a drop-in replacement for DecoderBlock. Tokens are assigned
    to one of K hyper nodes, and attention is computed only within the
    same node (with separate RoPE per node/timeline).
    
    Usage in hybrid architecture:
        config = BlockConfig(dim=512, num_heads=8)
        sparse_config = SparseBlockConfig(dim=512, num_heads=8, num_hyper_nodes=4)
        
        layers = [
            DecoderBlock(config, 0),             # Standard attention
            SparseDecoderBlock(sparse_config, 1), # Sparse attention
            DecoderBlock(config, 2),             # Standard attention
            ...
        ]
    
    Args:
        config: SparseBlockConfig with block and sparse hyperparameters
        layer_idx: Index of this layer (for layer-specific behavior)
    """
    
    def __init__(self, config: SparseBlockConfig, layer_idx: int = 0):
        super().__init__()
        
        self.config = config
        self.layer_idx = layer_idx
        self.num_hyper_nodes = config.num_hyper_nodes
        
        # Normalization layers
        norm_cls = RMSNorm if config.norm_type == "rmsnorm" else nn.LayerNorm
        self.attn_norm = norm_cls(config.dim, eps=config.norm_eps)
        self.ffn_norm = norm_cls(config.dim, eps=config.norm_eps)
        
        # HyperGraph Sparse Attention with Gumbel-Softmax routing
        self.attention = HyperGraphSparseAttention(
            embed_dim=config.dim,
            num_heads=config.num_heads,
            num_hyper_nodes=config.num_hyper_nodes,
            num_kv_heads=config.num_kv_heads,
            dropout=config.attn_dropout,
            bias=config.bias,
            max_seq_len=config.max_seq_len,
            rope_base=config.rope_base,
            router_temperature=config.router_temperature,
            entropy_weight=config.entropy_weight,
            top_k=config.top_k,
            router_type=config.router_type,
            use_local_rope=config.use_local_rope,
            use_rope_freq_exploration=config.use_rope_freq_exploration,
            rope_freq_range=config.rope_freq_range,
            use_confidence_gate=config.use_confidence_gate,
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
        node_counts: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """
        Forward pass through the sparse decoder block.
        
        Args:
            x: Input tensor of shape (batch, seq_len, dim)
            attn_mask: Optional attention mask
            position_offset: Position offset for KV cache during inference
            node_counts: Node counts for tracking positions during inference
                         Shape: (batch, num_heads, num_hyper_nodes)
        
        Returns:
            output: Output tensor of shape (batch, seq_len, dim)
            node_counts: Updated node counts (None during training)
            aux_loss: Load balance auxiliary loss (scalar)
        """
        # Self-attention with residual
        residual = x
        x = self.attn_norm(x)
        x, updated_node_counts, aux_loss = self.attention(
            x,
            attn_mask=attn_mask,
            position_offset=position_offset,
            node_counts=node_counts,
        )
        x = self.dropout(x)
        x = residual + x
        
        # FFN with residual
        residual = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = self.dropout(x)
        x = residual + x
        
        return x, updated_node_counts, aux_loss
