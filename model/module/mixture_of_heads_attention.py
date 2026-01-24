"""
Mixture of Heads (MoH) Sparse Attention Module.

Alternative architecture where each attention head IS a timeline,
and a router selects top-k heads per token. This is simpler than
the HyperGraph approach (K timelines per head) and has:

  - Same KV cache as normal MHA (no memory penalty)
  - Simpler routing (like MoE, single top-k operation)
  - Less Python overhead (no complex per-head indexing)
  - 4x compute reduction with K=4 sparsity factor

Architecture:
  - H' = H × K total heads (e.g., 32 = 8 × 4)
  - Each head maintains independent timeline with own RoPE
  - Router selects top_k heads (e.g., 8) per token
  - Only selected heads compute attention
  - Output is weighted combination from selected heads

Complexity:
  - Normal MHA (8 heads): 8 × N² = 8N²
  - MoH (32 heads, top_k=8): 8 × (N/4)² = 8 × N²/16 = N²/2
  - Speedup: 16x in attention, 4x overall (with routing overhead)

Trade-off:
  - 4x more head parameters (32 vs 8 heads worth of Q,K,V,O)
  - But inference memory same as MHA (8 KV pairs per token)

Requirements:
  - xformers (for efficient block-sparse attention)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .rope import rotate_half


# Import xformers (required)
try:
    from xformers.ops import memory_efficient_attention, fmha
    XFORMERS_AVAILABLE = True
except ImportError:
    XFORMERS_AVAILABLE = False
    raise ImportError(
        "xformers is required for Mixture of Heads Attention. "
        "Install with: pip install xformers"
    )


class MoHRoPE(nn.Module):
    """
    Rotary Position Embedding for MoH attention.
    
    Each head maintains its own position counter, enabling K× context
    extension without RoPE degradation.
    """
    
    def __init__(
        self,
        head_dim: int,
        max_seq_len: int = 8192,
        base: float = 10000.0,
    ):
        super().__init__()
        
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Compute inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # Pre-compute cos/sin for all possible positions
        self._build_cache(max_seq_len)
    
    def _build_cache(self, max_seq_len: int):
        """Pre-compute cos/sin values for positions 0 to max_seq_len-1."""
        t = torch.arange(max_seq_len, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)


class MixtureOfHeadsAttention(nn.Module):
    """
    Mixture of Heads (MoH) Sparse Attention.
    
    Each head IS a timeline. Router selects top_k heads per token.
    Only selected heads compute attention for that token.
    
    This is conceptually similar to MoE but applied to attention heads
    instead of FFN experts.
    
    Args:
        embed_dim: Total model dimension
        num_heads: Total number of heads (each head = 1 timeline)
        active_heads: Number of heads to activate per token (top_k)
        head_dim: Dimension per head (if None, computed from embed_dim/active_heads)
        dropout: Dropout probability
        bias: Whether to use bias in projections
        max_seq_len: Maximum sequence length per head
        rope_base: Base frequency for RoPE
        router_temperature: Gumbel-Softmax temperature
        entropy_weight: Weight for entropy regularization
        load_balance_weight: Weight for load balance loss
    
    Example:
        # Equivalent to HyperGraph with H=8 heads, K=4 timelines
        # but with simpler routing and same memory as MHA
        moh = MixtureOfHeadsAttention(
            embed_dim=512,
            num_heads=32,      # 32 total heads (= 8 × 4)
            active_heads=8,    # activate 8 per token
        )
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 32,
        active_heads: int = 8,
        head_dim: Optional[int] = None,
        dropout: float = 0.0,
        bias: bool = False,
        max_seq_len: int = 8192,
        rope_base: float = 10000.0,
        router_temperature: float = 1.0,
        entropy_weight: float = 0.01,
        load_balance_weight: float = 0.01,
    ):
        super().__init__()
        
        if not XFORMERS_AVAILABLE:
            raise ImportError("xformers is required for Mixture of Heads Attention")
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads  # Total heads (H')
        self.active_heads = min(active_heads, num_heads)  # Top-k to activate
        self.sparsity_factor = num_heads // active_heads  # K = H' / top_k
        
        # Head dimension: either specified or computed
        if head_dim is not None:
            self.head_dim = head_dim
        else:
            # Default: same total capacity as standard attention
            # Standard: embed_dim = num_active_heads × head_dim
            self.head_dim = embed_dim // active_heads
        
        self.dropout = dropout
        self.scale = self.head_dim ** -0.5
        
        # Router parameters
        self.router_temperature = router_temperature
        self.entropy_weight = entropy_weight
        self.load_balance_weight = load_balance_weight
        
        # Q, K, V projections for ALL heads
        # Note: This uses more parameters than standard MHA
        # Total params: embed_dim × (num_heads × head_dim × 3)
        self.q_proj = nn.Linear(embed_dim, num_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, num_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, num_heads * self.head_dim, bias=bias)
        
        # Router: selects which heads each token uses
        self.router = nn.Linear(embed_dim, num_heads, bias=bias)
        
        # Output projection: from active_heads back to embed_dim
        # This projects from the combined active head outputs
        self.out_proj = nn.Linear(active_heads * self.head_dim, embed_dim, bias=bias)
        
        # Per-head RoPE (shared cache)
        self.rope = MoHRoPE(
            head_dim=self.head_dim,
            max_seq_len=max_seq_len,
            base=rope_base,
        )
        
        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
    
    def _compute_routing(
        self,
        x: torch.Tensor,
        temperature: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute top-k head routing with Gumbel noise.
        
        Args:
            x: Input tensor (batch, seq, embed_dim)
            temperature: Gumbel temperature
        
        Returns:
            head_indices: (batch, seq, active_heads) - selected head indices
            head_weights: (batch, seq, active_heads) - routing weights (sum to 1)
            soft_probs: (batch, seq, num_heads) - full soft probs for aux loss
        """
        if temperature is None:
            temperature = self.router_temperature
        
        batch, seq_len, _ = x.shape
        
        # Compute router logits: (batch, seq, num_heads)
        router_logits = self.router(x)
        
        # Add Gumbel noise during training
        if self.training:
            u = torch.rand_like(router_logits)
            gumbel_noise = -torch.log(-torch.log(u + 1e-8) + 1e-8)
            noisy_logits = (router_logits + gumbel_noise) / temperature
        else:
            noisy_logits = router_logits
        
        # Soft probabilities for auxiliary loss
        soft_probs = F.softmax(noisy_logits, dim=-1)
        
        # Top-k selection
        top_k_logits, head_indices = noisy_logits.topk(self.active_heads, dim=-1)
        
        # Renormalize weights over selected heads
        head_weights = F.softmax(top_k_logits, dim=-1)
        
        return head_indices, head_weights, soft_probs
    
    def _compute_aux_loss(
        self,
        soft_probs: torch.Tensor,
        head_indices: torch.Tensor,
        router_logits: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute auxiliary loss for load balancing and entropy.
        
        Args:
            soft_probs: (batch, seq, num_heads) - routing probabilities
            head_indices: (batch, seq, active_heads) - selected heads
            router_logits: Optional logits for z-loss
        
        Returns:
            aux_loss: Combined auxiliary loss (scalar)
        """
        batch, seq_len, H = soft_probs.shape
        
        # === 1. LOAD BALANCE LOSS ===
        # f_h: fraction of tokens routed to each head
        # We use the primary (highest weight) head for counting
        primary_heads = head_indices[..., 0]  # (batch, seq)
        one_hot = F.one_hot(primary_heads, H).float()  # (batch, seq, H)
        f = one_hot.mean(dim=1)  # (batch, H) - fraction per head
        
        # p_h: average routing probability for each head
        p = soft_probs.mean(dim=1)  # (batch, H)
        
        # Balance loss: H × sum_h(f_h × p_h)
        balance_loss = H * (f * p).sum(dim=-1).mean()
        
        # === 2. ENTROPY REGULARIZATION ===
        # Encourage exploration (high entropy = uncertain = good)
        entropy = -(soft_probs * torch.log(soft_probs + 1e-8)).sum(dim=-1)
        entropy_loss = -entropy.mean()  # Negative to maximize entropy
        
        # === 3. Z-LOSS (optional) ===
        z_loss = 0.0
        if router_logits is not None:
            z_loss = (router_logits.logsumexp(dim=-1) ** 2).mean() * 0.01
        
        # Combine
        aux_loss = (
            self.load_balance_weight * balance_loss +
            self.entropy_weight * entropy_loss +
            z_loss
        )
        
        return aux_loss
    
    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        position_offset: int = 0,
        head_counts: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """
        Forward pass with top-k head routing.
        
        Each token selects top_k heads. Only those heads compute attention.
        Each head maintains its own position counter for RoPE.
        
        Args:
            x: Input tensor (batch, seq, embed_dim)
            attn_mask: Optional attention mask (unused, for API compatibility)
            position_offset: Position offset for KV cache
            head_counts: Position counters per head (batch, num_heads)
        
        Returns:
            output: (batch, seq, embed_dim)
            updated_head_counts: (batch, num_heads) for KV cache
            aux_loss: Load balance auxiliary loss
        """
        batch, seq_len, _ = x.shape
        device = x.device
        dtype = x.dtype
        H = self.num_heads
        K = self.active_heads
        
        assert batch == 1, "MoH Attention currently requires batch_size=1"
        
        # Initialize head counts if not provided
        if head_counts is None:
            head_counts = torch.zeros(batch, H, device=device, dtype=torch.long)
        
        # === ROUTING: Select top-k heads per token ===
        head_indices, head_weights, soft_probs = self._compute_routing(x)
        # head_indices: (batch, seq, active_heads)
        # head_weights: (batch, seq, active_heads) - sum to 1
        
        # Compute auxiliary loss
        router_logits = self.router(x)
        aux_loss = self._compute_aux_loss(soft_probs, head_indices, router_logits)
        
        # === COMPUTE Q, K, V for ALL heads ===
        # Shape: (batch, seq, num_heads, head_dim)
        q_all = self.q_proj(x).view(batch, seq_len, H, self.head_dim)
        k_all = self.k_proj(x).view(batch, seq_len, H, self.head_dim)
        v_all = self.v_proj(x).view(batch, seq_len, H, self.head_dim)
        
        # === GATHER SELECTED HEADS ===
        # Expand indices for gathering: (batch, seq, active_heads, 1)
        gather_idx = head_indices.unsqueeze(-1).expand(-1, -1, -1, self.head_dim)
        
        # Gather Q, K, V for selected heads: (batch, seq, active_heads, head_dim)
        q = torch.gather(q_all, 2, gather_idx)
        k = torch.gather(k_all, 2, gather_idx)
        v = torch.gather(v_all, 2, gather_idx)
        
        # === BUILD BLOCKS FOR ATTENTION ===
        # Group tokens by their selected heads
        # Each head processes only the tokens that selected it
        
        # Flatten head indices to create (head_id, token_id) pairs
        # head_indices: (1, seq, active_heads) -> (seq × active_heads,) pairs
        flat_heads = head_indices[0].flatten()  # (seq × active_heads,)
        flat_tokens = torch.arange(seq_len, device=device).unsqueeze(1).expand(-1, K).flatten()
        flat_k_idx = torch.arange(K, device=device).unsqueeze(0).expand(seq_len, -1).flatten()
        
        # Sort by head to group tokens per head
        sort_order = flat_heads.argsort()
        sorted_heads = flat_heads[sort_order]
        sorted_tokens = flat_tokens[sort_order]
        sorted_k_idx = flat_k_idx[sort_order]
        
        # Count tokens per head
        head_token_counts = torch.bincount(sorted_heads, minlength=H)
        seqlens = head_token_counts[head_token_counts > 0].tolist()
        
        if len(seqlens) == 0:
            # Edge case: no tokens (shouldn't happen)
            out = torch.zeros(batch, seq_len, self.embed_dim, device=device, dtype=dtype)
            return out, head_counts, aux_loss
        
        # === GATHER Q, K, V IN BLOCK ORDER ===
        # q, k, v are (batch, seq, active_heads, head_dim)
        # We need to reorder to block order: all tokens for head0, then head1, etc.
        
        # Flatten q, k, v: (batch, seq × active_heads, head_dim)
        q_flat = q.view(batch, seq_len * K, self.head_dim)
        k_flat = k.view(batch, seq_len * K, self.head_dim)
        v_flat = v.view(batch, seq_len * K, self.head_dim)
        
        # Create gather index: token_idx * K + k_idx gives flat position
        flat_positions = sorted_tokens * K + sorted_k_idx
        
        # Gather in block order
        q_ordered = q_flat[:, flat_positions, :]
        k_ordered = k_flat[:, flat_positions, :]
        v_ordered = v_flat[:, flat_positions, :]
        
        # === COMPUTE POSITIONS (per-head RoPE) ===
        # Each head has independent position counter
        # Position = count of previous tokens in this head + offset
        
        # Get cumulative counts within each head group
        # Use segment cumsum trick
        total_tokens = sorted_heads.shape[0]
        ones = torch.ones(total_tokens, device=device, dtype=torch.long)
        
        # Mark head boundaries
        head_changes = torch.ones(total_tokens, device=device, dtype=torch.long)
        head_changes[1:] = (sorted_heads[1:] != sorted_heads[:-1]).long()
        
        # Cumsum within segments
        cumsum = ones.cumsum(dim=0)
        boundary_cumsum = (head_changes * cumsum)
        
        # Forward fill boundary values
        boundary_cumsum_max = boundary_cumsum.cummax(dim=0)[0]
        positions_in_head = cumsum - boundary_cumsum_max
        
        # Add head-specific offset from previous forward passes
        head_offsets = head_counts[0, sorted_heads]  # (total_tokens,)
        positions = positions_in_head + head_offsets
        positions = positions.clamp(0, self.rope.max_seq_len - 1)
        
        # === APPLY ROPE ===
        cos = self.rope.cos_cached[positions]
        sin = self.rope.sin_cached[positions]
        
        q_ordered = (q_ordered * cos.unsqueeze(0)) + (rotate_half(q_ordered) * sin.unsqueeze(0))
        k_ordered = (k_ordered * cos.unsqueeze(0)) + (rotate_half(k_ordered) * sin.unsqueeze(0))
        
        # === XFORMERS BLOCK-DIAGONAL CAUSAL ATTENTION ===
        # Each block = one head's tokens
        q_xf = q_ordered.unsqueeze(2)  # (batch, total, 1, head_dim)
        k_xf = k_ordered.unsqueeze(2)
        v_xf = v_ordered.unsqueeze(2)
        
        attn_bias = fmha.attn_bias.BlockDiagonalCausalMask.from_seqlens(seqlens)
        
        out_ordered = memory_efficient_attention(
            q_xf, k_xf, v_xf,
            attn_bias=attn_bias,
            scale=self.scale,
        ).squeeze(2)  # (batch, total, head_dim)
        
        # === SCATTER BACK AND WEIGHT ===
        # Scatter outputs back to (batch, seq, active_heads, head_dim)
        out_flat = torch.zeros(batch, seq_len * K, self.head_dim, device=device, dtype=dtype)
        out_flat[:, flat_positions, :] = out_ordered
        
        out = out_flat.view(batch, seq_len, K, self.head_dim)
        
        # Weight by routing weights: (batch, seq, active_heads, 1)
        weights = head_weights.unsqueeze(-1)
        out = (out * weights).sum(dim=2)  # (batch, seq, head_dim)
        
        # Actually we want (batch, seq, active_heads × head_dim) for out_proj
        out_combined = out_flat.view(batch, seq_len, K * self.head_dim)
        
        # Weighted combination before projection
        # For simplicity, we use the weighted sum approach
        out_weighted = (out_flat.view(batch, seq_len, K, self.head_dim) * weights).view(
            batch, seq_len, K * self.head_dim
        )
        
        # === UPDATE HEAD COUNTS ===
        # Count how many tokens each head received
        tokens_per_head = torch.bincount(flat_heads, minlength=H).unsqueeze(0)
        updated_head_counts = head_counts + tokens_per_head
        
        # === OUTPUT PROJECTION ===
        out = self.out_proj(out_weighted)
        
        return out, updated_head_counts, aux_loss


class GQAMixtureOfHeadsAttention(nn.Module):
    """
    GQA-style Mixture of Heads Sparse Attention.
    
    Key difference from standard MoH:
    - Multiple Q heads share each KV group
    - Router selects which KV groups each token uses
    - All Q heads within selected KV groups are activated
    
    Configuration (6 KV groups, 4 active, 2 Q per KV):
    - 6 KV groups total, each token selects 4
    - Each KV group has 2 Q heads
    - Total Q heads: 12, Q heads per token: 8
    - Tokens per KV group: 2N/3
    - FLOPs: 5.33N² (1.33× baseline)
    - Wall-clock: 4N²/9 (2.25× faster via parallelism)
    
    Args:
        embed_dim: Total model dimension
        num_kv_groups: Total number of KV groups (M)
        active_kv_groups: Number of KV groups to activate per token (A)
        q_heads_per_kv: Number of Q heads per KV group
        head_dim: Dimension per head
        dropout: Dropout probability
        bias: Whether to use bias in projections
        max_seq_len: Maximum sequence length
        rope_base: Base frequency for RoPE
        router_temperature: Gumbel-Softmax temperature
        entropy_weight: Weight for entropy regularization
        load_balance_weight: Weight for load balance loss
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_kv_groups: int = 6,
        active_kv_groups: int = 4,
        q_heads_per_kv: int = 2,
        head_dim: Optional[int] = None,
        dropout: float = 0.0,
        bias: bool = False,
        max_seq_len: int = 8192,
        rope_base: float = 10000.0,
        router_temperature: float = 1.0,
        entropy_weight: float = 0.01,
        load_balance_weight: float = 0.01,
    ):
        super().__init__()
        
        if not XFORMERS_AVAILABLE:
            raise ImportError("xformers is required for GQA MoH Attention")
        
        self.embed_dim = embed_dim
        self.num_kv_groups = num_kv_groups  # M
        self.active_kv_groups = min(active_kv_groups, num_kv_groups)  # A
        self.q_heads_per_kv = q_heads_per_kv  # Q heads per KV group
        
        # Total Q heads and active Q heads per token
        self.total_q_heads = num_kv_groups * q_heads_per_kv
        self.active_q_heads = self.active_kv_groups * q_heads_per_kv
        
        # Head dimension
        if head_dim is not None:
            self.head_dim = head_dim
        else:
            # Default: match baseline's total Q capacity
            self.head_dim = embed_dim // self.active_q_heads
        
        self.dropout = dropout
        self.scale = self.head_dim ** -0.5
        
        # Router parameters
        self.router_temperature = router_temperature
        self.entropy_weight = entropy_weight
        self.load_balance_weight = load_balance_weight
        
        # Q projection: total_q_heads * head_dim
        self.q_proj = nn.Linear(embed_dim, self.total_q_heads * self.head_dim, bias=bias)
        
        # K, V projections: num_kv_groups * head_dim (shared across Q heads in group)
        self.k_proj = nn.Linear(embed_dim, num_kv_groups * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, num_kv_groups * self.head_dim, bias=bias)
        
        # Router: selects which KV groups each token uses
        self.router = nn.Linear(embed_dim, num_kv_groups, bias=bias)
        
        # Output projection: from active_q_heads * head_dim to embed_dim
        self.out_proj = nn.Linear(self.active_q_heads * self.head_dim, embed_dim, bias=bias)
        
        # Per-group RoPE (shared cache)
        self.rope = MoHRoPE(
            head_dim=self.head_dim,
            max_seq_len=max_seq_len,
            base=rope_base,
        )
        
        self.attn_dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize router and output projection."""
        nn.init.kaiming_uniform_(self.router.weight, a=5**0.5)
        if self.router.bias is not None:
            nn.init.zeros_(self.router.bias)
        nn.init.kaiming_uniform_(self.out_proj.weight, a=5**0.5)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)
    
    def _compute_routing(
        self,
        x: torch.Tensor,
        temperature: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute top-k KV group routing with Gumbel noise.
        
        Returns:
            kv_indices: (batch, seq, active_kv_groups) - selected KV group indices
            kv_weights: (batch, seq, active_kv_groups) - routing weights
            soft_probs: (batch, seq, num_kv_groups) - full soft probs for aux loss
        """
        if temperature is None:
            temperature = self.router_temperature
        
        # Compute router logits: (batch, seq, num_kv_groups)
        router_logits = self.router(x)
        
        # Add Gumbel noise during training
        if self.training:
            u = torch.rand_like(router_logits)
            gumbel_noise = -torch.log(-torch.log(u + 1e-8) + 1e-8)
            noisy_logits = (router_logits + gumbel_noise) / temperature
        else:
            noisy_logits = router_logits
        
        # Soft probabilities for auxiliary loss
        soft_probs = F.softmax(noisy_logits, dim=-1)
        
        # Top-k selection of KV groups
        top_k_logits, kv_indices = noisy_logits.topk(self.active_kv_groups, dim=-1)
        
        # Renormalize weights over selected KV groups
        kv_weights = F.softmax(top_k_logits, dim=-1)
        
        return kv_indices, kv_weights, soft_probs
    
    def _compute_aux_loss(
        self,
        soft_probs: torch.Tensor,
        kv_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Compute auxiliary loss for load balancing."""
        batch, seq_len, M = soft_probs.shape
        
        # Load balance loss
        primary_kv = kv_indices[..., 0]
        one_hot = F.one_hot(primary_kv, M).float()
        f = one_hot.mean(dim=1)
        p = soft_probs.mean(dim=1)
        balance_loss = M * (f * p).sum(dim=-1).mean()
        
        # Entropy regularization
        entropy = -(soft_probs * torch.log(soft_probs + 1e-8)).sum(dim=-1)
        entropy_loss = -entropy.mean()
        
        aux_loss = (
            self.load_balance_weight * balance_loss +
            self.entropy_weight * entropy_loss
        )
        
        return aux_loss
    
    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        position_offset: int = 0,
        kv_counts: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """
        Forward pass with GQA-style routing.
        
        Each token selects active_kv_groups KV groups.
        All Q heads within selected KV groups are activated.
        """
        batch, seq_len, _ = x.shape
        device = x.device
        dtype = x.dtype
        M = self.num_kv_groups
        A = self.active_kv_groups
        Q = self.q_heads_per_kv
        
        assert batch == 1, "GQA MoH Attention currently requires batch_size=1"
        
        # Initialize KV group counts
        if kv_counts is None:
            kv_counts = torch.zeros(batch, M, device=device, dtype=torch.long)
        
        # === ROUTING: Select top-k KV groups per token ===
        kv_indices, kv_weights, soft_probs = self._compute_routing(x)
        # kv_indices: (batch, seq, active_kv_groups)
        # kv_weights: (batch, seq, active_kv_groups)
        
        aux_loss = self._compute_aux_loss(soft_probs, kv_indices)
        
        # === COMPUTE Q, K, V ===
        # Q: (batch, seq, total_q_heads, head_dim) = (batch, seq, M*Q, head_dim)
        q_all = self.q_proj(x).view(batch, seq_len, M, Q, self.head_dim)
        
        # K, V: (batch, seq, M, head_dim)
        k_all = self.k_proj(x).view(batch, seq_len, M, self.head_dim)
        v_all = self.v_proj(x).view(batch, seq_len, M, self.head_dim)
        
        # === GATHER SELECTED KV GROUPS ===
        # kv_indices: (batch, seq, A) -> expand for gathering
        kv_idx_q = kv_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, Q, self.head_dim)
        kv_idx_kv = kv_indices.unsqueeze(-1).expand(-1, -1, -1, self.head_dim)
        
        # Gather Q for selected KV groups: (batch, seq, A, Q, head_dim)
        q = torch.gather(q_all, 2, kv_idx_q)
        # Flatten to (batch, seq, A*Q, head_dim)
        q = q.view(batch, seq_len, A * Q, self.head_dim)
        
        # Gather K, V for selected KV groups: (batch, seq, A, head_dim)
        k = torch.gather(k_all, 2, kv_idx_kv)
        v = torch.gather(v_all, 2, kv_idx_kv)
        
        # Expand K, V to match Q heads: (batch, seq, A, head_dim) -> (batch, seq, A*Q, head_dim)
        k = k.unsqueeze(3).expand(-1, -1, -1, Q, -1).reshape(batch, seq_len, A * Q, self.head_dim)
        v = v.unsqueeze(3).expand(-1, -1, -1, Q, -1).reshape(batch, seq_len, A * Q, self.head_dim)
        
        # === BUILD BLOCKS FOR ATTENTION ===
        # Group tokens by their selected KV groups
        flat_kv_groups = kv_indices[0].flatten()  # (seq * A,)
        flat_tokens = torch.arange(seq_len, device=device).unsqueeze(1).expand(-1, A).flatten()
        flat_a_idx = torch.arange(A, device=device).unsqueeze(0).expand(seq_len, -1).flatten()
        
        # Sort by KV group
        sort_order = flat_kv_groups.argsort()
        sorted_kv_groups = flat_kv_groups[sort_order]
        sorted_tokens = flat_tokens[sort_order]
        sorted_a_idx = flat_a_idx[sort_order]
        
        # Count tokens per KV group
        kv_token_counts = torch.bincount(sorted_kv_groups, minlength=M)
        seqlens_kv = kv_token_counts[kv_token_counts > 0].tolist()
        
        if len(seqlens_kv) == 0:
            out = torch.zeros(batch, seq_len, self.embed_dim, device=device, dtype=dtype)
            return out, kv_counts, aux_loss
        
        # For Q heads, we have Q times more elements per KV group
        seqlens_q = [s * Q for s in seqlens_kv]
        
        # === GATHER Q, K, V IN BLOCK ORDER ===
        # K, V are per KV group
        k_flat = k.view(batch, seq_len * A * Q, self.head_dim)
        v_flat = v.view(batch, seq_len * A * Q, self.head_dim)
        q_flat = q.view(batch, seq_len * A * Q, self.head_dim)
        
        # Create gather indices for K, V (one per KV group assignment)
        flat_positions_kv = sorted_tokens * A + sorted_a_idx
        
        # For Q, we need to expand to include all Q heads per KV group
        # Each KV group assignment has Q query heads
        q_head_offsets = torch.arange(Q, device=device)
        flat_positions_q = (flat_positions_kv.unsqueeze(1) * Q + q_head_offsets).flatten()
        
        # Gather K, V in KV-group order (then expand for Q heads)
        k_ordered_kv = k_flat[:, flat_positions_kv * Q, :]  # Wrong - need to handle differently
        
        # Actually, let's rethink. For each sorted (token, kv_group) pair:
        # - K, V come from that specific KV group
        # - Q comes from all Q heads within that KV group
        
        # Simpler approach: expand K, V for all Q heads, then gather together with Q
        # K, V indexed by: token * A * Q + a_idx * Q + q_idx
        k_expanded = k.view(batch, seq_len * A * Q, self.head_dim)
        v_expanded = v.view(batch, seq_len * A * Q, self.head_dim)
        
        # For sorted order, we need: for each (token, kv_group), get all Q heads
        # flat_positions_base = sorted_tokens * A + sorted_a_idx  # position in (seq, A) space
        # Expand to (seq * A * Q) space
        flat_positions_expanded = (flat_positions_kv.unsqueeze(1) * Q + q_head_offsets.unsqueeze(0)).flatten()
        
        q_ordered = q_flat[:, flat_positions_expanded, :]
        k_ordered = k_expanded[:, flat_positions_expanded, :]
        v_ordered = v_expanded[:, flat_positions_expanded, :]
        
        # === COMPUTE POSITIONS (per-KV-group RoPE) ===
        total_tokens_kv = sorted_kv_groups.shape[0]
        
        # Mark KV group boundaries
        kv_changes = torch.ones(total_tokens_kv, device=device, dtype=torch.long)
        kv_changes[1:] = (sorted_kv_groups[1:] != sorted_kv_groups[:-1]).long()
        
        # Cumsum within segments
        cumsum = torch.arange(1, total_tokens_kv + 1, device=device, dtype=torch.long)
        boundary_cumsum = (kv_changes * cumsum).cummax(dim=0)[0]
        positions_in_kv = cumsum - boundary_cumsum
        
        # Expand positions for Q heads
        positions_expanded = positions_in_kv.unsqueeze(1).expand(-1, Q).flatten()
        positions_expanded = positions_expanded.clamp(0, self.rope.max_seq_len - 1)
        
        # === APPLY ROPE ===
        cos = self.rope.cos_cached[positions_expanded]
        sin = self.rope.sin_cached[positions_expanded]
        
        q_ordered = (q_ordered * cos.unsqueeze(0)) + (rotate_half(q_ordered) * sin.unsqueeze(0))
        k_ordered = (k_ordered * cos.unsqueeze(0)) + (rotate_half(k_ordered) * sin.unsqueeze(0))
        
        # === XFORMERS BLOCK-DIAGONAL CAUSAL ATTENTION ===
        # Each block = one KV group's tokens × Q heads
        q_xf = q_ordered.unsqueeze(2)  # (batch, total_q, 1, head_dim)
        k_xf = k_ordered.unsqueeze(2)
        v_xf = v_ordered.unsqueeze(2)
        
        attn_bias = fmha.attn_bias.BlockDiagonalCausalMask.from_seqlens(seqlens_q)
        attn_bias = attn_bias.to(device)
        
        out_ordered = memory_efficient_attention(
            q_xf, k_xf, v_xf,
            attn_bias=attn_bias,
            scale=self.scale,
        ).squeeze(2)
        
        # === SCATTER BACK ===
        out_flat = torch.zeros(batch, seq_len * A * Q, self.head_dim, device=device, dtype=dtype)
        out_flat[:, flat_positions_expanded, :] = out_ordered
        
        # Reshape to (batch, seq, A, Q, head_dim)
        out = out_flat.view(batch, seq_len, A, Q, self.head_dim)
        
        # Weight by KV group routing weights: (batch, seq, A, 1, 1)
        weights = kv_weights.unsqueeze(-1).unsqueeze(-1)
        out = out * weights  # (batch, seq, A, Q, head_dim)
        
        # Flatten to (batch, seq, A*Q*head_dim) for output projection
        out = out.view(batch, seq_len, A * Q * self.head_dim)
        
        # === UPDATE KV COUNTS ===
        tokens_per_kv = torch.bincount(flat_kv_groups, minlength=M).unsqueeze(0)
        updated_kv_counts = kv_counts + tokens_per_kv
        
        # === OUTPUT PROJECTION ===
        out = self.out_proj(out)
        
        return out, updated_kv_counts, aux_loss


class MoHSparseAttentionSimple(nn.Module):
    """
    Simplified Mixture of Heads - closer to standard MHA implementation.
    
    Key simplification: Instead of complex block-diagonal attention,
    this version computes attention for each selected head independently
    and combines results. Simpler but potentially less efficient.
    
    Use MixtureOfHeadsAttention for the optimized version.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 32,
        active_heads: int = 8,
        head_dim: Optional[int] = None,
        dropout: float = 0.0,
        bias: bool = False,
        max_seq_len: int = 8192,
        rope_base: float = 10000.0,
        router_temperature: float = 1.0,
        entropy_weight: float = 0.01,
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.active_heads = active_heads
        
        if head_dim is not None:
            self.head_dim = head_dim
        else:
            self.head_dim = embed_dim // active_heads
        
        self.scale = self.head_dim ** -0.5
        self.router_temperature = router_temperature
        self.entropy_weight = entropy_weight
        
        # Projections
        self.q_proj = nn.Linear(embed_dim, num_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, num_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, num_heads * self.head_dim, bias=bias)
        self.router = nn.Linear(embed_dim, num_heads, bias=bias)
        self.out_proj = nn.Linear(active_heads * self.head_dim, embed_dim, bias=bias)
        
        # RoPE per head
        self.rope = MoHRoPE(self.head_dim, max_seq_len, rope_base)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        position_offset: int = 0,
        head_counts: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """
        Simple forward: compute each head's attention independently.
        
        Less efficient than block-diagonal but easier to understand/debug.
        """
        batch, seq_len, _ = x.shape
        device = x.device
        dtype = x.dtype
        H = self.num_heads
        K = self.active_heads
        
        # Initialize head counts
        if head_counts is None:
            head_counts = torch.zeros(batch, H, device=device, dtype=torch.long)
        
        # === ROUTING ===
        router_logits = self.router(x)  # (batch, seq, H)
        
        if self.training:
            u = torch.rand_like(router_logits)
            gumbel = -torch.log(-torch.log(u + 1e-8) + 1e-8)
            noisy_logits = (router_logits + gumbel) / self.router_temperature
        else:
            noisy_logits = router_logits
        
        soft_probs = F.softmax(noisy_logits, dim=-1)
        top_k_logits, head_indices = noisy_logits.topk(K, dim=-1)
        head_weights = F.softmax(top_k_logits, dim=-1)  # (batch, seq, K)
        
        # === AUX LOSS ===
        primary = head_indices[..., 0]
        one_hot = F.one_hot(primary, H).float()
        f = one_hot.mean(dim=1)
        p = soft_probs.mean(dim=1)
        balance_loss = H * (f * p).sum(dim=-1).mean()
        entropy = -(soft_probs * torch.log(soft_probs + 1e-8)).sum(dim=-1).mean()
        aux_loss = balance_loss - self.entropy_weight * entropy
        
        # === Q, K, V ===
        q_all = self.q_proj(x).view(batch, seq_len, H, self.head_dim)
        k_all = self.k_proj(x).view(batch, seq_len, H, self.head_dim)
        v_all = self.v_proj(x).view(batch, seq_len, H, self.head_dim)
        
        # Gather selected heads
        idx = head_indices.unsqueeze(-1).expand(-1, -1, -1, self.head_dim)
        q = torch.gather(q_all, 2, idx)  # (batch, seq, K, head_dim)
        k = torch.gather(k_all, 2, idx)
        v = torch.gather(v_all, 2, idx)
        
        # Transpose for attention: (batch, K, seq, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # === APPLY ROPE ===
        # Simple version: use standard positions (not per-head tracking)
        positions = torch.arange(seq_len, device=device) + position_offset
        positions = positions.clamp(0, self.rope.max_seq_len - 1)
        cos = self.rope.cos_cached[positions]
        sin = self.rope.sin_cached[positions]
        
        q = (q * cos) + (rotate_half(q) * sin)
        k = (k * cos) + (rotate_half(k) * sin)
        
        # === STANDARD CAUSAL ATTENTION ===
        # (batch, K, seq, seq)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Causal mask
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=1
        )
        attn = attn.masked_fill(causal_mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention: (batch, K, seq, head_dim)
        out = torch.matmul(attn, v)
        
        # Weight by routing weights: (batch, K, seq, head_dim)
        # head_weights: (batch, seq, K) -> (batch, K, seq, 1)
        weights = head_weights.transpose(1, 2).unsqueeze(-1)  # (batch, K, seq, 1)
        out = out * weights  # (batch, K, seq, head_dim) * (batch, K, seq, 1)
        
        # Transpose back: (batch, seq, K, head_dim)
        out = out.transpose(1, 2)
        
        # Reshape and project
        out = out.reshape(batch, seq_len, K * self.head_dim)
        out = self.out_proj(out)
        
        return out, head_counts, aux_loss

