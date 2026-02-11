"""
HyperGraph Fixed-Capacity Sparse Attention

Key Design:
- Each timeline has FIXED capacity of N/K tokens
- Token-choice routing with probability-based overflow handling
- When timeline k has >N/K tokens wanting it, keep top N/K by softmax probability
- Enables standard attention kernels (FlashAttention compatible)
- Supports batch_size > 1 (fixed tensor shapes)

Advantages over original HyperGraph:
1. Batching: Fixed shapes enable batch_size > 1
2. Efficiency: Standard FlashAttention kernels work directly
3. Load balance: Capacity limits + soft probability selection
4. Timeline-local RoPE: Preserved for length generalization

Complexity: O(K × (N/K)²) = O(N²/K) per head, same as original
But with much better hardware utilization!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math

try:
    from flash_attn import flash_attn_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False

try:
    from xformers.ops import memory_efficient_attention
    XFORMERS_AVAILABLE = True
except ImportError:
    XFORMERS_AVAILABLE = False


class FixedCapacityRoPE(nn.Module):
    """RoPE with timeline-local positions."""
    
    def __init__(self, head_dim: int, max_seq_len: int = 8192, base: float = 10000.0):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)
    
    def _build_cache(self, max_seq_len: int):
        t = torch.arange(max_seq_len, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
    
    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Apply RoPE to x using given positions."""
        # x: (..., seq, head_dim), positions: (..., seq)
        cos = self.cos_cached[positions]  # (..., seq, head_dim)
        sin = self.sin_cached[positions]
        
        # Rotate
        x1, x2 = x[..., :self.head_dim//2], x[..., self.head_dim//2:]
        rotated = torch.cat([-x2, x1], dim=-1)
        
        return x * cos + rotated * sin


class HyperGraphFixedCapacity(nn.Module):
    """
    HyperGraph Sparse Attention with Fixed Timeline Capacity.
    
    Each timeline gets exactly capacity = ceil(N/K) tokens.
    Overflow is handled by keeping tokens with highest routing probability.
    
    Args:
        embed_dim: Model dimension
        num_heads: Number of attention heads
        num_timelines: K - number of timelines
        head_dim: Dimension per head
        capacity_factor: Multiplier for N/K capacity (1.0 = exact, 1.25 = 25% overflow)
        use_flash_attn: Use FlashAttention if available
        dropout: Attention dropout
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_timelines: int = 4,
        head_dim: Optional[int] = None,
        num_kv_heads: Optional[int] = None,
        max_seq_len: int = 8192,
        capacity_factor: float = 1.0,
        use_flash_attn: bool = True,
        dropout: float = 0.0,
        aux_loss_weight: float = 0.01,
        sliding_window: Optional[int] = None,  # Optional sliding window size within timelines
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_timelines = num_timelines
        self.head_dim = head_dim or embed_dim // num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.num_kv_groups = num_heads // self.num_kv_heads
        self.capacity_factor = capacity_factor
        self.aux_loss_weight = aux_loss_weight
        self.sliding_window = sliding_window  # None = full causal, int = sliding window size
        
        self.scale = self.head_dim ** -0.5
        self.use_flash_attn = use_flash_attn and FLASH_ATTN_AVAILABLE
        
        # Projections
        self.q_proj = nn.Linear(embed_dim, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(num_heads * self.head_dim, embed_dim, bias=False)
        
        # Router: token -> timeline probability
        self.router = nn.Linear(embed_dim, num_timelines, bias=False)
        
        # RoPE for timeline-local positions
        self.rope = FixedCapacityRoPE(self.head_dim, max_seq_len)
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.xavier_uniform_(self.router.weight)
    
    def _compute_routing(
        self,
        x: torch.Tensor,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute token-to-timeline routing with capacity limits (VECTORIZED).
        
        Two-stage selection:
        1. Softmax over timelines -> each token picks preferred timeline (argmax)
        2. Per-timeline topk -> when overflow, keep top-capacity by probability
        
        Returns:
            timeline_indices: (B, K, capacity) - which tokens go to each timeline
            timeline_mask: (B, K, capacity) - valid token mask
            routing_weights: (B, K, capacity) - soft weights for gradient
            aux_loss: Load balance auxiliary loss
        """
        B, N, D = x.shape
        K = self.num_timelines
        device = x.device
        dtype = x.dtype
        
        # === STAGE 1: Softmax routing - token picks timeline ===
        logits = self.router(x)  # (B, N, K)
        probs = F.softmax(logits / temperature, dim=-1)  # (B, N, K)
        
        # Each token's preferred timeline
        timeline_choice = logits.argmax(dim=-1)  # (B, N)
        
        # Capacity per timeline
        capacity = int(math.ceil(N / K * self.capacity_factor))
        capacity = min(capacity, N)
        
        # === STAGE 2: Vectorized per-timeline topk selection ===
        # Create score matrix: high score if token chose this timeline, -inf otherwise
        # Shape: (B, K, N)
        choice_onehot = F.one_hot(timeline_choice, K).float()  # (B, N, K)
        choice_onehot = choice_onehot.transpose(1, 2)  # (B, K, N)
        
        # Score = probability for chosen timeline, -inf for others
        probs_t = probs.transpose(1, 2)  # (B, K, N) - prob of each token for each timeline
        scores = torch.where(
            choice_onehot > 0.5,
            probs_t,  # Use probability as score
            torch.tensor(-1e9, device=device, dtype=dtype)  # Mask non-chosen
        )
        
        # Topk per timeline (vectorized!)
        topk_scores, topk_indices = torch.topk(scores, capacity, dim=-1)  # (B, K, capacity)
        
        # Create mask: valid if score > -1e8 (not masked out)
        timeline_mask = topk_scores > -1e8  # (B, K, capacity)
        
        # Routing weights = softmax scores (clamped for numerical stability)
        routing_weights = topk_scores.clamp(min=0.0)  # (B, K, capacity)
        
        # === AUXILIARY LOSS: Load balance ===
        # Switch Transformer style: K * sum(f_k * p_k)
        f = choice_onehot.mean(dim=-1)  # (B, K) fraction per timeline
        p = probs.mean(dim=1)  # (B, K) avg prob per timeline
        balance_loss = K * (f * p).sum(dim=-1).mean()
        
        aux_loss = balance_loss.clamp(min=0.0, max=10.0)
        
        return topk_indices, timeline_mask, routing_weights, aux_loss
    
    def _timeline_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        timeline_indices: torch.Tensor,
        timeline_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute attention within each timeline using efficient kernels.
        
        Args:
            q, k, v: (B, H, N, d)
            timeline_indices: (B, K, capacity)
            timeline_mask: (B, K, capacity)
        
        Returns:
            out: (B, H, N, d) - attention outputs scattered back
        """
        B, H, N, d = q.shape
        K = self.num_timelines
        capacity = timeline_indices.shape[-1]
        device = q.device
        dtype = q.dtype
        
        # Output tensor
        out = torch.zeros(B, H, N, d, device=device, dtype=dtype)
        out_counts = torch.zeros(B, 1, N, 1, device=device, dtype=dtype)
        
        for k_idx in range(K):
            # Get indices for this timeline
            indices = timeline_indices[:, k_idx]  # (B, capacity)
            mask = timeline_mask[:, k_idx]  # (B, capacity)
            
            # Skip if no valid tokens
            if not mask.any():
                continue
            
            # Gather Q, K, V for this timeline
            # indices: (B, capacity) -> (B, 1, capacity, 1) for gather
            idx_expanded = indices.unsqueeze(1).unsqueeze(-1).expand(B, H, capacity, d)
            
            q_timeline = torch.gather(q, 2, idx_expanded)  # (B, H, capacity, d)
            k_timeline = torch.gather(k, 2, idx_expanded)
            v_timeline = torch.gather(v, 2, idx_expanded)
            
            # Timeline-local positions: 0, 1, 2, ... for each timeline
            positions = torch.arange(capacity, device=device).unsqueeze(0).expand(B, -1)
            
            # Apply RoPE with local positions
            q_timeline = self.rope(q_timeline, positions.unsqueeze(1).expand(-1, H, -1))
            k_timeline = self.rope(k_timeline, positions.unsqueeze(1).expand(-1, H, -1))
            
            # Causal attention within timeline
            if self.use_flash_attn:
                # FlashAttention expects (B, N, H, d)
                q_fa = q_timeline.transpose(1, 2)
                k_fa = k_timeline.transpose(1, 2)
                v_fa = v_timeline.transpose(1, 2)
                
                # Zero out invalid positions to prevent garbage attention
                mask_expanded = mask.unsqueeze(1).unsqueeze(-1).expand(-1, H, -1, d)  # (B, H, cap, d)
                q_fa = q_fa.transpose(1, 2)  # (B, H, cap, d)
                k_fa = k_fa.transpose(1, 2)
                v_fa = v_fa.transpose(1, 2)
                q_fa = q_fa * mask_expanded.float()
                k_fa = k_fa * mask_expanded.float()
                v_fa = v_fa * mask_expanded.float()
                q_fa = q_fa.transpose(1, 2)  # Back to (B, cap, H, d)
                k_fa = k_fa.transpose(1, 2)
                v_fa = v_fa.transpose(1, 2)
                
                out_timeline = flash_attn_func(
                    q_fa, k_fa, v_fa,
                    causal=True,
                    softmax_scale=self.scale,
                )
                out_timeline = out_timeline.transpose(1, 2)  # Back to (B, H, cap, d)
                out_timeline = out_timeline * mask_expanded.float()  # Zero invalid outputs
            else:
                # Standard attention
                attn = torch.matmul(q_timeline, k_timeline.transpose(-2, -1)) * self.scale
                
                # Causal mask (upper triangular)
                causal_mask = torch.triu(
                    torch.ones(capacity, capacity, device=device, dtype=torch.bool),
                    diagonal=1
                )
                
                # Optional sliding window mask within timeline
                if self.sliding_window is not None:
                    # Mask positions outside sliding window
                    row_idx = torch.arange(capacity, device=device).unsqueeze(1)
                    col_idx = torch.arange(capacity, device=device).unsqueeze(0)
                    window_mask = (row_idx - col_idx) >= self.sliding_window
                    causal_mask = causal_mask | window_mask
                
                attn = attn.masked_fill(causal_mask, float('-inf'))
                
                # Mask invalid KEY positions (columns)
                key_mask = mask.unsqueeze(1).unsqueeze(2).expand(B, H, capacity, capacity)
                attn = attn.masked_fill(~key_mask, float('-inf'))
                
                # Safe softmax: handle all-masked rows
                attn = F.softmax(attn, dim=-1)
                attn = torch.nan_to_num(attn, nan=0.0)  # Replace NaN with 0
                attn = self.dropout(attn)
                out_timeline = torch.matmul(attn, v_timeline)
            
            # Scatter back to original positions
            out.scatter_add_(2, idx_expanded, out_timeline)
            
            # Track counts for averaging (tokens can appear in multiple timelines with top-k)
            count_idx = indices.unsqueeze(1).unsqueeze(-1).expand(B, 1, capacity, 1)
            out_counts.scatter_add_(2, count_idx, torch.ones_like(count_idx, dtype=dtype))
        
        # Average where multiple writes occurred
        out_counts = out_counts.clamp(min=1.0)
        out = out / out_counts
        
        return out
    
    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with fixed-capacity timeline routing.
        
        Args:
            x: (B, N, D) input tensor
            attn_mask: Unused (for API compatibility)
        
        Returns:
            out: (B, N, D) output tensor
            aux_loss: Auxiliary loss for load balance
        """
        B, N, D = x.shape
        H = self.num_heads
        
        # Compute Q, K, V
        q = self.q_proj(x).view(B, N, H, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # GQA expansion
        if self.num_kv_groups > 1:
            k = k.unsqueeze(2).expand(-1, -1, self.num_kv_groups, -1, -1)
            k = k.reshape(B, H, N, self.head_dim)
            v = v.unsqueeze(2).expand(-1, -1, self.num_kv_groups, -1, -1)
            v = v.reshape(B, H, N, self.head_dim)
        
        # Compute routing
        timeline_indices, timeline_mask, routing_weights, aux_loss = self._compute_routing(x)
        
        # Compute attention per timeline
        out = self._timeline_attention(q, k, v, timeline_indices, timeline_mask)
        
        # Project output
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        out = self.out_proj(out)
        
        return out, aux_loss * self.aux_loss_weight


# === BATCHED VERSION (Vectorized, no Python loops) ===

class HyperGraphFixedCapacityBatched(HyperGraphFixedCapacity):
    """
    Fully batched version - no Python loops over timelines.
    
    Key insight: Reshape to (B*K, capacity, d) and use standard batched attention.
    """
    
    def _timeline_attention_batched(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        timeline_indices: torch.Tensor,
        timeline_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Vectorized attention across all timelines simultaneously.
        
        Reshapes (B, H, N, d) -> (B*K, H, capacity, d) for batched attention.
        """
        B, H, N, d = q.shape
        K = self.num_timelines
        capacity = timeline_indices.shape[-1]
        device = q.device
        dtype = q.dtype
        
        # Expand indices for all heads: (B, K, cap) -> (B, K, H, cap, d)
        idx = timeline_indices.unsqueeze(2).unsqueeze(-1).expand(B, K, H, capacity, d)
        
        # Gather for all timelines at once
        # q: (B, H, N, d) -> (B, 1, H, N, d) -> gather -> (B, K, H, cap, d)
        q_exp = q.unsqueeze(1).expand(B, K, H, N, d)
        k_exp = k.unsqueeze(1).expand(B, K, H, N, d)
        v_exp = v.unsqueeze(1).expand(B, K, H, N, d)
        
        q_timelines = torch.gather(q_exp, 3, idx)  # (B, K, H, cap, d)
        k_timelines = torch.gather(k_exp, 3, idx)
        v_timelines = torch.gather(v_exp, 3, idx)
        
        # Reshape for batched attention: (B*K, H, cap, d)
        q_batch = q_timelines.reshape(B * K, H, capacity, d)
        k_batch = k_timelines.reshape(B * K, H, capacity, d)
        v_batch = v_timelines.reshape(B * K, H, capacity, d)
        
        # Timeline-local positions
        positions = torch.arange(capacity, device=device)
        positions = positions.unsqueeze(0).expand(B * K, -1)  # (B*K, cap)
        
        # Apply RoPE
        q_batch = self.rope(q_batch, positions.unsqueeze(1).expand(-1, H, -1))
        k_batch = self.rope(k_batch, positions.unsqueeze(1).expand(-1, H, -1))
        
        # Batched causal attention (with optional sliding window)
        # timeline_mask: (B, K, cap) -> (B*K, cap)
        batch_mask = timeline_mask.reshape(B * K, capacity)
        
        if self.use_flash_attn:
            q_fa = q_batch.transpose(1, 2)  # (B*K, cap, H, d)
            k_fa = k_batch.transpose(1, 2)
            v_fa = v_batch.transpose(1, 2)
            
            # Zero out invalid positions to prevent garbage attention
            mask_exp = batch_mask.unsqueeze(1).unsqueeze(-1).expand(-1, H, -1, d)
            q_fa = q_fa.transpose(1, 2) * mask_exp.float()  # (B*K, H, cap, d)
            k_fa = k_fa.transpose(1, 2) * mask_exp.float()
            v_fa = v_fa.transpose(1, 2) * mask_exp.float()
            q_fa = q_fa.transpose(1, 2)  # Back to (B*K, cap, H, d)
            k_fa = k_fa.transpose(1, 2)
            v_fa = v_fa.transpose(1, 2)
            
            # FlashAttention with sliding window if specified
            if self.sliding_window is not None:
                out_batch = flash_attn_func(
                    q_fa, k_fa, v_fa, 
                    causal=True, 
                    softmax_scale=self.scale,
                    window_size=(self.sliding_window, 0)  # (left, right) window
                )
            else:
                out_batch = flash_attn_func(q_fa, k_fa, v_fa, causal=True, softmax_scale=self.scale)
            out_batch = out_batch.transpose(1, 2)  # (B*K, H, cap, d)
            out_batch = out_batch * mask_exp.float()  # Zero invalid outputs
        else:
            attn = torch.matmul(q_batch, k_batch.transpose(-2, -1)) * self.scale
            
            # Causal mask
            causal_mask = torch.triu(torch.ones(capacity, capacity, device=device, dtype=torch.bool), diagonal=1)
            
            # Optional sliding window
            if self.sliding_window is not None:
                row_idx = torch.arange(capacity, device=device).unsqueeze(1)
                col_idx = torch.arange(capacity, device=device).unsqueeze(0)
                window_mask = (row_idx - col_idx) >= self.sliding_window
                causal_mask = causal_mask | window_mask
            
            attn = attn.masked_fill(causal_mask, float('-inf'))
            attn = F.softmax(attn, dim=-1)
            attn = torch.nan_to_num(attn, nan=0.0)
            out_batch = torch.matmul(attn, v_batch)
        
        # Reshape back: (B*K, H, cap, d) -> (B, K, H, cap, d)
        out_timelines = out_batch.reshape(B, K, H, capacity, d)
        
        # Scatter back to original positions
        out = torch.zeros(B, H, N, d, device=device, dtype=dtype)
        out_counts = torch.zeros(B, 1, N, 1, device=device, dtype=dtype)
        
        # Scatter for each timeline (vectorized over B, H)
        for k_idx in range(K):
            idx_k = timeline_indices[:, k_idx].unsqueeze(1).unsqueeze(-1).expand(B, H, capacity, d)
            out.scatter_add_(2, idx_k, out_timelines[:, k_idx])
            
            count_idx = timeline_indices[:, k_idx].unsqueeze(1).unsqueeze(-1).expand(B, 1, capacity, 1)
            out_counts.scatter_add_(2, count_idx, torch.ones_like(count_idx, dtype=dtype))
        
        out_counts = out_counts.clamp(min=1.0)
        return out / out_counts


def create_fixed_capacity_hypergraph(
    embed_dim: int = 512,
    num_heads: int = 8,
    num_timelines: int = 4,
    use_batched: bool = True,
    **kwargs
) -> nn.Module:
    """
    Create a fixed-capacity HyperGraph attention module.
    
    This version:
    - Supports batch_size > 1
    - Uses standard attention kernels (FlashAttention compatible)
    - Maintains timeline-local RoPE for length generalization
    - Has probability-based overflow handling
    """
    if use_batched:
        return HyperGraphFixedCapacityBatched(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_timelines=num_timelines,
            **kwargs
        )
    else:
        return HyperGraphFixedCapacity(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_timelines=num_timelines,
            **kwargs
        )

