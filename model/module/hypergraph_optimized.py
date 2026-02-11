"""
HyperGraph Optimized Sparse Attention - Expert-Choice Routing

Key optimizations over original HyperGraph:
1. Expert-Choice Routing: Each timeline picks top-N/K tokens (load balanced by design)
2. Inference Optimization: Only compute the timeline containing the query token
3. Training Shortcut: Option to only backprop through relevant timeline

This makes HyperGraph similar to MoSA but with:
- Timeline-local RoPE (better length generalization)
- Unified timeline routing (vs per-head expert selection)
- Optional cross-timeline bridges via top-k

Complexity:
    Standard:    O(H × N²)
    HyperGraph:  O(H × N²/K)  [All timelines computed]
    Optimized:   O(H × N²/K²) [Inference: only query's timeline]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from functools import lru_cache

from .rope import rotate_half

try:
    from xformers.ops import memory_efficient_attention, fmha
    XFORMERS_AVAILABLE = True
except ImportError:
    XFORMERS_AVAILABLE = False


class HyperGraphOptimizedAttention(nn.Module):
    """
    Optimized HyperGraph Sparse Attention with Expert-Choice Routing.
    
    Key differences from original:
    1. Expert-choice: timeline k picks top N/K tokens (vs token picks timeline)
    2. Inference-only mode: compute only last token's timeline
    3. Gumbel selection: optionally select across timelines with Gumbel-softmax
    
    Args:
        embed_dim: Model embedding dimension
        num_heads: Number of attention heads
        num_hyper_nodes: Number of timelines (K)
        head_dim: Dimension per head (default: embed_dim // num_heads)
        num_kv_heads: Number of KV heads for GQA (default: num_heads)
        use_local_rope: Use timeline-local positions (default: True)
        inference_optimize: Only compute query's timeline during inference
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_hyper_nodes: int = 4,
        head_dim: Optional[int] = None,
        num_kv_heads: Optional[int] = None,
        max_seq_len: int = 8192,
        use_local_rope: bool = True,
        inference_optimize: bool = True,
        use_gumbel_selection: bool = False,
        gumbel_temperature: float = 1.0,
        capacity_factor: float = 1.25,  # Allow 25% overflow per timeline
        aux_loss_weight: float = 0.01,
    ):
        super().__init__()
        
        assert XFORMERS_AVAILABLE, "xformers required for optimized HyperGraph"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_hyper_nodes = num_hyper_nodes
        self.head_dim = head_dim or embed_dim // num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.num_kv_groups = num_heads // self.num_kv_heads
        
        self.use_local_rope = use_local_rope
        self.inference_optimize = inference_optimize
        self.use_gumbel_selection = use_gumbel_selection
        self.gumbel_temperature = gumbel_temperature
        self.capacity_factor = capacity_factor
        self.aux_loss_weight = aux_loss_weight
        
        self.scale = self.head_dim ** -0.5
        
        # Projections
        self.q_proj = nn.Linear(embed_dim, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(num_heads * self.head_dim, embed_dim, bias=False)
        
        # Expert-choice router: computes affinity scores for each timeline
        # Unlike original HyperGraph, this is used by timelines to select tokens
        self.router = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, num_hyper_nodes),
        )
        
        # Timeline output gates (for Gumbel selection across timelines)
        if use_gumbel_selection:
            self.timeline_gate = nn.Linear(self.head_dim, 1, bias=True)
        
        # RoPE
        self.rope = HyperGraphRoPE(self.head_dim, max_seq_len)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        
        # Initialize router for uniform selection
        for m in self.router.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def _expert_choice_routing(
        self,
        x: torch.Tensor,
        seq_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Expert-choice routing: each timeline selects top N/K tokens.
        
        Unlike token-choice (original HyperGraph), this is load-balanced by design.
        Each timeline gets exactly capacity tokens (with slight overflow allowed).
        
        Args:
            x: Input tensor (batch, seq, dim)
            seq_len: Sequence length
        
        Returns:
            selected_indices: (K, capacity) - indices of selected tokens per timeline
            selection_weights: (K, capacity) - soft weights for selected tokens
            aux_loss: Load balance auxiliary loss
        """
        batch = x.shape[0]
        device = x.device
        K = self.num_hyper_nodes
        
        # Compute affinity scores: how much each timeline "wants" each token
        # Shape: (batch, seq, K)
        scores = self.router(x)  # (B, N, K)
        
        # Capacity per timeline (with overflow factor)
        capacity = int(seq_len / K * self.capacity_factor)
        capacity = min(capacity, seq_len)  # Can't exceed total tokens
        
        # Each timeline selects its top-capacity tokens
        # Transpose to (B, K, N) for timeline-wise selection
        scores_t = scores.transpose(1, 2)  # (B, K, N)
        
        # Top-k selection per timeline
        topk_scores, topk_indices = torch.topk(scores_t, capacity, dim=-1)  # (B, K, capacity)
        
        # Softmax weights for selected tokens (for gradient flow)
        selection_weights = F.softmax(topk_scores, dim=-1)  # (B, K, capacity)
        
        # Auxiliary loss: encourage each token to be selected by at least one timeline
        # This prevents "orphan" tokens that no timeline wants
        token_max_score = scores.max(dim=-1)[0]  # (B, N) - best score per token
        aux_loss = -token_max_score.mean()  # Maximize token coverage
        
        return topk_indices[0], selection_weights[0], aux_loss  # Remove batch dim (B=1)
    
    def _compute_timeline_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        timeline_indices: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute attention within a single timeline.
        
        Args:
            q, k, v: Full Q/K/V tensors (batch, heads, seq, head_dim)
            timeline_indices: Indices of tokens in this timeline (capacity,)
            positions: Position indices for RoPE (capacity,)
        
        Returns:
            out: Attention output for this timeline (batch, heads, capacity, head_dim)
        """
        batch, H, N, d = q.shape
        device = q.device
        
        # Gather tokens for this timeline
        indices_expanded = timeline_indices.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
        indices_expanded = indices_expanded.expand(batch, H, -1, d)
        
        q_timeline = torch.gather(q, 2, indices_expanded)  # (B, H, capacity, d)
        k_timeline = torch.gather(k, 2, indices_expanded)
        v_timeline = torch.gather(v, 2, indices_expanded)
        
        # Apply RoPE with timeline-local positions
        cos = self.rope.cos_cached[positions].unsqueeze(0).unsqueeze(0)  # (1, 1, cap, d)
        sin = self.rope.sin_cached[positions].unsqueeze(0).unsqueeze(0)
        
        q_rot = rotate_half(q_timeline)
        k_rot = rotate_half(k_timeline)
        
        q_timeline = q_timeline * cos + q_rot * sin
        k_timeline = k_timeline * cos + k_rot * sin
        
        # Causal attention within timeline
        # Use xformers for efficiency
        q_xf = q_timeline.transpose(1, 2).reshape(batch, -1, 1, d)  # (B, cap, 1, d)
        k_xf = k_timeline.transpose(1, 2).reshape(batch, -1, 1, d)
        v_xf = v_timeline.transpose(1, 2).reshape(batch, -1, 1, d)
        
        capacity = timeline_indices.shape[0]
        attn_bias = fmha.attn_bias.LowerTriangularMask()
        
        out = memory_efficient_attention(q_xf, k_xf, v_xf, attn_bias=attn_bias, scale=self.scale)
        out = out.squeeze(2).reshape(batch, capacity, H, d).transpose(1, 2)  # (B, H, cap, d)
        
        return out
    
    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        position_offset: int = 0,
        node_counts: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """
        Forward pass with expert-choice routing.
        
        Training: Compute all timelines, combine outputs
        Inference (optimized): Only compute timeline containing last token
        
        Args:
            x: Input (batch, seq, embed_dim)
            attn_mask: Unused (API compatibility)
            position_offset: KV cache position offset
            node_counts: Previous node counts (for KV cache)
        
        Returns:
            output: (batch, seq, embed_dim)
            updated_node_counts: For KV cache tracking
            aux_loss: Auxiliary loss for load balance
        """
        batch, seq_len, _ = x.shape
        device = x.device
        dtype = x.dtype
        K = self.num_hyper_nodes
        H = self.num_heads
        
        assert batch == 1, "Optimized HyperGraph requires batch_size=1"
        
        # === EXPERT-CHOICE ROUTING ===
        selected_indices, selection_weights, aux_loss = self._expert_choice_routing(x, seq_len)
        # selected_indices: (K, capacity) - which tokens each timeline selected
        # selection_weights: (K, capacity) - soft weights
        
        # === COMPUTE Q, K, V ===
        q = self.q_proj(x).view(batch, seq_len, H, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # GQA expansion
        if self.num_kv_groups > 1:
            k = k.unsqueeze(2).expand(-1, -1, self.num_kv_groups, -1, -1)
            k = k.reshape(batch, H, seq_len, self.head_dim)
            v = v.unsqueeze(2).expand(-1, -1, self.num_kv_groups, -1, -1)
            v = v.reshape(batch, H, seq_len, self.head_dim)
        
        # === INFERENCE OPTIMIZATION ===
        # Only compute the timeline that contains the last (query) token
        if not self.training and self.inference_optimize:
            # Find which timeline selected the last token
            last_token_idx = seq_len - 1
            
            # Check which timeline has the last token
            timeline_has_last = (selected_indices == last_token_idx).any(dim=1)  # (K,)
            
            if timeline_has_last.any():
                # Use the timeline that has the last token
                active_timeline = timeline_has_last.nonzero()[0].item()
            else:
                # Fallback: use timeline with highest score for last token
                last_token_scores = self.router(x[:, -1:, :])  # (1, 1, K)
                active_timeline = last_token_scores.argmax(dim=-1).item()
            
            # Compute attention only for this timeline
            timeline_indices = selected_indices[active_timeline]  # (capacity,)
            
            # Timeline-local positions
            if self.use_local_rope:
                positions = torch.arange(timeline_indices.shape[0], device=device)
            else:
                positions = timeline_indices  # Global positions
            
            # Compute attention for single timeline
            out_timeline = self._compute_timeline_attention(
                q, k, v, timeline_indices, positions
            )
            
            # Scatter back to full sequence
            out = torch.zeros(batch, H, seq_len, self.head_dim, device=device, dtype=dtype)
            indices_expanded = timeline_indices.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
            indices_expanded = indices_expanded.expand(batch, H, -1, self.head_dim)
            out.scatter_(2, indices_expanded, out_timeline)
            
        else:
            # === TRAINING: COMPUTE ALL TIMELINES ===
            out = torch.zeros(batch, H, seq_len, self.head_dim, device=device, dtype=dtype)
            out_counts = torch.zeros(batch, H, seq_len, 1, device=device, dtype=dtype)
            
            for k_idx in range(K):
                timeline_indices = selected_indices[k_idx]  # (capacity,)
                timeline_weights = selection_weights[k_idx]  # (capacity,)
                
                # Timeline-local positions
                if self.use_local_rope:
                    positions = torch.arange(timeline_indices.shape[0], device=device)
                else:
                    positions = timeline_indices
                
                # Compute attention for this timeline
                out_timeline = self._compute_timeline_attention(
                    q, k, v, timeline_indices, positions
                )  # (B, H, capacity, d)
                
                # Weight by selection probability (for gradient flow)
                out_timeline = out_timeline * timeline_weights.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
                
                # Scatter-add back to full sequence
                indices_expanded = timeline_indices.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
                indices_expanded = indices_expanded.expand(batch, H, -1, self.head_dim)
                
                out.scatter_add_(2, indices_expanded, out_timeline)
                
                # Track which positions have been written
                count_indices = timeline_indices.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
                count_indices = count_indices.expand(batch, H, -1, 1)
                ones = torch.ones_like(out_timeline[..., :1])
                out_counts.scatter_add_(2, count_indices, ones)
            
            # Average where multiple timelines wrote (tokens selected by multiple)
            out_counts = out_counts.clamp(min=1.0)
            out = out / out_counts
        
        # === GUMBEL SELECTION (optional) ===
        # Select across timelines using learned gate
        if self.use_gumbel_selection and self.training:
            gate_logits = self.timeline_gate(out).squeeze(-1)  # (B, H, N)
            gate = F.gumbel_softmax(gate_logits, tau=self.gumbel_temperature, hard=True, dim=-1)
            out = out * gate.unsqueeze(-1)
        
        # === OUTPUT PROJECTION ===
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.embed_dim)
        out = self.out_proj(out)
        
        # Node counts (for KV cache compatibility)
        updated_node_counts = None  # Not used in optimized version
        
        return out, updated_node_counts, aux_loss * self.aux_loss_weight


class HyperGraphRoPE(nn.Module):
    """RoPE for HyperGraph (copied for self-containment)."""
    
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


# === CONVENIENCE FUNCTION ===

def create_optimized_hypergraph(
    embed_dim: int = 512,
    num_heads: int = 8,
    num_timelines: int = 4,
    inference_optimize: bool = True,
    **kwargs
) -> HyperGraphOptimizedAttention:
    """
    Create an optimized HyperGraph attention module.
    
    This is drop-in compatible with the original HyperGraphSparseAttention
    but uses expert-choice routing for better load balance and supports
    inference optimization (only computing query's timeline).
    
    Example:
        attn = create_optimized_hypergraph(512, 8, 4)
        out, _, aux_loss = attn(x)  # x: (1, seq, 512)
    """
    return HyperGraphOptimizedAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_hyper_nodes=num_timelines,
        inference_optimize=inference_optimize,
        **kwargs
    )

