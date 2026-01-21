"""
Rotary Position Embedding (RoPE) implementation.

RoPE encodes position information by rotating queries and keys in the attention
mechanism. Introduced in "RoFormer: Enhanced Transformer with Rotary Position Embedding"
(Su et al., 2021).

Key properties:
- Relative position encoding through rotation
- Linear computational complexity
- Extrapolates well to longer sequences than seen during training
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding module.
    
    Args:
        dim: Dimension of the embedding (typically head_dim)
        max_seq_len: Maximum sequence length (default: 8192)
        base: Base for the frequency computation (default: 10000)
        scaling_factor: Factor to scale positions for length extrapolation (default: 1.0)
    """
    
    def __init__(
        self,
        dim: int,
        max_seq_len: int = 8192,
        base: float = 10000.0,
        scaling_factor: float = 1.0,
    ):
        super().__init__()
        
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        self.scaling_factor = scaling_factor
        
        # Compute inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # Cache for cos/sin values
        self._cos_cached = None
        self._sin_cached = None
        self._seq_len_cached = 0
        
    def _update_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        """Update cached cos/sin values if sequence length changes."""
        if seq_len > self._seq_len_cached:
            self._seq_len_cached = seq_len
            
            # Create position indices
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            t = t / self.scaling_factor
            
            # Compute frequencies: (seq_len, dim/2)
            freqs = torch.outer(t, self.inv_freq)
            
            # Create rotation matrix components: (seq_len, dim)
            emb = torch.cat([freqs, freqs], dim=-1)
            
            self._cos_cached = emb.cos().to(dtype)
            self._sin_cached = emb.sin().to(dtype)
    
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        seq_len: Optional[int] = None,
        offset: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary embeddings to queries and keys.
        
        Args:
            q: Query tensor of shape (batch, heads, seq_len, head_dim)
            k: Key tensor of shape (batch, heads, seq_len, head_dim)
            seq_len: Sequence length (inferred from q if not provided)
            offset: Position offset for incremental decoding
        
        Returns:
            Tuple of rotated (q, k) tensors
        """
        if seq_len is None:
            seq_len = q.shape[2]
            
        self._update_cache(seq_len + offset, q.device, q.dtype)
        
        cos = self._cos_cached[offset:offset + seq_len]
        sin = self._sin_cached[offset:offset + seq_len]
        
        q_embed = apply_rotary_pos_emb(q, cos, sin)
        k_embed = apply_rotary_pos_emb(k, cos, sin)
        
        return q_embed, k_embed


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotate half the hidden dims of the input.
    
    For a vector [x1, x2, x3, x4], returns [-x3, -x4, x1, x2]
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """
    Apply rotary position embedding to input tensor.
    
    Args:
        x: Input tensor of shape (batch, heads, seq_len, head_dim)
        cos: Cosine values of shape (seq_len, head_dim)
        sin: Sine values of shape (seq_len, head_dim)
    
    Returns:
        Rotated tensor of same shape as input
    """
    # Broadcast cos/sin to match input shape
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim)
    sin = sin.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim)
    
    return (x * cos) + (rotate_half(x) * sin)


class RoPEAttention(nn.Module):
    """
    Self-attention module with integrated Rotary Position Embeddings.
    
    This combines the attention mechanism with RoPE for convenience.
    
    Args:
        embed_dim: Total dimension of the model
        num_heads: Number of attention heads
        num_kv_heads: Number of key-value heads for GQA (default: None = MHA)
        dropout: Dropout probability (default: 0.0)
        bias: Whether to use bias in projections (default: False)
        max_seq_len: Maximum sequence length for RoPE (default: 8192)
        rope_base: Base frequency for RoPE (default: 10000)
        causal: Whether to apply causal masking (default: True)
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        dropout: float = 0.0,
        bias: bool = False,
        max_seq_len: int = 8192,
        rope_base: float = 10000.0,
        causal: bool = True,
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
        
        # RoPE
        self.rope = RotaryEmbedding(
            dim=self.head_dim,
            max_seq_len=max_seq_len,
            base=rope_base,
        )
        
    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        position_offset: int = 0,
    ) -> torch.Tensor:
        """
        Forward pass with RoPE.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)
            attn_mask: Optional attention mask
            position_offset: Position offset for KV cache (default: 0)
        
        Returns:
            Output tensor of shape (batch_size, seq_len, embed_dim)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE to Q and K
        q, k = self.rope(q, k, seq_len=seq_len, offset=position_offset)
        
        # Expand KV for Grouped Query Attention
        if self.num_kv_groups > 1:
            k = k.unsqueeze(2).expand(-1, -1, self.num_kv_groups, -1, -1)
            k = k.reshape(batch_size, self.num_heads, seq_len, self.head_dim)
            v = v.unsqueeze(2).expand(-1, -1, self.num_kv_groups, -1, -1)
            v = v.reshape(batch_size, self.num_heads, seq_len, self.head_dim)
        
        # Scaled dot-product attention with Flash Attention
        import torch.nn.functional as F
        
        dropout_p = self.dropout if self.training else 0.0
        is_causal = self.causal and (attn_mask is None)
        
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=self.scale,
        )
        
        # Reshape and project output
        out = out.transpose(1, 2).contiguous()
        out = out.reshape(batch_size, seq_len, self.embed_dim)
        out = self.out_proj(out)
        
        return out


class YaRNRoPE(RotaryEmbedding):
    """
    YaRN (Yet another RoPE extensioN) for improved length extrapolation.
    
    Implements the YaRN method from "YaRN: Efficient Context Window Extension
    of Large Language Models" which combines NTK-aware interpolation with
    attention scaling.
    
    Args:
        dim: Dimension of the embedding
        max_seq_len: Maximum sequence length
        base: Base for frequency computation
        original_max_seq_len: Original training sequence length
        scale: Scaling factor for extended context
        beta_fast: Fast beta for interpolation (default: 32)
        beta_slow: Slow beta for interpolation (default: 1)
    """
    
    def __init__(
        self,
        dim: int,
        max_seq_len: int = 8192,
        base: float = 10000.0,
        original_max_seq_len: int = 4096,
        scale: float = 1.0,
        beta_fast: float = 32.0,
        beta_slow: float = 1.0,
    ):
        super().__init__(dim, max_seq_len, base)
        
        self.original_max_seq_len = original_max_seq_len
        self.scale = scale
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        
        # Compute YaRN frequencies
        self._compute_yarn_freqs()
        
    def _compute_yarn_freqs(self):
        """Compute YaRN-adjusted inverse frequencies."""
        pos_freqs = self.base ** (torch.arange(0, self.dim, 2).float() / self.dim)
        inv_freq_extrapolation = 1.0 / pos_freqs
        inv_freq_interpolation = 1.0 / (self.scale * pos_freqs)
        
        low = max(0.0, self.beta_fast / self.scale - 1.0)
        high = min(1.0, self.beta_slow / self.scale - 1.0)
        
        # Linear ramp for blending
        ramp = torch.linspace(0, 1, self.dim // 2)
        ramp = ramp.clamp(low, high)
        ramp = (ramp - low) / (high - low) if high > low else torch.zeros_like(ramp)
        
        inv_freq = inv_freq_interpolation * (1 - ramp) + inv_freq_extrapolation * ramp
        self.register_buffer("inv_freq", inv_freq, persistent=False)


class ALiBiPositionalBias(nn.Module):
    """
    Attention with Linear Biases (ALiBi) - an alternative to RoPE.
    
    ALiBi adds a linear bias to attention scores based on distance between
    positions, rather than modifying the queries and keys.
    
    Args:
        num_heads: Number of attention heads
        max_seq_len: Maximum sequence length
    """
    
    def __init__(self, num_heads: int, max_seq_len: int = 8192):
        super().__init__()
        
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        
        # Compute head-specific slopes
        slopes = self._get_slopes(num_heads)
        self.register_buffer("slopes", slopes, persistent=False)
        
        # Pre-compute bias matrix
        bias = self._compute_bias(max_seq_len)
        self.register_buffer("bias", bias, persistent=False)
        
    def _get_slopes(self, num_heads: int) -> torch.Tensor:
        """Get geometric sequence of slopes for each head."""
        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(torch.log2(torch.tensor(n, dtype=torch.float32)) - 3).item()))
            ratio = start
            return [start * (ratio ** i) for i in range(n)]
        
        if (num_heads & (num_heads - 1)) == 0:  # Power of 2
            return torch.tensor(get_slopes_power_of_2(num_heads))
        else:
            closest_power_of_2 = 2 ** torch.tensor(num_heads).float().log2().floor().int().item()
            slopes = get_slopes_power_of_2(closest_power_of_2)
            extra_slopes = get_slopes_power_of_2(2 * closest_power_of_2)[0::2][:num_heads - closest_power_of_2]
            return torch.tensor(slopes + extra_slopes)
    
    def _compute_bias(self, seq_len: int) -> torch.Tensor:
        """Compute attention bias matrix."""
        positions = torch.arange(seq_len)
        relative_positions = positions.unsqueeze(0) - positions.unsqueeze(1)
        relative_positions = relative_positions.abs().neg().unsqueeze(0)  # (1, seq, seq)
        
        # Scale by slopes: (num_heads, seq, seq)
        bias = relative_positions * self.slopes.view(-1, 1, 1)
        return bias.unsqueeze(0)  # (1, num_heads, seq, seq)
    
    def forward(self, seq_len: int) -> torch.Tensor:
        """
        Get ALiBi bias for given sequence length.
        
        Args:
            seq_len: Current sequence length
        
        Returns:
            Bias tensor of shape (1, num_heads, seq_len, seq_len)
        """
        if seq_len > self.max_seq_len:
            self.bias = self._compute_bias(seq_len).to(self.bias.device)
            self.max_seq_len = seq_len
            
        return self.bias[:, :, :seq_len, :seq_len]

