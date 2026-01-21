"""
SwiGLU (Swish-Gated Linear Unit) implementation.

SwiGLU is an activation function introduced in the paper "GLU Variants Improve Transformer"
(Shazeer, 2020). It combines the Swish activation with a gating mechanism, providing
improved performance over standard FFN layers in transformers.

The formulation is: SwiGLU(x, W, V, b, c) = Swish(xW + b) ⊙ (xV + c)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SwiGLU(nn.Module):
    """
    SwiGLU activation function module.
    
    Computes: Swish(x @ W1) * (x @ W2)
    where Swish(x) = x * sigmoid(x)
    
    Args:
        in_features: Size of input features
        hidden_features: Size of hidden features (default: 4 * in_features)
        out_features: Size of output features (default: in_features)
        bias: Whether to use bias in linear layers (default: False)
        dropout: Dropout probability (default: 0.0)
    """
    
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        bias: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        out_features = out_features or in_features
        hidden_features = hidden_features or int(4 * in_features)
        
        # For SwiGLU, we use 2/3 of the hidden dimension for each gate
        # to maintain parameter count similar to standard FFN
        self.hidden_features = int(2 * hidden_features / 3)
        
        # Gate and up projections (combined for efficiency)
        self.gate_up_proj = nn.Linear(in_features, 2 * self.hidden_features, bias=bias)
        
        # Down projection
        self.down_proj = nn.Linear(self.hidden_features, out_features, bias=bias)
        
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of SwiGLU.
        
        Args:
            x: Input tensor of shape (..., in_features)
        
        Returns:
            Output tensor of shape (..., out_features)
        """
        # Project to gate and up simultaneously
        gate_up = self.gate_up_proj(x)
        gate, up = gate_up.chunk(2, dim=-1)
        
        # SwiGLU: swish(gate) * up
        hidden = F.silu(gate) * up
        
        # Apply dropout and down projection
        hidden = self.dropout(hidden)
        output = self.down_proj(hidden)
        
        return output


class SwiGLUFFN(nn.Module):
    """
    Full Feed-Forward Network block using SwiGLU activation.
    
    This is a drop-in replacement for standard transformer FFN blocks.
    Includes optional layer normalization and residual connection.
    
    Args:
        dim: Model dimension
        hidden_dim: Hidden dimension (default: 4 * dim, adjusted for SwiGLU)
        dropout: Dropout probability (default: 0.0)
        bias: Whether to use bias (default: False)
        layer_norm: Whether to apply layer norm (default: False)
        residual: Whether to add residual connection (default: False)
    """
    
    def __init__(
        self,
        dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
        bias: bool = False,
        layer_norm: bool = False,
        residual: bool = False,
    ):
        super().__init__()
        
        hidden_dim = hidden_dim or 4 * dim
        
        self.norm = nn.LayerNorm(dim) if layer_norm else nn.Identity()
        self.swiglu = SwiGLU(
            in_features=dim,
            hidden_features=hidden_dim,
            out_features=dim,
            bias=bias,
            dropout=dropout,
        )
        self.residual = residual
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (..., dim)
        
        Returns:
            Output tensor of shape (..., dim)
        """
        residual = x
        x = self.norm(x)
        x = self.swiglu(x)
        
        if self.residual:
            x = x + residual
            
        return x


class GeGLU(nn.Module):
    """
    GeGLU (GELU-Gated Linear Unit) variant.
    
    Similar to SwiGLU but uses GELU activation instead of Swish.
    
    Args:
        in_features: Size of input features
        hidden_features: Size of hidden features
        out_features: Size of output features
        bias: Whether to use bias (default: False)
    """
    
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        bias: bool = False,
    ):
        super().__init__()
        
        out_features = out_features or in_features
        hidden_features = hidden_features or int(4 * in_features)
        self.hidden_features = int(2 * hidden_features / 3)
        
        self.gate_up_proj = nn.Linear(in_features, 2 * self.hidden_features, bias=bias)
        self.down_proj = nn.Linear(self.hidden_features, out_features, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj(x)
        gate, up = gate_up.chunk(2, dim=-1)
        hidden = F.gelu(gate) * up
        return self.down_proj(hidden)


class ReGLU(nn.Module):
    """
    ReGLU (ReLU-Gated Linear Unit) variant.
    
    Uses ReLU activation for the gating mechanism.
    
    Args:
        in_features: Size of input features
        hidden_features: Size of hidden features
        out_features: Size of output features
        bias: Whether to use bias (default: False)
    """
    
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        bias: bool = False,
    ):
        super().__init__()
        
        out_features = out_features or in_features
        hidden_features = hidden_features or int(4 * in_features)
        self.hidden_features = int(2 * hidden_features / 3)
        
        self.gate_up_proj = nn.Linear(in_features, 2 * self.hidden_features, bias=bias)
        self.down_proj = nn.Linear(self.hidden_features, out_features, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj(x)
        gate, up = gate_up.chunk(2, dim=-1)
        hidden = F.relu(gate) * up
        return self.down_proj(hidden)

