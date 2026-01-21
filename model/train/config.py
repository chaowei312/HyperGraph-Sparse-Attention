"""
Training configuration for Sparse Attention models.
"""

from dataclasses import dataclass, field
from typing import Optional, Literal
import torch


@dataclass
class TrainingConfig:
    """
    Configuration for model training.
    
    Attributes:
        # Optimization
        learning_rate: Peak learning rate
        min_learning_rate: Minimum LR for cosine decay
        weight_decay: L2 regularization (AdamW)
        beta1: Adam beta1
        beta2: Adam beta2
        eps: Adam epsilon
        max_grad_norm: Gradient clipping threshold
        
        # Schedule
        num_epochs: Total training epochs
        warmup_steps: Linear warmup steps
        warmup_ratio: Alternative: warmup as fraction of total steps
        lr_scheduler: Type of LR scheduler
        
        # Batch
        batch_size: Per-device batch size
        gradient_accumulation_steps: Accumulate gradients over N steps
        
        # Mixed precision
        mixed_precision: Enable AMP (auto mixed precision)
        bf16: Use bfloat16 instead of float16
        
        # Checkpointing
        save_every_n_steps: Save checkpoint every N steps
        save_every_n_epochs: Save checkpoint every N epochs
        keep_last_n_checkpoints: Keep only last N checkpoints
        checkpoint_dir: Directory for checkpoints
        resume_from: Path to resume training from
        
        # Logging
        log_every_n_steps: Log metrics every N steps
        eval_every_n_steps: Run evaluation every N steps
        wandb_project: Weights & Biases project name
        wandb_run_name: Weights & Biases run name
        
        # Data
        max_seq_len: Maximum sequence length
        num_workers: DataLoader workers
        
        # Device
        device: Device to train on
        compile_model: Use torch.compile (PyTorch 2.0+)
    """
    # Optimization
    learning_rate: float = 3e-4
    min_learning_rate: float = 1e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    max_grad_norm: float = 1.0
    
    # Schedule
    num_epochs: int = 10
    warmup_steps: int = 0
    warmup_ratio: float = 0.1
    lr_scheduler: Literal["cosine", "linear", "constant"] = "cosine"
    
    # Batch
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    
    # Mixed precision
    mixed_precision: bool = True
    bf16: bool = False
    
    # Checkpointing
    save_every_n_steps: int = 1000
    save_every_n_epochs: int = 1
    keep_last_n_checkpoints: int = 3
    checkpoint_dir: str = "checkpoints"
    resume_from: Optional[str] = None
    
    # Logging
    log_every_n_steps: int = 10
    eval_every_n_steps: int = 500
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None
    
    # Data
    max_seq_len: int = 2048
    num_workers: int = 4
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    compile_model: bool = False
    
    @property
    def effective_batch_size(self) -> int:
        """Total batch size including gradient accumulation."""
        return self.batch_size * self.gradient_accumulation_steps
    
    @property
    def dtype(self) -> torch.dtype:
        """Get the dtype for mixed precision training."""
        if not self.mixed_precision:
            return torch.float32
        return torch.bfloat16 if self.bf16 else torch.float16

