"""
Common training utilities shared across all training scripts.

This module provides:
- Learning rate scheduling (cosine warmup)
- Evaluation utilities
- Checkpoint management
- Logging utilities

Usage:
    from train.training_utils import get_lr, evaluate_model, save_checkpoint
"""

import os
import math
import json
import time
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


@dataclass
class LRConfig:
    """Learning rate schedule configuration."""
    base_lr: float = 3e-4
    warmup_steps: int = 2000
    total_steps: int = 20000
    min_lr_ratio: float = 0.1
    schedule: str = "cosine"  # cosine, linear, constant
    
    @property
    def min_lr(self) -> float:
        return self.base_lr * self.min_lr_ratio


def get_lr(step: int, config: LRConfig) -> float:
    """
    Get learning rate with cosine warmup schedule.
    
    Args:
        step: Current training step
        config: LR configuration
    
    Returns:
        Learning rate for this step
    """
    base_lr = config.base_lr
    warmup_steps = config.warmup_steps
    total_steps = config.total_steps
    min_lr = config.min_lr
    
    # Warmup phase
    if step < warmup_steps:
        return base_lr * step / max(1, warmup_steps)
    
    # Constant schedule
    if config.schedule == "constant":
        return base_lr
    
    # Decay phase
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, progress)
    
    if config.schedule == "cosine":
        return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))
    elif config.schedule == "linear":
        return base_lr - (base_lr - min_lr) * progress
    else:
        return base_lr


def update_lr(optimizer: torch.optim.Optimizer, lr: float):
    """Update learning rate in optimizer."""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    """
    Evaluate model on a dataloader.
    
    Args:
        model: Model to evaluate
        dataloader: Evaluation dataloader
        device: Device to run on
        max_batches: Maximum number of batches to evaluate (None = all)
    
    Returns:
        Dictionary with 'loss' and 'ppl' keys
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    n_batches = 0
    
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        
        # Forward pass (handle both (logits,) and (logits, aux_loss) returns)
        output = model(input_ids)
        if isinstance(output, tuple):
            logits = output[0]
        else:
            logits = output
        
        # Compute loss
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            reduction='sum'
        )
        
        total_loss += loss.item()
        total_tokens += labels.numel()
        n_batches += 1
        
        if max_batches and n_batches >= max_batches:
            break
    
    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    
    return {
        'loss': avg_loss,
        'ppl': ppl,
        'n_batches': n_batches,
        'n_tokens': total_tokens,
    }


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    step: int = 0,
    val_loss: float = float('inf'),
    config: Optional[Any] = None,
    extra: Optional[Dict[str, Any]] = None,
):
    """
    Save training checkpoint.
    
    Args:
        path: Path to save checkpoint
        model: Model to save
        optimizer: Optimizer to save (optional)
        step: Current training step
        val_loss: Current validation loss
        config: Training configuration (will be converted to dict)
        extra: Extra data to save
    """
    checkpoint = {
        'step': step,
        'val_loss': val_loss,
        'model_state_dict': model.state_dict(),
    }
    
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    if config is not None:
        if hasattr(config, '__dataclass_fields__'):
            checkpoint['config'] = asdict(config)
        else:
            checkpoint['config'] = config
    
    if extra is not None:
        checkpoint.update(extra)
    
    torch.save(checkpoint, path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: torch.device = torch.device('cpu'),
) -> Dict[str, Any]:
    """
    Load training checkpoint.
    
    Args:
        path: Path to checkpoint
        model: Model to load weights into
        optimizer: Optimizer to load state into (optional)
        device: Device to load to
    
    Returns:
        Checkpoint dictionary with metadata
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint


def save_results(
    path: str,
    results: Dict[str, Any],
):
    """Save results to JSON file."""
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)


def format_time(seconds: float) -> str:
    """Format seconds to human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


class TrainingLogger:
    """Simple training logger with progress tracking."""
    
    def __init__(self, total_steps: int, log_every: int = 100, prefix: str = ""):
        self.total_steps = total_steps
        self.log_every = log_every
        self.prefix = prefix
        self.start_time = time.time()
        self.step_times = []
        
        self.accum_loss = 0.0
        self.accum_aux = 0.0
        self.accum_count = 0
    
    def accumulate(self, loss: float, aux_loss: float = 0.0):
        """Accumulate losses for logging."""
        self.accum_loss += loss
        self.accum_aux += aux_loss
        self.accum_count += 1
    
    def should_log(self, step: int) -> bool:
        """Check if we should log at this step."""
        return step > 0 and step % self.log_every == 0
    
    def log(self, step: int, lr: float = 0.0) -> str:
        """Log training progress and return message."""
        if self.accum_count == 0:
            return ""
        
        avg_loss = self.accum_loss / self.accum_count
        avg_aux = self.accum_aux / self.accum_count
        elapsed = time.time() - self.start_time
        progress = step / self.total_steps * 100
        
        msg = f"{self.prefix}step {step:5d}/{self.total_steps} ({progress:.1f}%) | loss: {avg_loss:.4f}"
        if avg_aux > 0:
            msg += f" | aux: {avg_aux:.4f}"
        msg += f" | lr: {lr:.2e} | time: {format_time(elapsed)}"
        
        # Reset accumulators
        self.accum_loss = 0.0
        self.accum_aux = 0.0
        self.accum_count = 0
        
        return msg


class EarlyStopping:
    """Early stopping tracker."""
    
    def __init__(self, patience: int = 5, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.best_step = 0
        self.counter = 0
        self.should_stop = False
    
    def __call__(self, val_loss: float, step: int) -> bool:
        """
        Check if we should stop.
        
        Returns:
            True if this is a new best, False otherwise
        """
        if val_loss < self.best_loss - self.min_delta:
            # New best
            self.best_loss = val_loss
            self.best_step = step
            self.counter = 0
            return True
        else:
            # No improvement
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
            return False
    
    def status(self) -> str:
        """Get status string."""
        if self.counter == 0:
            return "★ NEW BEST"
        else:
            return f"(patience: {self.counter}/{self.patience})"

