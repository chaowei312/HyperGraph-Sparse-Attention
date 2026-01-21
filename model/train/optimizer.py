"""
Optimizer and learning rate scheduler utilities.
"""

import math
from typing import Optional, List, Tuple
import torch
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR, _LRScheduler

from .config import TrainingConfig


def create_optimizer(
    model: torch.nn.Module,
    config: TrainingConfig,
) -> AdamW:
    """
    Create AdamW optimizer with weight decay applied only to non-bias, non-norm parameters.
    
    Args:
        model: The model to optimize
        config: Training configuration
        
    Returns:
        Configured AdamW optimizer
    """
    # Separate parameters into decay and no-decay groups
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        # Don't apply weight decay to bias, normalization, or embedding
        if param.ndim == 1 or "bias" in name or "norm" in name or "embedding" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    
    param_groups = [
        {"params": decay_params, "weight_decay": config.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    
    optimizer = AdamW(
        param_groups,
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        eps=config.eps,
    )
    
    return optimizer


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.1,
) -> LambdaLR:
    """
    Create cosine annealing scheduler with linear warmup.
    
    Args:
        optimizer: The optimizer
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total number of training steps
        min_lr_ratio: Minimum LR as a ratio of peak LR
        
    Returns:
        LambdaLR scheduler
    """
    def lr_lambda(current_step: int) -> float:
        # Warmup phase
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        # Cosine decay phase
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    return LambdaLR(optimizer, lr_lambda)


def get_linear_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
) -> LambdaLR:
    """
    Create linear decay scheduler with linear warmup.
    
    Args:
        optimizer: The optimizer
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total number of training steps
        
    Returns:
        LambdaLR scheduler
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        return max(
            0.0,
            float(num_training_steps - current_step)
            / float(max(1, num_training_steps - num_warmup_steps)),
        )
    
    return LambdaLR(optimizer, lr_lambda)


def get_constant_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
) -> LambdaLR:
    """
    Create constant scheduler with linear warmup.
    
    Args:
        optimizer: The optimizer
        num_warmup_steps: Number of warmup steps
        
    Returns:
        LambdaLR scheduler
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return 1.0
    
    return LambdaLR(optimizer, lr_lambda)


def create_scheduler(
    optimizer: Optimizer,
    config: TrainingConfig,
    num_training_steps: int,
) -> _LRScheduler:
    """
    Create learning rate scheduler based on config.
    
    Args:
        optimizer: The optimizer
        config: Training configuration
        num_training_steps: Total number of training steps
        
    Returns:
        Learning rate scheduler
    """
    # Calculate warmup steps
    if config.warmup_steps > 0:
        num_warmup_steps = config.warmup_steps
    else:
        num_warmup_steps = int(num_training_steps * config.warmup_ratio)
    
    # Create scheduler based on type
    if config.lr_scheduler == "cosine":
        min_lr_ratio = config.min_learning_rate / config.learning_rate
        return get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            min_lr_ratio=min_lr_ratio,
        )
    elif config.lr_scheduler == "linear":
        return get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
    elif config.lr_scheduler == "constant":
        return get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
        )
    else:
        raise ValueError(f"Unknown scheduler type: {config.lr_scheduler}")


def clip_grad_norm_(
    model: torch.nn.Module,
    max_norm: float,
) -> float:
    """
    Clip gradient norm and return the total norm.
    
    Args:
        model: The model
        max_norm: Maximum gradient norm
        
    Returns:
        Total gradient norm before clipping
    """
    return torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

