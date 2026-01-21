"""
Checkpoint saving and loading utilities.
"""

import os
import json
import glob
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: Optimizer,
    scheduler: Optional[_LRScheduler],
    epoch: int,
    global_step: int,
    loss: float,
    checkpoint_dir: str,
    config: Optional[Any] = None,
    metrics: Optional[Dict[str, float]] = None,
    keep_last_n: int = 3,
    is_best: bool = False,
) -> str:
    """
    Save training checkpoint.
    
    Args:
        model: The model to save
        optimizer: The optimizer
        scheduler: Learning rate scheduler (optional)
        epoch: Current epoch
        global_step: Global training step
        loss: Current loss
        checkpoint_dir: Directory to save checkpoints
        config: Model/training config (optional)
        metrics: Additional metrics to save (optional)
        keep_last_n: Keep only last N checkpoints (0 for all)
        is_best: If True, also save as best.pt
        
    Returns:
        Path to saved checkpoint
    """
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Build checkpoint dict
    checkpoint = {
        "epoch": epoch,
        "global_step": global_step,
        "loss": loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    
    if config is not None:
        # Handle dataclass configs
        if hasattr(config, "__dataclass_fields__"):
            checkpoint["config"] = {
                k: getattr(config, k) for k in config.__dataclass_fields__
            }
        else:
            checkpoint["config"] = config
    
    if metrics is not None:
        checkpoint["metrics"] = metrics
    
    # Save checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint-{global_step}.pt")
    torch.save(checkpoint, checkpoint_path)
    
    # Save latest symlink/copy
    latest_path = os.path.join(checkpoint_dir, "latest.pt")
    torch.save(checkpoint, latest_path)
    
    # Save best if specified
    if is_best:
        best_path = os.path.join(checkpoint_dir, "best.pt")
        torch.save(checkpoint, best_path)
    
    # Cleanup old checkpoints
    if keep_last_n > 0:
        _cleanup_old_checkpoints(checkpoint_dir, keep_last_n)
    
    return checkpoint_path


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[_LRScheduler] = None,
    device: str = "cpu",
    strict: bool = True,
) -> Dict[str, Any]:
    """
    Load training checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: The model to load weights into
        optimizer: The optimizer to load state into (optional)
        scheduler: The scheduler to load state into (optional)
        device: Device to load tensors to
        strict: Strict loading for model state dict
        
    Returns:
        Dictionary with epoch, global_step, loss, config, metrics
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Load model
    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    
    # Load optimizer
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    # Load scheduler
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    
    return {
        "epoch": checkpoint.get("epoch", 0),
        "global_step": checkpoint.get("global_step", 0),
        "loss": checkpoint.get("loss", float("inf")),
        "config": checkpoint.get("config"),
        "metrics": checkpoint.get("metrics"),
    }


def load_model_only(
    checkpoint_path: str,
    model: torch.nn.Module,
    device: str = "cpu",
    strict: bool = True,
) -> torch.nn.Module:
    """
    Load only model weights from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: The model to load weights into
        device: Device to load tensors to
        strict: Strict loading
        
    Returns:
        Model with loaded weights
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    return model


def _cleanup_old_checkpoints(checkpoint_dir: str, keep_last_n: int):
    """Remove old checkpoints, keeping only the last N."""
    # Find all numbered checkpoints
    pattern = os.path.join(checkpoint_dir, "checkpoint-*.pt")
    checkpoints = glob.glob(pattern)
    
    if len(checkpoints) <= keep_last_n:
        return
    
    # Sort by step number
    def get_step(path):
        try:
            name = os.path.basename(path)
            step = int(name.replace("checkpoint-", "").replace(".pt", ""))
            return step
        except ValueError:
            return 0
    
    checkpoints.sort(key=get_step)
    
    # Remove old checkpoints
    for checkpoint_path in checkpoints[:-keep_last_n]:
        try:
            os.remove(checkpoint_path)
        except OSError:
            pass


def get_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """
    Get path to the latest checkpoint in a directory.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        
    Returns:
        Path to latest checkpoint, or None if not found
    """
    latest_path = os.path.join(checkpoint_dir, "latest.pt")
    if os.path.exists(latest_path):
        return latest_path
    
    # Fall back to finding highest numbered checkpoint
    pattern = os.path.join(checkpoint_dir, "checkpoint-*.pt")
    checkpoints = glob.glob(pattern)
    
    if not checkpoints:
        return None
    
    def get_step(path):
        try:
            name = os.path.basename(path)
            step = int(name.replace("checkpoint-", "").replace(".pt", ""))
            return step
        except ValueError:
            return 0
    
    checkpoints.sort(key=get_step)
    return checkpoints[-1]


def save_model_for_inference(
    model: torch.nn.Module,
    output_path: str,
    config: Optional[Any] = None,
):
    """
    Save model for inference (weights only, no optimizer state).
    
    Args:
        model: The model to save
        output_path: Path to save the model
        config: Model config to save alongside
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    save_dict = {"model_state_dict": model.state_dict()}
    
    if config is not None:
        if hasattr(config, "__dataclass_fields__"):
            save_dict["config"] = {
                k: getattr(config, k) for k in config.__dataclass_fields__
            }
        else:
            save_dict["config"] = config
    
    torch.save(save_dict, output_path)

