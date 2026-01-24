"""
Training metrics and evaluation utilities.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import time
import torch
import torch.nn.functional as F


@dataclass
class MetricsTracker:
    """
    Track and aggregate training metrics.
    
    Tracks:
        - Loss (train/eval)
        - Perplexity
        - Gradient norms
        - Learning rate
        - Throughput (tokens/sec)
        - Sparse attention statistics
    """
    # Running totals for averaging
    _loss_sum: float = 0.0
    _loss_count: int = 0
    _grad_norm_sum: float = 0.0
    _grad_norm_count: int = 0
    _token_count: int = 0
    _step_count: int = 0
    
    # Time tracking
    _start_time: float = field(default_factory=time.time)
    _last_log_time: float = field(default_factory=time.time)
    _last_log_tokens: int = 0
    
    # Current values
    current_lr: float = 0.0
    current_loss: float = 0.0
    current_grad_norm: float = 0.0
    
    def __post_init__(self):
        self._start_time = time.time()
        self._last_log_time = time.time()
    
    def reset(self):
        """Reset all tracked metrics."""
        self._loss_sum = 0.0
        self._loss_count = 0
        self._grad_norm_sum = 0.0
        self._grad_norm_count = 0
        self._token_count = 0
        self._step_count = 0
        self._start_time = time.time()
        self._last_log_time = time.time()
        self._last_log_tokens = 0
    
    def update(
        self,
        loss: float,
        grad_norm: Optional[float] = None,
        lr: Optional[float] = None,
        num_tokens: int = 0,
    ):
        """Update metrics with values from current step."""
        self._loss_sum += loss
        self._loss_count += 1
        self.current_loss = loss
        
        if grad_norm is not None:
            self._grad_norm_sum += grad_norm
            self._grad_norm_count += 1
            self.current_grad_norm = grad_norm
        
        if lr is not None:
            self.current_lr = lr
        
        self._token_count += num_tokens
        self._step_count += 1
    
    @property
    def avg_loss(self) -> float:
        """Average loss since last reset."""
        if self._loss_count == 0:
            return 0.0
        return self._loss_sum / self._loss_count
    
    @property
    def avg_grad_norm(self) -> float:
        """Average gradient norm since last reset."""
        if self._grad_norm_count == 0:
            return 0.0
        return self._grad_norm_sum / self._grad_norm_count
    
    @property
    def perplexity(self) -> float:
        """Perplexity from average loss."""
        return min(float("inf"), 2.0 ** self.avg_loss)
    
    @property
    def tokens_per_second(self) -> float:
        """Overall tokens per second."""
        elapsed = time.time() - self._start_time
        if elapsed == 0:
            return 0.0
        return self._token_count / elapsed
    
    @property
    def instantaneous_tokens_per_second(self) -> float:
        """Tokens per second since last log."""
        current_time = time.time()
        elapsed = current_time - self._last_log_time
        tokens_diff = self._token_count - self._last_log_tokens
        
        if elapsed == 0:
            return 0.0
        return tokens_diff / elapsed
    
    def mark_log(self):
        """Mark current position for instantaneous throughput calculation."""
        self._last_log_time = time.time()
        self._last_log_tokens = self._token_count
    
    def get_metrics(self) -> Dict[str, float]:
        """Get all metrics as a dictionary."""
        return {
            "loss": self.current_loss,
            "avg_loss": self.avg_loss,
            "perplexity": self.perplexity,
            "grad_norm": self.current_grad_norm,
            "avg_grad_norm": self.avg_grad_norm,
            "lr": self.current_lr,
            "tokens_per_sec": self.tokens_per_second,
            "instantaneous_tok_per_sec": self.instantaneous_tokens_per_second,
            "total_tokens": self._token_count,
            "steps": self._step_count,
        }
    
    def get_log_string(self, step: int, epoch: Optional[int] = None) -> str:
        """Format metrics for logging."""
        parts = []
        if epoch is not None:
            parts.append(f"epoch={epoch}")
        parts.append(f"step={step}")
        parts.append(f"loss={self.current_loss:.4f}")
        parts.append(f"ppl={self.perplexity:.2f}")
        parts.append(f"lr={self.current_lr:.2e}")
        parts.append(f"grad_norm={self.current_grad_norm:.4f}")
        parts.append(f"tok/s={self.instantaneous_tokens_per_second:.0f}")
        
        return " | ".join(parts)


@torch.no_grad()
def compute_perplexity(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> float:
    """
    Compute perplexity on a dataset.
    
    Args:
        model: The model to evaluate
        dataloader: DataLoader with evaluation data
        device: Device to use
        max_batches: Maximum number of batches (None for all)
        
    Returns:
        Perplexity score
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    for batch_idx, batch in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        
        input_ids = batch["input_ids"].to(device)
        labels = batch.get("labels", input_ids[:, 1:]).to(device)
        
        # Forward pass (returns logits and aux_loss)
        logits, _ = model(input_ids)
        
        # Shift for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels.contiguous()
        
        if shift_labels.shape[1] != shift_logits.shape[1]:
            shift_labels = shift_labels[:, :shift_logits.shape[1]]
        
        # Compute loss
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="sum",
            ignore_index=-100,
        )
        
        # Count non-padded tokens
        num_tokens = (shift_labels != -100).sum().item()
        
        total_loss += loss.item()
        total_tokens += num_tokens
    
    model.train()
    
    if total_tokens == 0:
        return float("inf")
    
    avg_loss = total_loss / total_tokens
    perplexity = 2.0 ** avg_loss
    
    return min(perplexity, float("inf"))


@torch.no_grad()
def compute_node_distribution_stats(
    node_probs: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute statistics about node assignment distribution.
    
    Useful for monitoring sparse attention behavior during training.
    
    Args:
        node_probs: Node probabilities [batch, num_heads, seq_len, num_nodes]
        
    Returns:
        Dictionary with distribution statistics
    """
    # Entropy of node assignments (higher = more uniform)
    entropy = -(node_probs * (node_probs + 1e-10).log()).sum(dim=-1).mean()
    
    # Maximum probability (how confident assignments are)
    max_prob = node_probs.max(dim=-1).values.mean()
    
    # Per-node usage (how balanced is usage across nodes)
    node_usage = node_probs.mean(dim=(0, 1, 2))  # Average over batch, heads, seq
    usage_std = node_usage.std()
    
    return {
        "node_entropy": entropy.item(),
        "node_max_prob": max_prob.item(),
        "node_usage_std": usage_std.item(),
        "node_usage": node_usage.tolist(),
    }


class EarlyStopping:
    """
    Early stopping based on validation loss.
    
    Args:
        patience: Number of evaluations to wait for improvement
        min_delta: Minimum change to qualify as improvement
        mode: 'min' for loss, 'max' for accuracy
    """
    
    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.0,
        mode: str = "min",
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = None
        self.should_stop = False
    
    def __call__(self, value: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            value: Current metric value
            
        Returns:
            True if training should stop
        """
        if self.best_value is None:
            self.best_value = value
            return False
        
        if self.mode == "min":
            improved = value < self.best_value - self.min_delta
        else:
            improved = value > self.best_value + self.min_delta
        
        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        
        return self.should_stop
    
    def reset(self):
        """Reset early stopping state."""
        self.counter = 0
        self.best_value = None
        self.should_stop = False

