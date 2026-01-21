"""
Training utilities for Sparse Attention models.

This module provides:
    - TrainingConfig: Configuration dataclass for training hyperparameters
    - Trainer: Main training loop with AMP, checkpointing, and logging
    - Optimizer utilities: AdamW with proper weight decay grouping
    - Schedulers: Cosine, linear, and constant with warmup
    - Metrics: Loss tracking, perplexity, throughput
    - Checkpointing: Save/load training state

Example:
    ```python
    from model import CausalLM, ModelConfig
    from model.train import Trainer, TrainingConfig
    
    # Create model
    model = CausalLM(ModelConfig.small())
    
    # Configure training
    config = TrainingConfig(
        learning_rate=3e-4,
        batch_size=8,
        num_epochs=10,
        mixed_precision=True,
        checkpoint_dir="./checkpoints",
    )
    
    # Train
    trainer = Trainer(model, config)
    trainer.train(train_dataloader, eval_dataloader)
    ```
"""

from .config import TrainingConfig
from .trainer import Trainer
from .optimizer import (
    create_optimizer,
    create_scheduler,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_constant_schedule_with_warmup,
    clip_grad_norm_,
)
from .metrics import (
    MetricsTracker,
    compute_perplexity,
    compute_node_distribution_stats,
    EarlyStopping,
)
from .checkpoint import (
    save_checkpoint,
    load_checkpoint,
    load_model_only,
    get_latest_checkpoint,
    save_model_for_inference,
)

__all__ = [
    # Config
    "TrainingConfig",
    # Trainer
    "Trainer",
    # Optimizer
    "create_optimizer",
    "create_scheduler",
    "get_cosine_schedule_with_warmup",
    "get_linear_schedule_with_warmup",
    "get_constant_schedule_with_warmup",
    "clip_grad_norm_",
    # Metrics
    "MetricsTracker",
    "compute_perplexity",
    "compute_node_distribution_stats",
    "EarlyStopping",
    # Checkpoint
    "save_checkpoint",
    "load_checkpoint",
    "load_model_only",
    "get_latest_checkpoint",
    "save_model_for_inference",
]

