"""
Training utilities and scripts for HyperGraph Sparse Attention.

Main scripts:
- benchmark_comparison.py: Fair comparison of Baseline vs HyperGraph vs MoH
- parallel_train.py: Multi-GPU parallel training with architecture search
- benchmark.py: Inference speed benchmarking

Shared utilities:
- training_utils.py: Common training utilities (LR schedule, evaluation, etc.)
- architectures.py: Architecture definitions and configurations
"""

from .training_utils import (
    LRConfig,
    get_lr,
    update_lr,
    evaluate_model,
    save_checkpoint,
    load_checkpoint,
    save_results,
    TrainingLogger,
    EarlyStopping,
)

from .architectures import (
    ARCHITECTURES,
    get_model_config,
)

__all__ = [
    # LR utilities
    "LRConfig",
    "get_lr",
    "update_lr",
    # Evaluation
    "evaluate_model",
    # Checkpointing
    "save_checkpoint",
    "load_checkpoint",
    "save_results",
    # Logging
    "TrainingLogger",
    "EarlyStopping",
    # Architectures
    "ARCHITECTURES",
    "get_model_config",
]
