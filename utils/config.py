"""
Configuration loading utilities.

Supports YAML config files with inheritance/overrides.
"""

import os
from typing import Dict, Any, Optional

# Try to import yaml, provide helpful error if not installed
try:
    import yaml
except ImportError:
    yaml = None


def load_yaml(path: str) -> Dict[str, Any]:
    """Load a YAML file."""
    if yaml is None:
        raise ImportError("PyYAML required. Install with: pip install pyyaml")
    
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_yaml(config: Dict[str, Any], path: str):
    """Save config to YAML file."""
    if yaml is None:
        raise ImportError("PyYAML required. Install with: pip install pyyaml")
    
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two config dictionaries.
    
    Override values take precedence over base values.
    """
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    
    return result


def load_config(
    config_path: str,
    base_config_path: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Load experiment configuration.
    
    Args:
        config_path: Path to main config file
        base_config_path: Optional base config to inherit from
        overrides: Optional dict of overrides (e.g., from command line)
        
    Returns:
        Merged configuration dictionary
        
    Example:
        ```python
        config = load_config(
            "configs/hybrid_4_4.yaml",
            base_config_path="configs/base.yaml",
            overrides={"model": {"num_hyper_nodes": 8}}
        )
        ```
    """
    # Load base config if provided
    if base_config_path:
        config = load_yaml(base_config_path)
    else:
        config = {}
    
    # Merge with main config
    main_config = load_yaml(config_path)
    config = merge_configs(config, main_config)
    
    # Apply overrides
    if overrides:
        config = merge_configs(config, overrides)
    
    return config


def config_to_model_config(config: Dict[str, Any]):
    """
    Convert config dict to ModelConfig dataclass.
    
    Args:
        config: Configuration dictionary with 'model' section
        
    Returns:
        ModelConfig instance
    """
    from model import ModelConfig
    
    model_cfg = config.get("model", {})
    
    return ModelConfig(
        dim=model_cfg.get("dim", 512),
        num_heads=model_cfg.get("num_heads", 8),
        num_kv_heads=model_cfg.get("num_kv_heads"),
        n_standard_blocks=model_cfg.get("n_standard_blocks", 4),
        n_sparse_blocks=model_cfg.get("n_sparse_blocks", 4),
        num_hyper_nodes=model_cfg.get("num_hyper_nodes", 4),
        vocab_size=model_cfg.get("vocab_size", 50257),
        max_seq_len=model_cfg.get("max_seq_len", 2048),
        ffn_multiplier=model_cfg.get("ffn_multiplier", 4.0),
        dropout=model_cfg.get("dropout", 0.0),
        attn_dropout=model_cfg.get("attn_dropout", 0.0),
        bias=model_cfg.get("bias", False),
        norm_type=model_cfg.get("norm_type", "rmsnorm"),
        norm_eps=model_cfg.get("norm_eps", 1e-6),
        rope_base=model_cfg.get("rope_base", 10000.0),
        tie_embeddings=model_cfg.get("tie_embeddings", True),
    )


def config_to_training_config(config: Dict[str, Any]):
    """
    Convert config dict to TrainingConfig dataclass.
    
    Args:
        config: Configuration dictionary with 'training' section
        
    Returns:
        TrainingConfig instance
    """
    from model.train import TrainingConfig
    
    train_cfg = config.get("training", {})
    
    return TrainingConfig(
        learning_rate=train_cfg.get("learning_rate", 3e-4),
        min_learning_rate=train_cfg.get("min_learning_rate", 1e-5),
        weight_decay=train_cfg.get("weight_decay", 0.1),
        beta1=train_cfg.get("beta1", 0.9),
        beta2=train_cfg.get("beta2", 0.95),
        eps=train_cfg.get("eps", 1e-8),
        max_grad_norm=train_cfg.get("max_grad_norm", 1.0),
        num_epochs=train_cfg.get("num_epochs", 10),
        warmup_steps=train_cfg.get("warmup_steps", 0),
        warmup_ratio=train_cfg.get("warmup_ratio", 0.1),
        lr_scheduler=train_cfg.get("lr_scheduler", "cosine"),
        batch_size=train_cfg.get("batch_size", 8),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 1),
        mixed_precision=train_cfg.get("mixed_precision", True),
        bf16=train_cfg.get("bf16", False),
        save_every_n_steps=train_cfg.get("save_every_n_steps", 1000),
        save_every_n_epochs=train_cfg.get("save_every_n_epochs", 1),
        keep_last_n_checkpoints=train_cfg.get("keep_last_n_checkpoints", 3),
        checkpoint_dir=train_cfg.get("checkpoint_dir", "checkpoints"),
        log_every_n_steps=train_cfg.get("log_every_n_steps", 10),
        eval_every_n_steps=train_cfg.get("eval_every_n_steps", 500),
        wandb_project=train_cfg.get("wandb_project"),
        wandb_run_name=train_cfg.get("wandb_run_name"),
        max_seq_len=config.get("data", {}).get("max_seq_len", 2048),
        num_workers=config.get("data", {}).get("num_workers", 4),
    )
