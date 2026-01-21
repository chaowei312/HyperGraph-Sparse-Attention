"""
Training script for Sparse Attention experiments.

Usage:
    python train.py --config configs/hybrid_4_4.yaml
    python train.py --config configs/hybrid_4_4.yaml --base configs/base.yaml
"""

import argparse
import gc
import torch


def log_memory():
    """Log GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"    GPU Memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")

from data import Tokenizer, create_dataloaders
from model import CausalLM, ModelConfig
from model.train import Trainer, TrainingConfig
from utils import load_config, config_to_model_config, config_to_training_config


def main():
    parser = argparse.ArgumentParser(description="Train Sparse Attention model")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--base", type=str, default="configs/base.yaml", help="Base config")
    parser.add_argument("--dataset", type=str, default="wikitext-2", help="Dataset name")
    parser.add_argument("--tiny", action="store_true", help="Tiny: ~64K tokens (500 samples)")
    parser.add_argument("--small", action="store_true", help="Small: ~256K tokens (2000 samples)")
    args = parser.parse_args()
    
    print("=" * 60)
    print("SPARSE ATTENTION TRAINING")
    print("=" * 60)
    
    # Load config
    print(f"\n[1] Loading config: {args.config}")
    config = load_config(args.config, base_config_path=args.base)
    
    model_config = config_to_model_config(config)
    train_config = config_to_training_config(config)
    
    # Dataset size modes (all use WikiText-2 subsets)
    max_train_samples = None
    max_eval_samples = None
    
    if args.tiny:
        print("    TINY MODE: ~64K tokens (WikiText-2 subset)")
        train_config.num_epochs = 1
        train_config.batch_size = 4
        train_config.log_every_n_steps = 5
        train_config.eval_every_n_steps = 100
        train_config.save_every_n_steps = 0
        train_config.num_workers = 0
        model_config.max_seq_len = 128
        max_train_samples = 500   # ~64K tokens
        max_eval_samples = 50
        args.dataset = "wikitext-2"  # Force WikiText-2
    
    elif args.small:
        print("    SMALL MODE: ~256K tokens (WikiText-2 subset)")
        train_config.num_epochs = 1
        train_config.batch_size = 4
        train_config.log_every_n_steps = 10
        train_config.eval_every_n_steps = 200
        train_config.save_every_n_steps = 0
        train_config.num_workers = 0
        model_config.max_seq_len = 128
        max_train_samples = 2000  # ~256K tokens
        max_eval_samples = 100
        args.dataset = "wikitext-2"  # Force WikiText-2
    
    print(f"    Model: {model_config.n_standard_blocks} standard + {model_config.n_sparse_blocks} sparse blocks")
    print(f"    dim={model_config.dim}, heads={model_config.num_heads}, K={model_config.num_hyper_nodes}")
    
    # Setup tokenizer
    print(f"\n[2] Loading tokenizer (GPT-2)")
    tokenizer = Tokenizer(max_length=model_config.max_seq_len)
    print(f"    vocab_size={tokenizer.vocab_size}")
    
    # Load dataset
    print(f"\n[3] Loading dataset: {args.dataset}")
    loaders = create_dataloaders(
        tokenizer,
        dataset_name=args.dataset,
        max_length=model_config.max_seq_len,
        batch_size=train_config.batch_size,
        num_workers=train_config.num_workers,
        max_train_samples=max_train_samples,
        max_eval_samples=max_eval_samples,
    )
    print(f"    Train batches: {len(loaders['train']):,}")
    if "val" in loaders:
        print(f"    Val batches: {len(loaders['val']):,}")
    
    # Create model
    print(f"\n[4] Creating model")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_memory()
    model = CausalLM(model_config)
    model = model.to(device)
    print(f"    Device: {device}")
    print(f"    Parameters: {model.num_parameters():,}")
    print(model)
    log_memory()
    
    # Create trainer
    print(f"\n[5] Starting training")
    print(f"    Epochs: {train_config.num_epochs}")
    print(f"    Batch size: {train_config.batch_size}")
    print(f"    Learning rate: {train_config.learning_rate}")
    print(f"    Seq length: {model_config.max_seq_len}")
    
    trainer = Trainer(model, train_config)
    
    # Train
    print("\n" + "-" * 60)
    metrics = trainer.train(
        train_dataloader=loaders["train"],
        eval_dataloader=loaders.get("val"),
    )
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Final loss: {metrics['avg_loss']:.4f}")
    print(f"Total tokens: {metrics['total_tokens']:,}")


if __name__ == "__main__":
    main()


