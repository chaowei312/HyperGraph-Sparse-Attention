#!/usr/bin/env python3
"""
Multi-GPU Parallel Training for Architecture Comparison.

Trains multiple architecture configurations in parallel, each on a separate GPU.
Results are saved to JSON files for visualization in demo notebook.

Usage:
    python train/parallel_train.py --num_steps 1000 --output_dir results/
    python train/parallel_train.py --configs baseline ff_10s_ff --gpus 0 1
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from model import ModelConfig, CausalLM
from data import FastTokenizer, create_dataloaders
from train.architectures import ARCHITECTURES, get_model_config as _get_model_config


@dataclass
class TrainConfig:
    """Training configuration."""
    batch_size: int = 1
    grad_accum_steps: int = 16
    seq_len: int = 1024
    lr: float = 3e-4
    num_steps: int = 20000  # Increased to allow full cosine decay
    log_every: int = 100
    eval_every: int = 500  # Eval less frequently for longer training
    save_checkpoint: bool = True
    # Model hyperparameters
    dim: int = 512
    num_heads: int = 8
    num_hyper_nodes: int = 4  # K timelines per head
    top_k: int = 1  # Number of timelines each token routes to
    # Learning rate schedule
    warmup_steps: int = 2000  # 10% of max steps for warmup
    lr_schedule: str = "cosine"  # cosine, linear, or constant
    min_lr_ratio: float = 0.1  # Minimum LR as ratio of max LR (for cosine/linear)
    # Early stopping
    early_stopping: bool = True
    patience: int = 5  # Number of eval steps without improvement before stopping
    min_delta: float = 0.001  # Minimum change to qualify as improvement
    # Load balance auxiliary loss
    aux_loss_weight: float = 0.01  # Weight for load balance loss (0.01 typical for MoE)
    # Dataset
    dataset: str = "gutenberg"  # gutenberg, wikitext-2, wikitext-103, openwebtext, slimpajama


def get_model_config(arch_name: str, train_cfg: TrainConfig) -> ModelConfig:
    """Create ModelConfig for a given architecture using TrainConfig parameters."""
    return _get_model_config(
        arch_name=arch_name,
        dim=train_cfg.dim,
        num_heads=train_cfg.num_heads,
        num_hyper_nodes=train_cfg.num_hyper_nodes,
        top_k=train_cfg.top_k,
        max_seq_len=train_cfg.seq_len * 2,  # Allow for longer context
    )


def get_lr(step: int, train_cfg: TrainConfig) -> float:
    """
    Get learning rate for a given step using the configured schedule.
    
    Supports:
    - cosine: Cosine decay with warmup (recommended)
    - linear: Linear decay with warmup
    - constant: Constant LR (no schedule)
    """
    import math
    
    base_lr = train_cfg.lr
    warmup_steps = train_cfg.warmup_steps
    total_steps = train_cfg.num_steps
    min_lr = base_lr * train_cfg.min_lr_ratio
    
    # Warmup phase
    if step < warmup_steps:
        return base_lr * step / warmup_steps
    
    # After warmup
    if train_cfg.lr_schedule == "constant":
        return base_lr
    
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, progress)  # Clamp to [0, 1]
    
    if train_cfg.lr_schedule == "cosine":
        # Cosine decay to min_lr
        return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))
    elif train_cfg.lr_schedule == "linear":
        # Linear decay to min_lr
        return base_lr - (base_lr - min_lr) * progress
    else:
        return base_lr


def train_single_config(
    gpu_id: int,
    config_name: str,
    train_cfg: TrainConfig,
    output_dir: str,
    result_queue: mp.Queue,
):
    """
    Train a single configuration on a specific GPU.
    This function runs in a separate process.
    """
    try:
        # Set GPU
        torch.cuda.set_device(gpu_id)
        device = torch.device(f'cuda:{gpu_id}')
        
        print(f"[GPU {gpu_id}] Starting {config_name}")
        
        # Create model config
        model_config = get_model_config(config_name, train_cfg)
        pattern = ''.join(model_config.get_block_pattern())
        n_sparse = pattern.count('S')
        sparse_pct = n_sparse / len(pattern) * 100
        
        print(f"[GPU {gpu_id}] {config_name}: {pattern} ({sparse_pct:.1f}% sparse)")
        
        # Load data
        tokenizer = FastTokenizer(max_length=train_cfg.seq_len)
        loaders = create_dataloaders(
            tokenizer,
            dataset_name=train_cfg.dataset,
            max_length=train_cfg.seq_len,
            batch_size=train_cfg.batch_size,
            num_workers=2,
            max_train_samples=None,
            max_eval_samples=None,
        )
        
        # Create model
        model = CausalLM(model_config).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr)
        
        print(f"[GPU {gpu_id}] {config_name}: {model.num_parameters():,} params")
        
        # Training history
        history = {
            "config_name": config_name,
            "pattern": pattern,
            "sparse_pct": sparse_pct,
            "gpu_id": gpu_id,
            "step": [],
            "train_loss": [],
            "val_loss": [],
            "lr": [],
        }
        
        # Early stopping state
        best_val_loss = float('inf')
        best_step = 0
        patience_counter = 0
        best_model_state = None
        early_stopped = False
        
        # Training loop
        model.train()
        step = 0
        micro_step = 0
        accum_loss = 0.0
        running_loss = 0.0
        start_time = time.time()
        
        optimizer.zero_grad()
        
        for epoch in range(50):  # Enough epochs for num_steps
            for batch in loaders["train"]:
                if step >= train_cfg.num_steps:
                    break
                
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                
                logits, aux_loss = model(input_ids)
                ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
                # Add load balance auxiliary loss (only applies to sparse layers)
                loss = ce_loss + train_cfg.aux_loss_weight * aux_loss
                scaled_loss = loss / train_cfg.grad_accum_steps
                scaled_loss.backward()
                
                accum_loss += loss.item()
                micro_step += 1
                
                if micro_step % train_cfg.grad_accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    
                    # Update learning rate
                    lr = get_lr(step, train_cfg)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr
                    
                    optimizer.step()
                    optimizer.zero_grad()
                    step += 1
                    
                    running_loss += accum_loss / train_cfg.grad_accum_steps
                    accum_loss = 0.0
                    
                    # Logging
                    if step % train_cfg.log_every == 0:
                        avg_loss = running_loss / train_cfg.log_every
                        current_lr = get_lr(step, train_cfg)
                        elapsed = time.time() - start_time
                        print(f"[GPU {gpu_id}] {config_name} step {step:4d} | loss: {avg_loss:.4f} | lr: {current_lr:.2e} | time: {elapsed:.1f}s")
                        history["step"].append(step)
                        history["train_loss"].append(avg_loss)
                        history["lr"].append(current_lr)
                        running_loss = 0.0
                    
                    # Validation
                    if step % train_cfg.eval_every == 0:
                        model.eval()
                        val_losses = []
                        with torch.no_grad():
                            for b in loaders["val"]:
                                v_logits, _ = model(b["input_ids"].to(device))
                                v_loss = F.cross_entropy(
                                    v_logits.view(-1, model_config.vocab_size),
                                    b["labels"].to(device).view(-1)
                                )
                                val_losses.append(v_loss.item())
                        val_loss = sum(val_losses) / len(val_losses)
                        history["val_loss"].append(val_loss)
                        
                        # Early stopping check
                        if train_cfg.early_stopping:
                            if val_loss < best_val_loss - train_cfg.min_delta:
                                # Improvement found
                                best_val_loss = val_loss
                                best_step = step
                                patience_counter = 0
                                # Save best model state
                                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                                print(f"[GPU {gpu_id}] {config_name} >>> val_loss: {val_loss:.4f} ★ NEW BEST")
                            else:
                                # No improvement
                                patience_counter += 1
                                print(f"[GPU {gpu_id}] {config_name} >>> val_loss: {val_loss:.4f} (patience: {patience_counter}/{train_cfg.patience})")
                                
                                if patience_counter >= train_cfg.patience:
                                    print(f"[GPU {gpu_id}] {config_name} >>> EARLY STOPPING at step {step} (best was {best_val_loss:.4f} at step {best_step})")
                                    early_stopped = True
                        else:
                            print(f"[GPU {gpu_id}] {config_name} >>> val_loss: {val_loss:.4f}")
                        
                        model.train()
                        
                        if early_stopped:
                            break
            
            if step >= train_cfg.num_steps or early_stopped:
                break
        
        # Restore best model if early stopping was used and we have a saved state
        if train_cfg.early_stopping and best_model_state is not None:
            model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
            print(f"[GPU {gpu_id}] {config_name} >>> Restored best model from step {best_step}")
        
        # Test evaluation
        if "test" in loaders and len(loaders["test"]) > 0:
            model.eval()
            test_losses = []
            with torch.no_grad():
                for b in loaders["test"]:
                    t_logits, _ = model(b["input_ids"].to(device))
                    t_loss = F.cross_entropy(
                        t_logits.view(-1, model_config.vocab_size),
                        b["labels"].to(device).view(-1)
                    )
                    test_losses.append(t_loss.item())
            test_loss = sum(test_losses) / len(test_losses)
            history["test_loss"] = test_loss
            print(f"[GPU {gpu_id}] {config_name} >>> TEST loss: {test_loss:.4f}")
        
        history["total_time"] = time.time() - start_time
        history["final_train_loss"] = history["train_loss"][-1] if history["train_loss"] else None
        history["final_val_loss"] = history["val_loss"][-1] if history["val_loss"] else None
        
        # Early stopping metadata
        history["early_stopping"] = {
            "enabled": train_cfg.early_stopping,
            "early_stopped": early_stopped,
            "best_val_loss": best_val_loss if best_val_loss != float('inf') else None,
            "best_step": best_step,
            "patience": train_cfg.patience,
        }
        
        # Save results
        output_path = Path(output_dir) / f"{config_name}.json"
        with open(output_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        print(f"[GPU {gpu_id}] {config_name} DONE in {history['total_time']:.1f}s | Saved to {output_path}")
        
        # Save checkpoint if requested
        if train_cfg.save_checkpoint:
            ckpt_path = Path(output_dir) / f"{config_name}_checkpoint.pt"
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': asdict(model_config) if hasattr(model_config, '__dataclass_fields__') else vars(model_config),
                'history': history,
            }, ckpt_path)
            print(f"[GPU {gpu_id}] Checkpoint saved to {ckpt_path}")
        
        result_queue.put((config_name, history, None))
        
    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        print(f"[GPU {gpu_id}] {config_name} FAILED: {e}")
        result_queue.put((config_name, None, error_msg))


def main():
    parser = argparse.ArgumentParser(description="Multi-GPU Parallel Training")
    parser.add_argument("--configs", nargs="+", default=list(ARCHITECTURES.keys()),
                        help="Config names to train")
    parser.add_argument("--gpus", nargs="+", type=int, default=None,
                        help="GPU IDs to use (default: all available)")
    parser.add_argument("--num_steps", type=int, default=20000,
                        help="Number of training steps")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size (must be 1 for xformers)")
    parser.add_argument("--grad_accum", type=int, default=16,
                        help="Gradient accumulation steps")
    parser.add_argument("--seq_len", type=int, default=512,
                        help="Sequence length")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=2000,
                        help="Warmup steps for LR schedule (default: 10% of max steps)")
    parser.add_argument("--lr_schedule", type=str, default="cosine",
                        choices=["cosine", "linear", "constant"],
                        help="LR schedule: cosine, linear, or constant")
    parser.add_argument("--min_lr_ratio", type=float, default=0.1,
                        help="Minimum LR as ratio of max LR (for cosine/linear)")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Output directory for results")
    parser.add_argument("--save_checkpoint", action="store_true",
                        help="Save model checkpoints")
    parser.add_argument("--sequential", action="store_true",
                        help="Run sequentially instead of parallel")
    parser.add_argument("--dim", type=int, default=512,
                        help="Model dimension (d_model)")
    parser.add_argument("--num_heads", type=int, default=8,
                        help="Number of attention heads")
    parser.add_argument("--num_hyper_nodes", type=int, default=4,
                        help="Number of hyper nodes (K timelines)")
    parser.add_argument("--top_k", type=int, default=1,
                        help="Number of timelines each token routes to (1=hard, 2=soft)")
    parser.add_argument("--run_benchmark", action="store_true",
                        help="Run benchmarks after training")
    # Early stopping
    parser.add_argument("--early_stopping", action="store_true", default=True,
                        help="Enable early stopping (default: True)")
    parser.add_argument("--no_early_stopping", action="store_false", dest="early_stopping",
                        help="Disable early stopping")
    parser.add_argument("--patience", type=int, default=3,
                        help="Early stopping patience (number of evals without improvement)")
    parser.add_argument("--min_delta", type=float, default=0.001,
                        help="Minimum improvement to reset patience")
    # Dataset
    parser.add_argument("--dataset", type=str, default="gutenberg",
                        choices=["gutenberg", "wikitext-2", "wikitext-103", "openwebtext", "slimpajama"],
                        help="Dataset to use (default: gutenberg)")
    args = parser.parse_args()
    
    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get available GPUs
    num_gpus = torch.cuda.device_count()
    if args.gpus is None:
        gpu_ids = list(range(num_gpus))
    else:
        gpu_ids = args.gpus
    
    # Training config
    train_cfg = TrainConfig(
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        seq_len=args.seq_len,
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        lr_schedule=args.lr_schedule,
        min_lr_ratio=args.min_lr_ratio,
        num_steps=args.num_steps,
        save_checkpoint=args.save_checkpoint,
        dim=args.dim,
        num_heads=args.num_heads,
        num_hyper_nodes=args.num_hyper_nodes,
        top_k=args.top_k,
        early_stopping=args.early_stopping,
        patience=args.patience,
        min_delta=args.min_delta,
        dataset=args.dataset,
    )
    
    print("=" * 80)
    print("MULTI-GPU PARALLEL TRAINING")
    print("=" * 80)
    print(f"Dataset: {train_cfg.dataset}")
    print(f"Configs: {args.configs}")
    print(f"GPUs available: {num_gpus}, using: {gpu_ids}")
    print(f"Steps: {train_cfg.num_steps} (max)")
    print(f"Effective batch: {train_cfg.batch_size} × {train_cfg.grad_accum_steps} = {train_cfg.batch_size * train_cfg.grad_accum_steps}")
    print(f"Model: dim={train_cfg.dim}, heads={train_cfg.num_heads}, K={train_cfg.num_hyper_nodes}")
    print(f"LR schedule: {train_cfg.lr_schedule}, warmup={train_cfg.warmup_steps}, base_lr={train_cfg.lr}, min_lr_ratio={train_cfg.min_lr_ratio}")
    if train_cfg.early_stopping:
        print(f"Early stopping: patience={train_cfg.patience}, min_delta={train_cfg.min_delta}")
    else:
        print(f"Early stopping: DISABLED")
    print(f"Output: {output_dir}")
    print("=" * 80)
    
    # List architectures
    print("\nArchitectures to train:")
    for name in args.configs:
        arch = ARCHITECTURES[name]
        pattern = ''.join(arch["block_pattern"])
        print(f"  {name:<20} {pattern:<20} {arch['description']}")
    print()
    
    if args.sequential:
        # Sequential training (explicit request)
        print(f"Running SEQUENTIALLY (--sequential flag)")
        
        all_results = {}
        for i, config_name in enumerate(args.configs):
            gpu_id = gpu_ids[i % len(gpu_ids)]
            result_queue = mp.Queue()
            
            train_single_config(
                gpu_id=gpu_id,
                config_name=config_name,
                train_cfg=train_cfg,
                output_dir=str(output_dir),
                result_queue=result_queue,
            )
            
            name, history, error = result_queue.get()
            if error:
                print(f"❌ {name} failed")
            else:
                all_results[name] = history
                print(f"✅ {name} completed")
    else:
        # Parallel training (in batches if configs > GPUs)
        n_gpus = len(gpu_ids)
        n_configs = len(args.configs)
        n_batches = (n_configs + n_gpus - 1) // n_gpus  # ceiling division
        
        print(f"Running {n_configs} configs in PARALLEL on {n_gpus} GPUs ({n_batches} batch(es))")
        
        mp.set_start_method('spawn', force=True)
        all_results = {}
        start_time = time.time()
        
        # Process in batches
        for batch_idx in range(n_batches):
            batch_start = batch_idx * n_gpus
            batch_end = min(batch_start + n_gpus, n_configs)
            batch_configs = args.configs[batch_start:batch_end]
            
            if n_batches > 1:
                print(f"\n--- Batch {batch_idx + 1}/{n_batches}: {batch_configs} ---")
            
            result_queue = mp.Queue()
            processes = []
            
            for i, config_name in enumerate(batch_configs):
                gpu_id = gpu_ids[i]
                p = mp.Process(
                    target=train_single_config,
                    args=(gpu_id, config_name, train_cfg, str(output_dir), result_queue)
                )
                p.start()
                processes.append(p)
            
            # Collect results for this batch
            for _ in range(len(batch_configs)):
                name, history, error = result_queue.get()
                if error:
                    print(f"❌ {name} failed: {error[:200]}...")
                else:
                    all_results[name] = history
            
            # Wait for all processes in this batch
            for p in processes:
                p.join()
        
        total_time = time.time() - start_time
        print(f"\n{'='*80}")
        print(f"ALL TRAINING COMPLETE in {total_time:.1f}s")
        print(f"{'='*80}")
    
    # Save combined results
    combined_path = output_dir / "all_results.json"
    with open(combined_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nCombined results saved to {combined_path}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'Config':<20} {'Pattern':<16} {'Sparse%':<10} {'Val Loss':<12} {'Test Loss':<12} {'Time(s)':<10}")
    print("-" * 80)
    
    for name, hist in sorted(all_results.items(), key=lambda x: x[1].get('final_val_loss', 999)):
        pattern = hist.get('pattern', 'N/A')[:14]
        sparse_pct = hist.get('sparse_pct', 0)
        val_loss = hist.get('final_val_loss', 'N/A')
        test_loss = hist.get('test_loss', 'N/A')
        total_time = hist.get('total_time', 0)
        
        val_str = f"{val_loss:.4f}" if isinstance(val_loss, float) else val_loss
        test_str = f"{test_loss:.4f}" if isinstance(test_loss, float) else test_loss
        
        print(f"{name:<20} {pattern:<16} {sparse_pct:<10.1f} {val_str:<12} {test_str:<12} {total_time:<10.1f}")
    
    # Run benchmarks if requested
    if args.run_benchmark:
        print("\n" + "=" * 80)
        print("RUNNING BENCHMARKS")
        print("=" * 80)
        
        from train.benchmark import run_benchmarks
        
        run_benchmarks(
            results_dir=output_dir,
            configs=list(all_results.keys()),
            seq_lengths=[256, 512, 1024, 2048, 4096],
            prefill_lens=[512, 1024, 2048],
            dim=train_cfg.dim,
            num_heads=train_cfg.num_heads,
            num_hyper_nodes=train_cfg.num_hyper_nodes,
            eval_test=True,
            gpu_id=0,
        )


if __name__ == "__main__":
    main()

