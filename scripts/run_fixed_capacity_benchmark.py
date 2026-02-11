#!/usr/bin/env python3
"""
Fixed-Capacity HyperGraph Benchmark - Compare with MoSA

Trains hybrid and pure variants with fixed-capacity timelines
on 2.03B tokens for fair comparison with MoSA.

Key Features:
- Supports batch_size > 1 (unlike original HyperGraph)
- Uses standard attention kernels (FlashAttention compatible)
- Maintains timeline-local RoPE for length generalization
- Probability-based overflow handling when timeline exceeds capacity
"""

import os
import sys
import time
import math
import argparse
from pathlib import Path
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.module.fixed_capacity_block import (
    FixedCapacityModelConfig,
    FixedCapacityCausalLM,
)
from data import FastTokenizer, create_dataloaders


@dataclass
class BenchmarkConfig:
    """Benchmark experiment configuration."""
    name: str
    description: str = ""
    
    # Model
    dim: int = 512
    num_heads: int = 8
    num_layers: int = 14
    num_timelines: int = 4
    block_pattern: str = "FSSFSSFSSFSSFF"  # F=Full, S=Sparse
    
    # Training
    seq_len: int = 1024
    batch_size: int = 4  # ~8GB memory
    grad_accum: int = 4  # Effective batch: 4*4*1024 = 16,384 tokens/step
    lr: float = 3e-4
    num_steps: int = 124000  # 2.03B tokens
    
    # Misc
    vocab_size: int = 50257
    dataset: str = "pg19"


# Define benchmark configurations
BENCHMARKS = [
    # Hybrid variant - optimal pattern from ablations
    BenchmarkConfig(
        name="fixedcap_hybrid_k4",
        description="Fixed-Capacity Hybrid: FSSFSSFSSFSSFF pattern, K=4",
        dim=512,
        num_heads=8,
        num_layers=14,
        num_timelines=4,
        block_pattern="FSSFSSFSSFSSFF",
        num_steps=124000,  # 2.03B tokens
        batch_size=4,  # ~8GB memory
        grad_accum=4,  # Same effective batch: 16,384 tokens/step
    ),
    
    # Pure sparse variant
    BenchmarkConfig(
        name="fixedcap_pure_k4",
        description="Fixed-Capacity Pure: all S layers, K=4",
        dim=512,
        num_heads=8,
        num_layers=14,
        num_timelines=4,
        block_pattern="SSSSSSSSSSSSSS",  # All sparse
        num_steps=124000,
        batch_size=4,
        grad_accum=4,
    ),
    
    # K=6 variant for IsoFLOP comparison
    BenchmarkConfig(
        name="fixedcap_hybrid_k6",
        description="Fixed-Capacity Hybrid: FSSFSSFSSFSSFF pattern, K=6",
        dim=512,
        num_heads=8,
        num_layers=14,
        num_timelines=6,
        block_pattern="FSSFSSFSSFSSFF",
        num_steps=124000,
        batch_size=4,
        grad_accum=4,
    ),
    
    # Deeper variant (15 layers) for fair MoSA comparison
    BenchmarkConfig(
        name="fixedcap_hybrid_deep",
        description="Fixed-Capacity Hybrid: 15 layers, K=4 (matches MoSA deep)",
        dim=512,
        num_heads=8,
        num_layers=15,
        num_timelines=4,
        block_pattern="FSSFSSFSSFSSFSS",  # Extended pattern
        num_steps=124000,
        batch_size=4,
        grad_accum=4,
    ),
]


def train(config: BenchmarkConfig, gpu_id: int, output_dir: str, num_workers: int = 0):
    """Train Fixed-Capacity HyperGraph model."""
    os.makedirs(output_dir, exist_ok=True)
    log_file = Path(output_dir) / f"{config.name}.log"
    
    def log(msg):
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}][GPU {gpu_id}] {msg}"
        # Only write to file (stdout is redirected to same file by nohup)
        with open(log_file, 'a') as f:
            f.write(line + "\n")
            f.flush()
    
    torch.cuda.set_device(gpu_id)
    device = torch.device(f'cuda:{gpu_id}')
    
    log(f"Starting {config.name}: {config.description}")
    log(f"Pattern: {config.block_pattern}, K={config.num_timelines}")
    
    # Data
    tokenizer = FastTokenizer(max_length=config.seq_len)
    loaders = create_dataloaders(
        tokenizer,
        dataset_name=config.dataset,
        max_length=config.seq_len,
        batch_size=config.batch_size,
        num_workers=num_workers,
    )
    
    # Model
    model_config = FixedCapacityModelConfig(
        dim=config.dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        num_timelines=config.num_timelines,
        vocab_size=config.vocab_size,
        max_seq_len=config.seq_len,
        block_pattern=config.block_pattern,
    )
    model = FixedCapacityCausalLM(model_config).to(device)
    
    num_params = model.num_parameters()
    log(f"{config.name}: {num_params:,} params")
    
    # Calculate tokens
    tokens_per_step = config.batch_size * config.grad_accum * config.seq_len
    total_tokens = config.num_steps * tokens_per_step
    log(f"Training: {config.num_steps:,} steps × {tokens_per_step:,} tokens/step = {total_tokens/1e9:.2f}B tokens")
    
    # Optimizer + Cosine Scheduler with Warmup
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0.1)
    warmup_steps = min(2000, config.num_steps // 10)
    
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, config.num_steps - warmup_steps))
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Training loop
    model.train()
    step = 0
    running_loss = 0.0
    running_aux = 0.0
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 5
    start_time = time.time()
    
    optimizer.zero_grad()
    
    for epoch in range(500):
        for batch in loaders["train"]:
            if step >= config.num_steps:
                break
            
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            
            # Forward pass
            logits, aux_loss, loss = model(input_ids, labels)
            
            # Backward with gradient accumulation
            scaled_loss = loss / config.grad_accum
            scaled_loss.backward()
            
            running_loss += loss.item()
            running_aux += aux_loss.item() if isinstance(aux_loss, torch.Tensor) else aux_loss
            
            # Optimizer step
            if (step + 1) % config.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            step += 1
            
            # Logging every 500 steps
            if step % 500 == 0:
                avg_loss = running_loss / 500
                avg_aux = running_aux / 500
                elapsed = time.time() - start_time
                current_lr = scheduler.get_last_lr()[0]
                tokens_seen = step * tokens_per_step
                
                log(f"{config.name} step {step} | loss: {avg_loss:.4f} | aux: {avg_aux:.4f} | lr: {current_lr:.2e} | tokens: {tokens_seen/1e9:.2f}B | time: {elapsed:.1f}s")
                running_loss = 0.0
                running_aux = 0.0
            
            # Validation every 2000 steps
            if step % 2000 == 0:
                model.eval()
                val_losses = []
                
                with torch.no_grad():
                    for i, b in enumerate(loaders["val"]):
                        if i >= 50:
                            break
                        v_logits, _, v_loss = model(
                            b["input_ids"].to(device),
                            b["labels"].to(device)
                        )
                        val_losses.append(v_loss.item())
                
                val_loss = sum(val_losses) / len(val_losses)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    torch.save(model.state_dict(), Path(output_dir) / f"{config.name}_best.pt")
                    log(f"{config.name} >>> val_loss: {val_loss:.4f} ★ NEW BEST")
                else:
                    patience_counter += 1
                    log(f"{config.name} >>> val_loss: {val_loss:.4f} (best: {best_val_loss:.4f}, patience: {patience_counter}/{patience})")
                
                model.train()
        
        if step >= config.num_steps:
            break
    
    # Final summary
    elapsed = time.time() - start_time
    log(f"{config.name} DONE | best_val: {best_val_loss:.4f} | total_time: {elapsed/3600:.2f}h")
    
    return best_val_loss


def main():
    parser = argparse.ArgumentParser(description="Fixed-Capacity HyperGraph Benchmark")
    parser.add_argument("--experiment", type=int, default=0, help="Experiment index (0-3)")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID")
    parser.add_argument("--output-dir", type=str, default="results/fixedcap_benchmark")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")
    args = parser.parse_args()
    
    if args.experiment >= len(BENCHMARKS):
        print(f"Error: experiment {args.experiment} not found. Available: 0-{len(BENCHMARKS)-1}")
        print("\nAvailable experiments:")
        for i, cfg in enumerate(BENCHMARKS):
            print(f"  [{i}] {cfg.name}: {cfg.description}")
        return
    
    config = BENCHMARKS[args.experiment]
    train(config, args.gpu, args.output_dir, args.num_workers)


if __name__ == "__main__":
    main()
