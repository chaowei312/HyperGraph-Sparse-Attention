"""
Benchmark comparison: Baseline vs HyperGraph vs MoH

Trains models in parallel on separate GPUs for fair comparison.
Uses shared training utilities and model definitions.

Usage:
    # With command line args
    python train/benchmark_comparison.py --num_steps 20000 --gpus 0 1 2
    
    # With JSON config file
    python train/benchmark_comparison.py --config configs/benchmark_20k.json
"""

import os
import sys
import json
import time
import argparse
import multiprocessing as mp
from dataclasses import dataclass, asdict, field
from typing import Optional, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Use existing modules instead of duplicating
from model import CausalLM, ModelConfig
from model.module.mixture_of_heads_attention import GQAMixtureOfHeadsAttention
from model.module.flash_attention import FlashMultiHeadAttention
from data.dataset import create_dataloaders
from train.training_utils import (
    LRConfig, get_lr, update_lr,
    evaluate_model, save_checkpoint, load_checkpoint, save_results,
    TrainingLogger, EarlyStopping,
)
import tiktoken


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark training."""
    # Model architecture
    dim: int = 256
    num_heads: int = 4
    num_hyper_nodes: int = 4  # K timelines for HyperGraph
    top_k: int = 2  # Top-K routing for HyperGraph
    # GQA-style MoH: 6 KV groups, 4 active, 2 Q per KV = 5.33N² FLOPs, 2.25x wall-clock speedup
    moh_num_kv_groups: int = 6  # Total KV groups for MoH
    moh_active_kv_groups: int = 4  # Active KV groups per token
    moh_q_heads_per_kv: int = 2  # Q heads per KV group
    # Legacy (for backward compat)
    moh_num_heads: int = 6  # Alias for num_kv_groups
    moh_active_heads: int = 4  # Alias for active_kv_groups
    num_layers: int = 12
    vocab_size: int = 50257
    max_seq_len: int = 1024
    
    # Architecture pattern: F=Full, S=Sparse
    pattern: str = "FFSSFSSFSSFF"
    
    # Training
    batch_size: int = 1
    grad_accum: int = 16
    num_steps: int = 20000
    lr: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 2000
    min_lr_ratio: float = 0.1
    aux_loss_weight: float = 0.01
    
    # Per-model LR scaling (slower learning for faster-converging models)
    # Based on previous convergence: baseline=5000, moh=4000, hypergraph=3600 steps
    lr_scale_baseline: float = 1.0    # Reference
    lr_scale_hypergraph: float = 0.72  # 3600/5000 = 0.72
    lr_scale_moh: float = 0.8          # 4000/5000 = 0.8
    
    # Early stopping
    patience: int = 5
    min_delta: float = 0.001
    
    # Logging
    log_every: int = 100
    eval_every: int = 500
    
    # Dataset
    dataset: str = "wikitext-103-small"
    seq_len: int = 1024
    
    # Output
    output_dir: str = "results/benchmark"
    save_checkpoint: bool = True
    
    # Models and GPUs
    models: List[str] = field(default_factory=lambda: ["baseline", "hypergraph", "moh"])
    gpus: List[int] = field(default_factory=lambda: [0, 1, 2])
    
    def get_lr_scale(self, model_type: str) -> float:
        """Get LR scaling factor for a model type."""
        scales = {
            "baseline": self.lr_scale_baseline,
            "hypergraph": self.lr_scale_hypergraph,
            "moh": self.lr_scale_moh,
        }
        return scales.get(model_type, 1.0)
    
    @classmethod
    def from_json(cls, path: str) -> "BenchmarkConfig":
        """Load config from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls(**data)
    
    def to_json(self, path: str):
        """Save config to JSON file."""
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)
    
    def get_lr_config(self, model_type: str = "baseline") -> LRConfig:
        """Get LR schedule config with per-model scaling."""
        scaled_lr = self.lr * self.get_lr_scale(model_type)
        return LRConfig(
            base_lr=scaled_lr,
            warmup_steps=self.warmup_steps,
            total_steps=self.num_steps,
            min_lr_ratio=self.min_lr_ratio,
        )


# ============================================================================
# Model Builders - Thin wrappers that use existing model infrastructure
# ============================================================================

def build_baseline_model(config: BenchmarkConfig) -> nn.Module:
    """Build baseline model using CausalLM with all full attention layers."""
    model_config = ModelConfig(
        dim=config.dim,
        num_heads=config.num_heads,
        vocab_size=config.vocab_size,
        max_seq_len=config.max_seq_len,
        block_pattern="F" * config.num_layers,
    )
    return CausalLM(model_config)


def build_hypergraph_model(config: BenchmarkConfig) -> nn.Module:
    """Build HyperGraph sparse attention model using CausalLM."""
    model_config = ModelConfig(
        dim=config.dim,
        num_heads=config.num_heads,
        vocab_size=config.vocab_size,
        max_seq_len=config.max_seq_len,
        block_pattern=config.pattern,
        num_hyper_nodes=config.num_hyper_nodes,
        top_k=config.top_k,
    )
    return CausalLM(model_config)


class MoHBlock(nn.Module):
    """Decoder block using GQA-style Mixture of Heads attention."""
    
    def __init__(self, embed_dim: int, num_kv_groups: int, active_kv_groups: int, 
                 q_heads_per_kv: int = 2, max_seq_len: int = 1024):
        super().__init__()
        self.attn = GQAMixtureOfHeadsAttention(
            embed_dim=embed_dim,
            num_kv_groups=num_kv_groups,
            active_kv_groups=active_kv_groups,
            q_heads_per_kv=q_heads_per_kv,
            max_seq_len=max_seq_len,
        )
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim, bias=False),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim, bias=False),
        )
    
    def forward(self, x, aux_state=None):
        h = self.ln1(x)
        attn_out, new_state, aux_loss = self.attn(h, kv_counts=aux_state)
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x, new_state, aux_loss


class FullBlock(nn.Module):
    """Standard full attention block (minimal version for MoH model)."""
    
    def __init__(self, embed_dim: int, num_heads: int, max_seq_len: int = 1024):
        super().__init__()
        self.attn = FlashMultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads, causal=True)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim, bias=False),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim, bias=False),
        )
    
    def forward(self, x, aux_state=None):
        h = self.ln1(x)
        x = x + self.attn(h)
        x = x + self.ffn(self.ln2(x))
        return x, aux_state, torch.tensor(0.0, device=x.device)


class MoHLanguageModel(nn.Module):
    """Language model with MoH attention (needs separate class due to different head config)."""
    
    def __init__(self, config: BenchmarkConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.dim)
        
        # Build layers according to pattern
        self.layers = nn.ModuleList()
        for char in config.pattern:
            if char == 'F':
                self.layers.append(FullBlock(config.dim, config.num_heads, config.max_seq_len))
            elif char == 'S':
                self.layers.append(MoHBlock(
                    config.dim, config.moh_num_kv_groups, config.moh_active_kv_groups,
                    config.moh_q_heads_per_kv, config.max_seq_len
                ))
        
        self.ln_f = nn.LayerNorm(config.dim)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight  # Tie weights
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                torch.nn.init.ones_(module.weight)
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, input_ids):
        x = self.token_embedding(input_ids)
        aux_state = torch.zeros(input_ids.size(0), self.config.moh_num_kv_groups, device=input_ids.device, dtype=torch.long)
        total_aux = 0.0
        
        for layer in self.layers:
            x, aux_state, aux_loss = layer(x, aux_state)
            total_aux = total_aux + aux_loss
        
        logits = self.lm_head(self.ln_f(x))
        return logits, total_aux
    
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())


def build_model(config: BenchmarkConfig, model_type: str) -> nn.Module:
    """Build model based on type."""
    if model_type == "baseline":
        return build_baseline_model(config)
    elif model_type == "hypergraph":
        return build_hypergraph_model(config)
    elif model_type == "moh":
        return MoHLanguageModel(config)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


# ============================================================================
# Training Loop - Uses shared training utilities
# ============================================================================

def train_single(config: BenchmarkConfig, model_type: str, gpu_id: int):
    """Train a single model on a specific GPU."""
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(device)
    prefix = f"[GPU {gpu_id}] {model_type}: "
    
    print(f"{prefix}Starting...")
    pattern = "F" * config.num_layers if model_type == "baseline" else config.pattern
    print(f"{prefix}Pattern: {pattern}")
    
    # Data
    tokenizer = tiktoken.get_encoding("gpt2")
    loaders = create_dataloaders(
        tokenizer, dataset_name=config.dataset, max_length=config.seq_len,
        batch_size=config.batch_size, num_workers=2,
    )
    
    # Model
    model = build_model(config, model_type).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{prefix}{n_params:,} params")
    
    # Optimizer (use scaled LR for this model type)
    lr_config = config.get_lr_config(model_type)
    optimizer = AdamW(model.parameters(), lr=lr_config.base_lr, weight_decay=config.weight_decay)
    
    print(f"{prefix}LR: {lr_config.base_lr:.2e} (scale={config.get_lr_scale(model_type):.2f})")
    
    # Training state
    logger = TrainingLogger(config.num_steps, config.log_every, prefix)
    early_stop = EarlyStopping(config.patience, config.min_delta)
    best_model_state = None
    
    # Training loop
    model.train()
    optimizer.zero_grad()
    step = 0
    train_iter = iter(loaders["train"])
    
    while step < config.num_steps and not early_stop.should_stop:
        # Get batch
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(loaders["train"])
            batch = next(train_iter)
        
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        
        # Forward
        output = model(input_ids)
        logits, aux_loss = output if isinstance(output, tuple) else (output, 0.0)
        
        ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
        loss = ce_loss + config.aux_loss_weight * (aux_loss if isinstance(aux_loss, torch.Tensor) else torch.tensor(aux_loss))
        (loss / config.grad_accum).backward()
        
        logger.accumulate(ce_loss.item(), aux_loss.item() if isinstance(aux_loss, torch.Tensor) else aux_loss)
        step += 1
        
        # Optimizer step
        if step % config.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            lr = get_lr(step, lr_config)
            update_lr(optimizer, lr)
            optimizer.step()
            optimizer.zero_grad()
        
        # Logging
        if logger.should_log(step):
            print(logger.log(step, get_lr(step, lr_config)))
        
        # Validation
        if step % config.eval_every == 0:
            metrics = evaluate_model(model, loaders["val"], device, max_batches=50)
            is_best = early_stop(metrics['loss'], step)
            
            if is_best:
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                if config.save_checkpoint:
                    save_checkpoint(
                        os.path.join(config.output_dir, f"{model_type}_best.pt"),
                        model, optimizer, step, metrics['loss'], config
                    )
            
            print(f"{prefix}val_loss: {metrics['loss']:.4f} {early_stop.status()}")
            model.train()
    
    # Test evaluation
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    test_metrics = evaluate_model(model, loaders["test"], device)
    print(f"{prefix}Test Loss: {test_metrics['loss']:.4f}, PPL: {test_metrics['ppl']:.2f}")
    
    # Save results
    results = {
        "model_type": model_type,
        "pattern": pattern,
        "num_params": n_params,
        "best_val_loss": early_stop.best_loss,
        "test_loss": test_metrics['loss'],
        "test_ppl": test_metrics['ppl'],
        "final_step": step,
        "config": asdict(config),
    }
    save_results(os.path.join(config.output_dir, f"{model_type}_results.json"), results)
    
    return results


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Benchmark Comparison: Baseline vs HyperGraph vs MoH")
    
    parser.add_argument("--config", type=str, help="Path to JSON config file")
    
    # Model
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_hyper_nodes", type=int, default=4)
    parser.add_argument("--top_k", type=int, default=2)
    parser.add_argument("--moh_num_kv_groups", type=int, default=6, help="Total KV groups for GQA MoH")
    parser.add_argument("--moh_active_kv_groups", type=int, default=4, help="Active KV groups per token")
    parser.add_argument("--moh_q_heads_per_kv", type=int, default=2, help="Q heads per KV group")
    # Legacy aliases
    parser.add_argument("--moh_num_heads", type=int, default=6, help="(Legacy) Alias for moh_num_kv_groups")
    parser.add_argument("--moh_active_heads", type=int, default=4, help="(Legacy) Alias for moh_active_kv_groups")
    parser.add_argument("--pattern", type=str, default="FFSSFSSFSSFF")
    
    # Training
    parser.add_argument("--num_steps", type=int, default=20000)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup_steps", type=int, default=2000)
    parser.add_argument("--patience", type=int, default=5)
    
    # Per-model LR scaling (to equalize convergence speed)
    parser.add_argument("--lr_scale_baseline", type=float, default=1.0)
    parser.add_argument("--lr_scale_hypergraph", type=float, default=0.72,
                        help="LR scale for hypergraph (0.72 = 3600/5000 convergence ratio)")
    parser.add_argument("--lr_scale_moh", type=float, default=0.8,
                        help="LR scale for moh (0.8 = 4000/5000 convergence ratio)")
    
    # Data
    parser.add_argument("--dataset", type=str, default="wikitext-103-small")
    parser.add_argument("--seq_len", type=int, default=1024)
    
    # Output
    parser.add_argument("--output_dir", type=str, default="results/benchmark_20k")
    parser.add_argument("--no_checkpoint", action="store_true")
    
    # Execution
    parser.add_argument("--models", type=str, nargs="+", default=["baseline", "hypergraph", "moh"])
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--sequential", action="store_true")
    
    args = parser.parse_args()
    
    # Load or create config
    if args.config:
        config = BenchmarkConfig.from_json(args.config)
    else:
        config = BenchmarkConfig(
            dim=args.dim, num_heads=args.num_heads, num_hyper_nodes=args.num_hyper_nodes,
            top_k=args.top_k, 
            moh_num_kv_groups=args.moh_num_kv_groups, moh_active_kv_groups=args.moh_active_kv_groups,
            moh_q_heads_per_kv=args.moh_q_heads_per_kv,
            moh_num_heads=args.moh_num_heads, moh_active_heads=args.moh_active_heads,
            pattern=args.pattern, num_layers=len(args.pattern), num_steps=args.num_steps,
            batch_size=args.batch_size, grad_accum=args.grad_accum, lr=args.lr,
            warmup_steps=args.warmup_steps, patience=args.patience, dataset=args.dataset,
            seq_len=args.seq_len, output_dir=args.output_dir, save_checkpoint=not args.no_checkpoint,
            models=args.models, gpus=args.gpus,
            lr_scale_baseline=args.lr_scale_baseline,
            lr_scale_hypergraph=args.lr_scale_hypergraph,
            lr_scale_moh=args.lr_scale_moh,
        )
    
    os.makedirs(config.output_dir, exist_ok=True)
    config.to_json(os.path.join(config.output_dir, "config.json"))
    
    # Print config
    print("=" * 80)
    print("BENCHMARK: Baseline vs HyperGraph vs MoH")
    print("=" * 80)
    print(f"Models: {config.models}")
    print(f"GPUs: {config.gpus[:len(config.models)]}")
    print(f"Pattern: {config.pattern}")
    print(f"Training: {config.num_steps} steps, warmup={config.warmup_steps}")
    print(f"Dataset: {config.dataset}")
    print("=" * 80)
    
    # Run training
    model_gpu_pairs = list(zip(config.models, config.gpus[:len(config.models)]))
    
    if args.sequential:
        for model_type, gpu_id in model_gpu_pairs:
            train_single(config, model_type, gpu_id)
    else:
        ctx = mp.get_context('spawn')
        processes = [ctx.Process(target=train_single, args=(config, m, g)) for m, g in model_gpu_pairs]
        for p in processes:
            p.start()
        for p in processes:
            p.join()
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for model_type in config.models:
        result_path = os.path.join(config.output_dir, f"{model_type}_results.json")
        if os.path.exists(result_path):
            with open(result_path) as f:
                r = json.load(f)
            print(f"{model_type:12}: Loss={r['test_loss']:.4f}, PPL={r['test_ppl']:.2f}, Steps={r['final_step']}, Params={r['num_params']:,}")


if __name__ == "__main__":
    main()
