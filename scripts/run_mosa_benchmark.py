#!/usr/bin/env python3
"""
MoSA Benchmark - Compare with HyperGraph Sparse Attention

Runs MoSA (Mixture of Sparse Attention) with same settings as our ablation experiments:
- 70M parameters
- PG-19 dataset
- Cosine LR schedule with warmup
- Same batch size, sequence length
"""

import os
import sys
import time
import math
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, "/home/lopedg/project/MoSA")

from mosa import MoSA, PureMoSA
from data import FastTokenizer, create_dataloaders


@dataclass
class MoSAModelConfig:
    """Configuration for MoSA-based language model."""
    dim: int = 512
    num_layers: int = 14
    num_mosa_heads: int = 4
    num_dense_heads: int = 4
    head_dim: int = 64
    sparsity: int = 4  # Similar to K=4 timelines
    max_seq_len: int = 1024
    vocab_size: int = 50257
    ffn_multiplier: float = 4.0
    dropout: float = 0.0
    hybrid_type: str = "dense"  # "dense" or "local"
    
    @property
    def ffn_dim(self):
        return int(self.dim * self.ffn_multiplier)


class SwiGLU(nn.Module):
    """SwiGLU feed-forward network."""
    def __init__(self, in_features, hidden_features, out_features, bias=True, dropout=0.0):
        super().__init__()
        self.w1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.w2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.w3 = nn.Linear(in_features, hidden_features, bias=bias)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class MoSADecoderBlock(nn.Module):
    """Transformer decoder block with MoSA attention."""
    def __init__(self, config: MoSAModelConfig):
        super().__init__()
        self.norm1 = nn.RMSNorm(config.dim)
        self.norm2 = nn.RMSNorm(config.dim)
        
        # MoSA attention (hybrid with dense heads)
        self.attn = MoSA(
            h=config.dim,
            h_prim=config.head_dim,
            num_mosa_heads=config.num_mosa_heads,
            num_other_heads=config.num_dense_heads,
            max_seq_len=config.max_seq_len,
            sparsity=config.sparsity,
            hybrid_type=config.hybrid_type,
            include_first=1,  # Include first token for causal
            rotate_fraction=0.5,
            rope_base=10000,
        )
        
        # FFN
        self.ffn = SwiGLU(
            config.dim,
            config.ffn_dim,
            config.dim,
            dropout=config.dropout,
        )
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class MoSACausalLM(nn.Module):
    """Causal Language Model with MoSA attention."""
    def __init__(self, config: MoSAModelConfig):
        super().__init__()
        self.config = config
        
        self.embed = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList([
            MoSADecoderBlock(config) for _ in range(config.num_layers)
        ])
        self.norm = nn.RMSNorm(config.dim)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        
        # Tie weights
        self.lm_head.weight = self.embed.weight
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids):
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits, torch.tensor(0.0, device=logits.device)  # No aux loss
    
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())


@dataclass
class BenchmarkConfig:
    """Benchmark experiment configuration."""
    name: str = "mosa_hybrid"
    description: str = "MoSA with 4 sparse + 4 dense heads"
    
    # Model
    dim: int = 512
    num_layers: int = 14
    num_mosa_heads: int = 4
    num_dense_heads: int = 4
    head_dim: int = 64
    sparsity: int = 4
    hybrid_type: str = "dense"
    
    # Training
    seq_len: int = 1024
    batch_size: int = 16
    grad_accum: int = 8
    lr: float = 3e-4
    num_steps: int = 85000  # Chinchilla optimal for ~70M
    
    # Data
    dataset: str = "pg19"


def train_mosa(config: BenchmarkConfig, gpu_id: int, output_dir: str, num_workers: int = 0):
    """Train MoSA model."""
    os.makedirs(output_dir, exist_ok=True)
    log_file = Path(output_dir) / f"{config.name}_gpu{gpu_id}.log"
    
    def log(msg):
        timestamp = time.strftime("%H:%M:%S")
        line = f"[{timestamp}][GPU {gpu_id}] {msg}"
        print(line, flush=True)
        with open(log_file, 'a') as f:
            f.write(line + "\n")
    
    torch.cuda.set_device(gpu_id)
    device = torch.device(f'cuda:{gpu_id}')
    
    log(f"Starting {config.name}: {config.description}")
    
    # Create model config
    model_config = MoSAModelConfig(
        dim=config.dim,
        num_layers=config.num_layers,
        num_mosa_heads=config.num_mosa_heads,
        num_dense_heads=config.num_dense_heads,
        head_dim=config.head_dim,
        sparsity=config.sparsity,
        max_seq_len=config.seq_len,
        hybrid_type=config.hybrid_type,
    )
    
    log(f"{config.name}: sparsity={config.sparsity}, mosa_heads={config.num_mosa_heads}, dense_heads={config.num_dense_heads}")
    
    # Load data
    tokenizer = FastTokenizer(max_length=config.seq_len)
    loaders = create_dataloaders(
        tokenizer,
        dataset_name=config.dataset,
        max_length=config.seq_len,
        batch_size=config.batch_size,
        num_workers=num_workers,
    )
    
    # Create model
    model = MoSACausalLM(model_config).to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    
    # Cosine scheduler with warmup
    warmup_steps = min(2000, config.num_steps // 10)
    
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        else:
            progress = float(step - warmup_steps) / float(max(1, config.num_steps - warmup_steps))
            return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    log(f"{config.name}: {model.num_parameters():,} params (warmup={warmup_steps}, cosine decay)")
    
    # Training loop
    model.train()
    step = 0
    accum_loss = 0.0
    running_loss = 0.0
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
            
            logits, _ = model(input_ids)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1)
            )
            scaled_loss = loss / config.grad_accum
            scaled_loss.backward()
            
            accum_loss += loss.item()
            
            if (step + 1) % config.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            step += 1
            running_loss += accum_loss / config.grad_accum if (step % config.grad_accum == 0) else 0
            if step % config.grad_accum == 0:
                accum_loss = 0.0
            
            # Log every 500 steps
            if step % 500 == 0:
                avg_loss = running_loss / 500 * config.grad_accum
                elapsed = time.time() - start_time
                current_lr = scheduler.get_last_lr()[0]
                log(f"{config.name} step {step} | loss: {avg_loss:.4f} | lr: {current_lr:.2e} | time: {elapsed:.1f}s")
                running_loss = 0.0
            
            # Validate every 2000 steps
            if step % 2000 == 0:
                model.eval()
                val_losses = []
                with torch.no_grad():
                    for i, b in enumerate(loaders["val"]):
                        if i >= 50:
                            break
                        v_logits, _ = model(b["input_ids"].to(device))
                        v_loss = F.cross_entropy(
                            v_logits.view(-1, model_config.vocab_size),
                            b["labels"].to(device).view(-1)
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
                
                if patience_counter >= patience:
                    log(f"{config.name} early stopping at step {step}")
                    break
                
                model.train()
        
        if step >= config.num_steps or patience_counter >= patience:
            break
    
    # Extrapolation evaluation
    log(f"{config.name}: Running extrapolation evaluation...")
    model.eval()
    model.load_state_dict(torch.load(Path(output_dir) / f"{config.name}_best.pt"))
    
    for eval_len in [2048, 4096]:
        try:
            eval_tokenizer = FastTokenizer(max_length=eval_len)
            eval_loaders = create_dataloaders(
                eval_tokenizer,
                dataset_name=config.dataset,
                max_length=eval_len,
                batch_size=max(1, config.batch_size // (eval_len // config.seq_len)),
                num_workers=0,
            )
            
            eval_losses = []
            with torch.no_grad():
                for i, b in enumerate(eval_loaders["val"]):
                    if i >= 20:
                        break
                    e_logits, _ = model(b["input_ids"].to(device))
                    e_loss = F.cross_entropy(
                        e_logits.view(-1, model_config.vocab_size),
                        b["labels"].to(device).view(-1)
                    )
                    eval_losses.append(e_loss.item())
            
            avg_eval_loss = sum(eval_losses) / len(eval_losses) if eval_losses else float('inf')
            log(f"{config.name}: seq_len={eval_len} → loss={avg_eval_loss:.4f}")
        except Exception as e:
            log(f"{config.name}: seq_len={eval_len} → FAILED: {e}")
    
    log(f"{config.name} DONE | best_val: {best_val_loss:.4f} @ step {step}")
    return best_val_loss


# Define benchmark configurations (ISO-PARAMS: ~70M to match HyperGraph)
# HyperGraph baseline: dim=512, 8 heads, head_dim=64, 14 layers ≈ 70M params
# Training: 85K steps, cosine schedule, same as hybrid/extrap experiments

MOSA_BENCHMARKS = [
    # === PRIMARY EXPERIMENTS (Fair comparison with HyperGraph) ===
    
    # 1. Pure MoSA - all 8 heads are content-based sparse (~70M params)
    # NOTE: Uses 2.0B tokens (same as IsoFLOP experiments) for fair comparison
    BenchmarkConfig(
        name="mosa_pure_8head",
        description="Pure MoSA: 8 sparse heads, sparsity=4, dim=460 (~70M)",
        dim=460,  # Reduced to match HyperGraph's ~70M params
        num_layers=14,
        num_mosa_heads=8,
        num_dense_heads=0,
        head_dim=58,  # 460/8 ≈ 58
        sparsity=4,  # Each head selects seq/4 = 256 tokens
        hybrid_type="dense",  # Ignored when dense_heads=0
        num_steps=124000,  # 2.03B tokens (EXACT match with IsoFLOP experiments)
        batch_size=8,   # Reduced for memory (OOM with 16)
        grad_accum=2,   # Same effective batch: 8*2*1024 = 16,384 tokens/step
    ),
    
    # 2. Hybrid MoSA - 4 sparse + 4 dense (similar to HyperGraph's FSSFSS pattern)
    BenchmarkConfig(
        name="mosa_hybrid_4s4d",
        description="Hybrid MoSA: 4 sparse + 4 dense heads",
        dim=512,
        num_layers=14,
        num_mosa_heads=4,
        num_dense_heads=4,
        head_dim=64,
        sparsity=4,
        hybrid_type="dense",  # Dense heads use full causal attention
        num_steps=124000,  # 2.03B tokens (EXACT match with IsoFLOP experiments)
        batch_size=8,   # Reduced for memory (OOM with 16)
        grad_accum=2,   # Same effective batch: 8*2*1024 = 16,384 tokens/step
    ),
    
    # === ALTERNATIVE CONFIGS ===
    
    # 3. Pure MoSA - 4 heads (more aggressive sparsity per head)
    BenchmarkConfig(
        name="mosa_pure_4head",
        description="Pure MoSA: 4 sparse heads, sparsity=4",
        dim=512,
        num_layers=14,
        num_mosa_heads=4,
        num_dense_heads=0,
        head_dim=64,
        sparsity=4,
        hybrid_type="dense",
        num_steps=85000,
    ),
    
    # 4. Hybrid MoSA with local attention instead of dense
    BenchmarkConfig(
        name="mosa_hybrid_4s4l",
        description="Hybrid MoSA: 4 sparse + 4 local attention heads",
        dim=512,
        num_layers=14,
        num_mosa_heads=4,
        num_dense_heads=4,
        head_dim=64,
        sparsity=4,
        hybrid_type="local",  # Local window attention
        num_steps=85000,
    ),
    
    # === ISO-FLOP CONFIGS (Same inference cost as HyperGraph 14L @ d=512) ===
    
    # 5. ISO-FLOP Pure MoSA - Deeper (15 layers)
    # MoSA attention is K² cheaper, so can afford +1 layer for same cost
    BenchmarkConfig(
        name="mosa_isoflop_deep",
        description="ISO-FLOP Pure MoSA: 15 layers (matches HyperGraph 14L cost)",
        dim=460,  # Match ~70M params base
        num_layers=15,  # +1 layer vs HyperGraph
        num_mosa_heads=8,
        num_dense_heads=0,
        head_dim=58,
        sparsity=4,
        hybrid_type="dense",
        num_steps=124000,  # 2.03B tokens (EXACT match with IsoFLOP)
        batch_size=8,   # Reduced for memory
        grad_accum=2,   # Same effective batch: 16,384 tokens/step
    ),
    
    # 6. ISO-FLOP Pure MoSA - Wider (d=528)
    # Same 14 layers but wider to match FLOPs
    BenchmarkConfig(
        name="mosa_isoflop_wide",
        description="ISO-FLOP Pure MoSA: d=528 (matches HyperGraph 14L cost)",
        dim=528,
        num_layers=14,
        num_mosa_heads=8,
        num_dense_heads=0,
        head_dim=66,  # 528/8
        sparsity=4,
        hybrid_type="dense",
        num_steps=124000,  # 2.03B tokens (EXACT match with IsoFLOP)
        batch_size=8,   # Reduced for memory
        grad_accum=2,   # Same effective batch: 16,384 tokens/step
    ),
    
    # 7. ISO-FLOP Hybrid MoSA - Deeper (15 layers)
    BenchmarkConfig(
        name="mosa_hybrid_isoflop_deep",
        description="ISO-FLOP Hybrid MoSA: 15 layers, 4 sparse + 4 dense",
        dim=512,
        num_layers=15,
        num_mosa_heads=4,
        num_dense_heads=4,
        head_dim=64,
        sparsity=4,
        hybrid_type="dense",
        num_steps=124000,  # 2.03B tokens (EXACT match with IsoFLOP)
        batch_size=8,   # Reduced for memory
        grad_accum=2,   # Same effective batch: 16,384 tokens/step
    ),
    
    # 8. ISO-FLOP for Long Context (8K seq) - Much deeper
    # At 8K tokens, MoSA can afford 19 layers for same cost
    BenchmarkConfig(
        name="mosa_isoflop_8k",
        description="ISO-FLOP MoSA for 8K context: 19 layers",
        dim=480,
        num_layers=19,  # +5 layers vs HyperGraph
        num_mosa_heads=8,
        num_dense_heads=0,
        head_dim=60,
        sparsity=4,
        hybrid_type="dense",
        seq_len=8192,  # Long context
        num_steps=15500,  # 2.03B tokens / (16*8192) = 15.5K steps
        batch_size=2,   # Long context needs smaller batch
        grad_accum=8,   # Same effective batch: 2*8*8192 = 131K tokens/step
    ),
]


def run_all_primary(gpus: list, output_dir: str, num_workers: int = 0):
    """Run primary experiments (pure + hybrid) on specified GPUs."""
    import subprocess
    
    # Primary experiments: indices 0 and 1
    experiments = [(0, gpus[0]), (1, gpus[1] if len(gpus) > 1 else gpus[0])]
    
    procs = []
    for exp_idx, gpu_id in experiments:
        config = MOSA_BENCHMARKS[exp_idx]
        print(f"Launching {config.name} on GPU {gpu_id}")
        cmd = [
            sys.executable, __file__,
            "--gpu", str(gpu_id),
            "--experiment", str(exp_idx),
            "--output-dir", output_dir,
            "--num-workers", str(num_workers),
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        procs.append((config.name, proc))
    
    return procs


def main():
    parser = argparse.ArgumentParser(description="MoSA Benchmark")
    parser.add_argument("--gpu", type=int, default=0, help="GPU to use")
    parser.add_argument("--gpus", type=str, default=None, help="Comma-separated GPU IDs for parallel runs (e.g. '4,5')")
    parser.add_argument("--output-dir", type=str, default="results/mosa_benchmark", help="Output directory")
    parser.add_argument("--experiment", type=int, default=None, help="Which experiment to run (0=pure, 1=hybrid, 2=pure_4head, 3=hybrid_local)")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--list", action="store_true", help="List available experiments")
    args = parser.parse_args()
    
    if args.list:
        print("\nAvailable MoSA Benchmark Experiments:")
        print("=" * 60)
        for i, cfg in enumerate(MOSA_BENCHMARKS):
            print(f"  [{i}] {cfg.name}: {cfg.description}")
        print()
        return
    
    # Run with --gpus for parallel execution
    if args.gpus:
        gpu_ids = [int(g) for g in args.gpus.split(",")]
        print(f"\n{'='*60}")
        print(f"MoSA Benchmark - Parallel Execution")
        print(f"GPUs: {gpu_ids}")
        print(f"Output: {args.output_dir}")
        print(f"{'='*60}\n")
        
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Launch experiments 0 and 1 (pure and hybrid)
        procs = run_all_primary(gpu_ids, args.output_dir, args.num_workers)
        
        print(f"\nLaunched {len(procs)} experiments. Monitor with:")
        print(f"  tail -f {args.output_dir}/*.log")
        return
    
    # Single experiment mode
    if args.experiment is None:
        parser.error("Please specify --experiment or --gpus")
    
    config = MOSA_BENCHMARKS[args.experiment]
    print(f"\n{'='*60}")
    print(f"Running: {config.name}")
    print(f"Description: {config.description}")
    print(f"GPU: {args.gpu}")
    print(f"Steps: {config.num_steps}")
    print(f"{'='*60}\n")
    
    train_mosa(config, args.gpu, args.output_dir, args.num_workers)


if __name__ == "__main__":
    main()

