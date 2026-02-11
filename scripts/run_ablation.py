#!/usr/bin/env python3
"""
Ablation Study Runner for HyperGraph Sparse Attention.

Phase 1 - Basic Architecture Ablations:
1. Timeline count: K=4 vs K=6 (pure sparse, 512 dim, 8 heads)
2. Temperature: τ=0.5 vs τ=1.0 (fixed K=6)
3. Router architecture: Linear vs 2-layer MLP
4. Position encoding: Global vs Timeline-local RoPE (K=4)
5. Long context: Baseline vs Interlaced (FSSFSSFSSFSSFF) vs Pure Sparse @ 2048 tokens

Phase 2 - Advanced Ablations:
6. IsoFLOP: K=4 vs K=6 with matched compute budget:
   - Deeper (21 layers), Wider (d=640), or Longer context (1536 seq)
   - Longer context matches tokens/timeline: K=6@1536 = K=4@1024 (256 tokens each)
7. Hybrid RoPE: Full=global vs Sparse=local vs mixed (F=global, S=local)
8. Extrapolation: Train 1024, test 2048/4096/8192 (length generalization)

Usage:
    # Run all Phase 1 experiments on 8 GPUs with PG-19
    python scripts/run_ablation.py --num-gpus 8 --dataset pg19
    
    # Run specific experiments
    python scripts/run_ablation.py --experiments timeline_k long_context
    
    # Run Phase 2 IsoFLOP experiments
    python scripts/run_ablation.py --experiments isoflop --num-gpus 3
    
    # Run extrapolation tests (length generalization)
    python scripts/run_ablation.py --experiments extrapolation --num-gpus 4
    
    # Run hybrid RoPE experiments
    python scripts/run_ablation.py --experiments hybrid_rope --num-gpus 3
    
    # Resume from crashed training (loads checkpoints, continues from best step)
    python scripts/run_ablation.py --experiments timeline_k temperature router position \
        --num-gpus 8 --resume --output-dir results/ablation_phase1
    
    # Check status only
    python scripts/run_ablation.py --check-status
"""

import argparse
import json
import math
import os
import sys
import time
import subprocess
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import torch
import torch.multiprocessing as mp

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from model import ModelConfig, CausalLM
from data import FastTokenizer, create_dataloaders


# =============================================================================
# ABLATION CONFIGURATIONS
# =============================================================================

@dataclass
class AblationConfig:
    """Single ablation experiment configuration."""
    name: str
    description: str
    # Model params
    dim: int = 512
    num_heads: int = 8
    num_layers: int = 14  # Number of transformer layers
    num_hyper_nodes: int = 4  # K timelines
    router_temperature: float = 1.0
    router_type: str = "linear"  # "linear" or "mlp"
    use_local_rope: bool = True  # Timeline-local vs global RoPE
    use_mixed_rope: bool = False  # If True: Full=global RoPE, Sparse=local RoPE
    block_pattern: Optional[str] = None  # None = pure sparse "S"*14, or specify e.g. "FSSFSSFSSFSSFF"
    # Training params
    num_steps: int = 100000  # Max steps (Chinchilla optimal for 85M model)
    batch_size: int = 1
    grad_accum: int = 16
    seq_len: int = 1024
    lr: float = 3e-4
    patience: int = 5  # Early stopping patience (evals without improvement)
    dataset: str = "pg19"  # Dataset: pg19, openwebtext, wikitext-103
    # Evaluation params (for extrapolation tests)
    eval_seq_lens: Optional[List[int]] = None  # If set, evaluate on these lengths too
    # RoPE frequency exploration (for length generalization)
    use_rope_freq_exploration: bool = False  # Random frequencies per timeline during training
    rope_freq_range: tuple = (1.0, 4.0)  # Range of frequency multipliers to explore
    use_confidence_gate: bool = False  # Gated attention (prevents attention sink problem)


# Experiment 1: Timeline Count (K=4 vs K=6)
TIMELINE_EXPERIMENTS = [
    AblationConfig(
        name="sparse_k4",
        description="Pure sparse, K=4 timelines",
        num_hyper_nodes=4,
    ),
    AblationConfig(
        name="sparse_k6",
        description="Pure sparse, K=6 timelines",
        num_hyper_nodes=6,
    ),
]

# Experiment 2: Temperature (τ=0.5 vs τ=1.0, fixed K=6)
TEMPERATURE_EXPERIMENTS = [
    AblationConfig(
        name="temp_0.5_k6",
        description="K=6, temperature=0.5 (sharper routing)",
        num_hyper_nodes=6,
        router_temperature=0.5,
    ),
    AblationConfig(
        name="temp_1.0_k6",
        description="K=6, temperature=1.0 (baseline)",
        num_hyper_nodes=6,
        router_temperature=1.0,
    ),
]

# Experiment 3: Router Architecture (Linear vs MLP)
ROUTER_EXPERIMENTS = [
    AblationConfig(
        name="router_linear",
        description="Linear router (baseline)",
        num_hyper_nodes=6,
        router_type="linear",
    ),
    AblationConfig(
        name="router_mlp",
        description="2-layer MLP router (feature extraction + head)",
        num_hyper_nodes=6,
        router_type="mlp",
    ),
]

# Experiment 4: Position Encoding (K=4)
POSITION_EXPERIMENTS = [
    AblationConfig(
        name="rope_local_k4",
        description="Timeline-local RoPE (positions reset per timeline)",
        num_hyper_nodes=4,
        use_local_rope=True,
    ),
    AblationConfig(
        name="rope_global_k4",
        description="Global RoPE (original positions preserved)",
        num_hyper_nodes=4,
        use_local_rope=False,
    ),
]

# Experiment 5: Long Context Architecture Comparison (2048 tokens)
LONG_CONTEXT_EXPERIMENTS = [
    AblationConfig(
        name="longctx_baseline",
        description="Baseline (all Full attention) @ 2048 tokens",
        block_pattern="F" * 14,  # FFFFFFFFFFFFFF
        seq_len=2048,
    ),
    AblationConfig(
        name="longctx_interlaced",
        description="Interlaced FSS (FSSFSSFSSFSSFF) @ 2048 tokens",
        block_pattern="FSSFSSFSSFSSFF",
        num_hyper_nodes=4,
        seq_len=2048,
    ),
    AblationConfig(
        name="longctx_pure_sparse",
        description="Pure Sparse (all S) @ 2048 tokens",
        block_pattern="S" * 14,  # SSSSSSSSSSSSSS
        num_hyper_nodes=4,
        seq_len=2048,
    ),
]

# =============================================================================
# PHASE 2: ADVANCED ABLATIONS (IsoFLOP, Hybrid RoPE, Extrapolation)
# =============================================================================

# Experiment 6: IsoFLOP Comparison (match compute budget, not params)
# K=6 has ~33% less attention FLOPs than K=4 at same depth
# To match FLOPs: either increase depth, width, or context length
# NOTE: All trained on 3.07B tokens for fair comparison
# NOTE: All use τ=1.0 (softer routing performs better than τ=0.5)
# NOTE: Iso-Tokens comparison: all train on 2.03B tokens for fair comparison
ISOFLOP_EXPERIMENTS = [
    AblationConfig(
        name="isoflop_k4_baseline",
        description="K=4, 14 layers, 1024 seq (baseline for IsoFLOP comparison)",
        num_hyper_nodes=4,
        num_layers=14,
        seq_len=1024,
        num_steps=124000,  # 2.03B tokens / (16 × 1024) = 124K steps
        router_temperature=1.0,
    ),
    AblationConfig(
        name="isoflop_k6_deeper",
        description="K=6, 21 layers (IsoFLOP: more depth to match K=4 compute)",
        num_hyper_nodes=6,
        num_layers=21,  # 14 * (6/4) = 21 layers to match attention FLOPs
        seq_len=1024,
        num_steps=124000,  # 2.03B tokens / (16 × 1024) = 124K steps
        router_temperature=1.0,
    ),
    AblationConfig(
        name="isoflop_k6_wider",
        description="K=6, dim=640 (IsoFLOP: more width to match K=4 compute)",
        num_hyper_nodes=6,
        num_layers=14,
        dim=640,  # sqrt(1.5) * 512 ≈ 627, round to 640
        seq_len=1024,
        num_steps=124000,  # 2.03B tokens / (16 × 1024) = 124K steps
        router_temperature=1.0,
    ),
    AblationConfig(
        name="isoflop_k6_longer_ctx",
        description="K=6, seq_len=1536 (same tokens/timeline as K=4@1024: 256)",
        num_hyper_nodes=6,
        num_layers=14,
        seq_len=1536,  # 1536/6 = 256 tokens/timeline = 1024/4
        num_steps=83000,  # 2.03B tokens / (16 × 1536) = 83K steps
        router_temperature=1.0,
    ),
]

# Experiment 7: Hybrid RoPE (Full=global, Sparse=local)
# Theory: Full attention needs global positions for full-sequence context,
#         Sparse attention benefits from local positions for extrapolation
# NOTE: All use τ=1.0 (softer routing performs better than τ=0.5)
# NOTE: All ~70M params, Chinchilla-optimal = 85K steps
HYBRID_ROPE_EXPERIMENTS = [
    AblationConfig(
        name="hybrid_rope_mixed",
        description="Hybrid: Full=global RoPE, Sparse=local RoPE",
        block_pattern="FSSFSSFSSFSSFF",
        num_hyper_nodes=4,
        use_mixed_rope=True,  # F layers get global, S layers get local
        seq_len=1024,
        num_steps=85000,  # 70M params × 20 = 1.4B tokens
        eval_seq_lens=[2048, 4096, 8192],  # Test extrapolation (includes 8K from dropped extrap_interlaced_mixed)
        router_temperature=1.0,
    ),
    AblationConfig(
        name="hybrid_rope_freq_explore",
        description="Hybrid + RoPE freq exploration per timeline (1-4x range)",
        block_pattern="FSSFSSFSSFSSFF",
        num_hyper_nodes=4,
        use_mixed_rope=True,
        use_rope_freq_exploration=True,  # Random frequencies per timeline
        rope_freq_range=(1.0, 4.0),  # Explore positions up to 4× training
        seq_len=1024,
        num_steps=85000,  # 70M params × 20 = 1.4B tokens
        eval_seq_lens=[2048, 4096, 8192],  # Test if freq exploration helps
        router_temperature=1.0,
    ),
    AblationConfig(
        name="hybrid_confidence_gate",
        description="Hybrid + Confidence Gate (prevents attention sink)",
        block_pattern="FSSFSSFSSFSSFF",
        num_hyper_nodes=4,
        use_mixed_rope=True,
        use_confidence_gate=True,  # Enable gated attention
        seq_len=1024,
        num_steps=85000,
        eval_seq_lens=[2048, 4096, 8192],
        router_temperature=1.0,
    ),
    AblationConfig(
        name="hybrid_interlaced_global",
        description="Interlaced (FSSFSS) with ALL global RoPE - fair comparison with baseline",
        block_pattern="FSSFSSFSSFSSFF",
        num_hyper_nodes=4,
        use_local_rope=False,  # All layers use global RoPE
        use_mixed_rope=False,  # No mixed mode
        seq_len=1024,
        num_steps=85000,  # 70M params × 20 = 1.4B tokens
        eval_seq_lens=[2048, 4096, 8192],  # Same as extrap_baseline_global
        router_temperature=1.0,
    ),
]

# Experiment 8: Long-Context Extrapolation Test
# Train on 1024 tokens, evaluate on 2048, 4096, 8192
# Tests whether mixed RoPE (F=global, S=local) enables length generalization
# 
# NOTE: Removed extrap_interlaced_local (F layers with local RoPE makes no sense)
# NOTE: Removed extrap_pure_sparse_local (reuse Phase 1 models for pure sparse eval)
# NOTE: All use τ=1.0 (softer routing performs better than τ=0.5)
# NOTE: All ~70M params, Chinchilla-optimal = 85K steps
EXTRAPOLATION_EXPERIMENTS = [
    AblationConfig(
        name="extrap_baseline_global",
        description="Baseline (all Full, global RoPE) - should degrade at long context",
        block_pattern="F" * 14,
        use_local_rope=False,
        seq_len=1024,
        num_steps=85000,  # 70M params × 20 = 1.4B tokens
        eval_seq_lens=[2048, 4096, 8192],
        router_temperature=1.0,
    ),
    # NOTE: extrap_interlaced_mixed removed - identical to hybrid_rope_mixed (only diff was eval at 8K)
]

ALL_EXPERIMENTS = {
    # Phase 1: Basic architecture ablations
    "timeline_k": TIMELINE_EXPERIMENTS,
    "temperature": TEMPERATURE_EXPERIMENTS,
    "router": ROUTER_EXPERIMENTS,
    "position": POSITION_EXPERIMENTS,
    "long_context": LONG_CONTEXT_EXPERIMENTS,
    # Phase 2: Advanced ablations
    "isoflop": ISOFLOP_EXPERIMENTS,
    "hybrid_rope": HYBRID_ROPE_EXPERIMENTS,
    "extrapolation": EXTRAPOLATION_EXPERIMENTS,
}


# =============================================================================
# DATA CHECK
# =============================================================================

def check_dataset_status(dataset: str = "openwebtext") -> Dict:
    """Check dataset download status."""
    cache_dir = Path.home() / ".cache/huggingface/datasets"
    
    # Dataset-specific checks
    if dataset == "pg19":
        pg19_cache = cache_dir / "emozilla___pg19"
        data_ready = pg19_cache.exists()
        target_size_gb = 11
        dataset_name = "PG-19"
    elif dataset == "openwebtext":
        data_dir = Path("/home/lopedg/project/data/openwebtext")
        owt_cache = cache_dir / "Skylion007___openwebtext"
        data_ready = (data_dir / "train").exists() or (owt_cache.exists() and 
                     sum(f.stat().st_size for f in owt_cache.rglob("*.arrow") if f.exists()) > 30e9)
        target_size_gb = 38
        dataset_name = "OpenWebText"
    elif dataset in ["wikitext-103", "gutenberg"]:
        data_ready = True  # Assume available (small datasets)
        target_size_gb = 1
        dataset_name = dataset
    else:
        data_ready = True
        target_size_gb = 1
        dataset_name = dataset
    
    # Check cache size
    cache_size_gb = 0
    if cache_dir.exists():
        try:
            result = subprocess.run(
                ["du", "-sb", str(cache_dir)],
                capture_output=True, text=True
            )
            cache_size_bytes = int(result.stdout.split()[0])
            cache_size_gb = cache_size_bytes / (1024**3)
        except:
            pass
    
    return {
        "dataset": dataset_name,
        "cache_size_gb": cache_size_gb,
        "target_size_gb": target_size_gb,
        "data_ready": data_ready,
    }


def wait_for_data(dataset: str = "openwebtext", check_interval: int = 60):
    """Wait for dataset download to complete."""
    print(f"Waiting for {dataset} download to complete...")
    print("=" * 60)
    
    while True:
        status = check_dataset_status(dataset)
        
        print(f"[{time.strftime('%H:%M:%S')}] "
              f"{status['dataset']}: "
              f"({status['cache_size_gb']:.1f}GB cached) "
              f"| Ready: {status['data_ready']}")
        
        if status['data_ready']:
            print(f"\n✅ {status['dataset']} is ready!")
            return True
        
        time.sleep(check_interval)


# =============================================================================
# TRAINING
# =============================================================================

def train_single_ablation(
    gpu_id: int,
    config: AblationConfig,
    output_dir: str,
    result_queue: mp.Queue,
    resume: bool = False,
    num_workers: int = 0,  # Default to 0 for stability
):
    """Train a single ablation configuration on a specific GPU."""
    import torch.nn.functional as F
    import sys
    import re
    
    # Set up per-process logging to file
    log_file = Path(output_dir) / f"{config.name}_gpu{gpu_id}.log"
    
    def log(msg):
        """Log to both file and stdout with flush."""
        timestamp = time.strftime("%H:%M:%S")
        line = f"[{timestamp}][GPU {gpu_id}] {msg}"
        print(line, flush=True)
        with open(log_file, 'a') as f:
            f.write(line + "\n")
    
    # Check for resume
    start_step = 0
    best_val_loss = float('inf')
    best_step = 0
    checkpoint_path = Path(output_dir) / f"{config.name}_best.pt"
    
    if resume and checkpoint_path.exists() and log_file.exists():
        # Parse log file for best step and val_loss
        try:
            log_content = log_file.read_text()
            # Find last "NEW BEST" line
            best_matches = re.findall(r'step (\d+).*\n.*val_loss: ([\d.]+) ★ NEW BEST', log_content)
            if best_matches:
                best_step = int(best_matches[-1][0])
                best_val_loss = float(best_matches[-1][1])
                start_step = best_step
                log(f"RESUMING from step {start_step}, best_val_loss={best_val_loss:.4f}")
        except Exception as e:
            log(f"Warning: Could not parse log for resume: {e}")
    
    try:
        torch.cuda.set_device(gpu_id)
        device = torch.device(f'cuda:{gpu_id}')
        
        log(f"Starting {config.name}: {config.description}")
        
        # Determine block pattern
        if config.block_pattern is not None:
            block_pattern = list(config.block_pattern)
        else:
            block_pattern = ["S"] * config.num_layers  # Default: pure sparse
        
        # Ensure block pattern matches num_layers
        if len(block_pattern) != config.num_layers:
            # Extend or truncate pattern to match num_layers
            if len(block_pattern) < config.num_layers:
                # Repeat pattern to fill
                pattern_cycle = block_pattern * ((config.num_layers // len(block_pattern)) + 1)
                block_pattern = pattern_cycle[:config.num_layers]
            else:
                block_pattern = block_pattern[:config.num_layers]
        
        # Create model config
        model_config = ModelConfig(
            dim=config.dim,
            num_heads=config.num_heads,
            num_hyper_nodes=config.num_hyper_nodes,
            router_temperature=config.router_temperature,
            router_type=config.router_type,
            use_local_rope=config.use_local_rope,
            use_mixed_rope=config.use_mixed_rope,
            use_rope_freq_exploration=config.use_rope_freq_exploration,
            rope_freq_range=config.rope_freq_range,
            use_confidence_gate=config.use_confidence_gate,
            block_pattern=block_pattern,
            max_seq_len=config.seq_len * 8,  # Allow longer sequences for extrapolation tests
        )
        
        # Log ablation params
        pattern_str = ''.join(block_pattern)
        ablation_params = {
            "router_type": config.router_type,
            "use_local_rope": config.use_local_rope,
            "use_mixed_rope": config.use_mixed_rope,
            "block_pattern": pattern_str,
            "num_layers": config.num_layers,
            "seq_len": config.seq_len,
            "eval_seq_lens": config.eval_seq_lens,
        }
        log(f"{config.name}: pattern={pattern_str}, layers={config.num_layers}, seq_len={config.seq_len}")
        
        # Load data (use num_workers=0 for stability with multiprocessing)
        tokenizer = FastTokenizer(max_length=config.seq_len)
        loaders = create_dataloaders(
            tokenizer,
            dataset_name=config.dataset,
            max_length=config.seq_len,
            batch_size=config.batch_size,
            num_workers=num_workers,
        )
        
        # Create model
        model = CausalLM(model_config).to(device)
        
        # Load checkpoint if resuming
        if resume and checkpoint_path.exists():
            log(f"Loading checkpoint from {checkpoint_path}")
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
        
        # Cosine annealing scheduler with warmup
        warmup_steps = min(2000, config.num_steps // 10)  # 10% warmup, max 2000 steps
        
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                # Linear warmup
                return float(current_step) / float(max(1, warmup_steps))
            else:
                # Cosine decay to 10% of initial LR
                progress = float(current_step - warmup_steps) / float(max(1, config.num_steps - warmup_steps))
                return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        log(f"{config.name}: {model.num_parameters():,} params (warmup={warmup_steps}, cosine decay)")
        
        # Training history
        history = {
            "config": asdict(config),
            "ablation_params": ablation_params,
            "gpu_id": gpu_id,
            "steps": [],
            "train_loss": [],
            "val_loss": [],
        }
        
        # Training loop with early stopping
        model.train()
        step = start_step  # Resume from checkpoint step
        micro_step = start_step * config.grad_accum  # Adjust micro_step
        accum_loss = 0.0
        running_loss = 0.0
        # best_val_loss and best_step already set from resume logic above
        patience_counter = 0
        patience = getattr(config, 'patience', 5)  # Default patience=5
        early_stopped = False
        start_time = time.time()
        
        if start_step > 0:
            log(f"{config.name}: Resuming training from step {start_step}")
        
        optimizer.zero_grad()
        
        for epoch in range(500):  # More epochs for longer training
            for batch in loaders["train"]:
                if step >= config.num_steps or early_stopped:
                    break
                
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                
                logits, aux_loss = model(input_ids)
                ce_loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1)
                )
                loss = ce_loss + 0.01 * aux_loss
                scaled_loss = loss / config.grad_accum
                scaled_loss.backward()
                
                accum_loss += loss.item()
                micro_step += 1
                
                if micro_step % config.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()  # Update learning rate
                    optimizer.zero_grad()
                    step += 1
                    
                    running_loss += accum_loss / config.grad_accum
                    accum_loss = 0.0
                    
                    # Log every 500 steps
                    if step % 500 == 0:
                        avg_loss = running_loss / 500
                        elapsed = time.time() - start_time
                        current_lr = scheduler.get_last_lr()[0]
                        log(f"{config.name} step {step} | loss: {avg_loss:.4f} | lr: {current_lr:.2e} | time: {elapsed:.1f}s")
                        history["steps"].append(step)
                        history["train_loss"].append(avg_loss)
                        running_loss = 0.0
                    
                    # Validate every 2000 steps
                    if step % 2000 == 0:
                        model.eval()
                        val_losses = []
                        with torch.no_grad():
                            for i, b in enumerate(loaders["val"]):
                                if i >= 50:  # Limit eval batches
                                    break
                                v_logits, _ = model(b["input_ids"].to(device))
                                v_loss = F.cross_entropy(
                                    v_logits.view(-1, model_config.vocab_size),
                                    b["labels"].to(device).view(-1)
                                )
                                val_losses.append(v_loss.item())
                        val_loss = sum(val_losses) / len(val_losses)
                        history["val_loss"].append(val_loss)
                        
                        # Early stopping check
                        if val_loss < best_val_loss - 0.001:  # min_delta=0.001
                            best_val_loss = val_loss
                            best_step = step
                            patience_counter = 0
                            # Save best checkpoint
                            ckpt_path = Path(output_dir) / f"{config.name}_best.pt"
                            torch.save(model.state_dict(), ckpt_path)
                            log(f"{config.name} >>> val_loss: {val_loss:.4f} ★ NEW BEST")
                        else:
                            patience_counter += 1
                            log(f"{config.name} >>> val_loss: {val_loss:.4f} (best: {best_val_loss:.4f}, patience: {patience_counter}/{patience})")
                            if patience_counter >= patience:
                                log(f"{config.name} >>> EARLY STOPPING at step {step}")
                                early_stopped = True
                        
                        model.train()
            
            if step >= config.num_steps or early_stopped:
                break
        
        # Final results
        history["total_time"] = time.time() - start_time
        history["best_val_loss"] = best_val_loss
        history["best_step"] = best_step
        history["final_step"] = step
        history["early_stopped"] = early_stopped
        
        # Extrapolation evaluation (if specified)
        if config.eval_seq_lens:
            log(f"{config.name}: Running extrapolation evaluation...")
            
            # Load best checkpoint
            best_ckpt = Path(output_dir) / f"{config.name}_best.pt"
            if best_ckpt.exists():
                model.load_state_dict(torch.load(best_ckpt, map_location=device))
            
            model.eval()
            extrapolation_results = {}
            
            for eval_len in config.eval_seq_lens:
                try:
                    # Create dataloader for longer sequences
                    eval_tokenizer = FastTokenizer(max_length=eval_len)
                    eval_loaders = create_dataloaders(
                        eval_tokenizer,
                        dataset_name=config.dataset,
                        max_length=eval_len,
                        batch_size=1,  # Longer sequences need smaller batch
                        num_workers=2,
                    )
                    
                    eval_losses = []
                    with torch.no_grad():
                        for i, b in enumerate(eval_loaders["val"]):
                            if i >= 25:  # Fewer samples for longer sequences
                                break
                            try:
                                v_logits, _ = model(b["input_ids"].to(device))
                                v_loss = F.cross_entropy(
                                    v_logits.view(-1, model_config.vocab_size),
                                    b["labels"].to(device).view(-1)
                                )
                                eval_losses.append(v_loss.item())
                            except RuntimeError as e:
                                if "out of memory" in str(e):
                                    torch.cuda.empty_cache()
                                    log(f"{config.name}: OOM at seq_len={eval_len}, skipping")
                                    break
                                raise
                    
                    if eval_losses:
                        avg_loss = sum(eval_losses) / len(eval_losses)
                        extrapolation_results[eval_len] = avg_loss
                        log(f"{config.name}: seq_len={eval_len} → loss={avg_loss:.4f}")
                    
                except Exception as e:
                    log(f"{config.name}: Failed eval at seq_len={eval_len}: {e}")
            
            history["extrapolation"] = extrapolation_results
        
        # Save results
        result_path = Path(output_dir) / f"{config.name}_results.json"
        with open(result_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        log(f"{config.name} DONE | best_val: {best_val_loss:.4f} @ step {best_step} | time: {history['total_time']:.1f}s")
        
        # Log extrapolation summary if available
        if config.eval_seq_lens and "extrapolation" in history:
            extrap_str = ", ".join([f"{k}:{v:.3f}" for k, v in history["extrapolation"].items()])
            log(f"{config.name} EXTRAPOLATION: {extrap_str}")
        
        result_queue.put((config.name, history, None))
        
    except Exception as e:
        import traceback
        error = traceback.format_exc()
        log(f"{config.name} FAILED: {e}\n{error}")
        result_queue.put((config.name, None, error))


def run_ablation_experiments(
    experiments: List[AblationConfig],
    output_dir: str,
    num_gpus: int = 8,
    resume: bool = False,
    num_workers: int = 0,
):
    """Run ablation experiments distributed across GPUs."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    mp.set_start_method('spawn', force=True)
    
    print("=" * 70)
    print("ABLATION STUDY")
    print("=" * 70)
    print(f"Experiments: {len(experiments)}")
    print(f"GPUs: {num_gpus}")
    print(f"Output: {output_dir}")
    print("=" * 70)
    
    for cfg in experiments:
        print(f"  {cfg.name}: {cfg.description}")
    print()
    
    # Run in batches
    all_results = {}
    n_batches = (len(experiments) + num_gpus - 1) // num_gpus
    
    for batch_idx in range(n_batches):
        batch_start = batch_idx * num_gpus
        batch_end = min(batch_start + num_gpus, len(experiments))
        batch_configs = experiments[batch_start:batch_end]
        
        print(f"\n--- Batch {batch_idx + 1}/{n_batches} ---")
        
        result_queue = mp.Queue()
        processes = []
        
        for i, config in enumerate(batch_configs):
            gpu_id = i % num_gpus
            p = mp.Process(
                target=train_single_ablation,
                args=(gpu_id, config, str(output_path), result_queue, resume, num_workers)
            )
            p.start()
            processes.append(p)
        
        # Collect results
        for _ in range(len(batch_configs)):
            name, history, error = result_queue.get()
            if error:
                print(f"❌ {name} failed")
            else:
                all_results[name] = history
        
        for p in processes:
            p.join()
    
    # Save combined results
    combined_path = output_path / "ablation_results.json"
    with open(combined_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\n" + "=" * 70)
    print("ABLATION RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Name':<20} {'Val Loss':<12} {'Time (s)':<10}")
    print("-" * 50)
    
    for name, hist in sorted(all_results.items(), 
                             key=lambda x: x[1].get('best_val_loss', 999)):
        val = hist.get('best_val_loss', 'N/A')
        t = hist.get('total_time', 0)
        val_str = f"{val:.4f}" if isinstance(val, float) else val
        print(f"{name:<20} {val_str:<12} {t:<10.1f}")
    
    return all_results


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="HyperGraph Ablation Study")
    parser.add_argument("--experiments", nargs="+", 
                        choices=list(ALL_EXPERIMENTS.keys()) + ["all"],
                        default=["all"],
                        help="Which experiments to run")
    parser.add_argument("--output-dir", type=str, 
                        default="results/ablation",
                        help="Output directory")
    parser.add_argument("--num-gpus", type=int, default=8,
                        help="Number of GPUs to use")
    parser.add_argument("--wait-for-data", action="store_true",
                        help="Wait for OpenWebText download to complete")
    parser.add_argument("--check-status", action="store_true",
                        help="Only check data status, don't run experiments")
    parser.add_argument("--num-steps", type=int, default=100000,
                        help="Max training steps per experiment (default: 100K for Chinchilla)")
    parser.add_argument("--patience", type=int, default=5,
                        help="Early stopping patience (evals without improvement)")
    parser.add_argument("--dataset", type=str, default="pg19",
                        choices=["pg19", "openwebtext", "wikitext-103", "gutenberg"],
                        help="Dataset to use (default: pg19)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from existing checkpoints")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="DataLoader num_workers (0=stable, >0=faster but may crash)")
    args = parser.parse_args()
    
    # Check data status
    status = check_dataset_status(args.dataset)
    print(f"\n{status['dataset']} Status:")
    print(f"  Cache: {status['cache_size_gb']:.1f} GB")
    print(f"  Ready: {status['data_ready']}")
    
    if args.check_status:
        return
    
    # Wait for data if requested
    if args.wait_for_data and not status['data_ready']:
        if not wait_for_data(args.dataset):
            print("Data not ready. Exiting.")
            return
    elif not status['data_ready']:
        print(f"\n⚠️  {status['dataset']} not ready! Use --wait-for-data to wait.")
        return
    
    # Collect experiments to run
    experiments = []
    exp_names = args.experiments
    if "all" in exp_names:
        exp_names = list(ALL_EXPERIMENTS.keys())
    
    for name in exp_names:
        for cfg in ALL_EXPERIMENTS[name]:
            cfg.num_steps = args.num_steps
            cfg.patience = args.patience
            cfg.dataset = args.dataset
            experiments.append(cfg)
    
    print(f"\nRunning {len(experiments)} ablation experiments...")
    
    # Run experiments
    results = run_ablation_experiments(
        experiments=experiments,
        output_dir=args.output_dir,
        num_gpus=args.num_gpus,
        resume=args.resume,
        num_workers=args.num_workers,
    )
    
    print(f"\nResults saved to {args.output_dir}/ablation_results.json")


if __name__ == "__main__":
    main()

