#!/usr/bin/env python3
"""
Auto-scheduler for ablation experiments.
Monitors GPU utilization and starts new experiments when GPUs become free.

Usage:
    python scripts/auto_schedule.py --check-interval 300
"""

import subprocess
import time
import sys
import os
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional
import torch.multiprocessing as mp

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

@dataclass
class AblationConfig:
    """Experiment configuration."""
    name: str
    description: str
    dim: int = 512
    num_heads: int = 8
    num_layers: int = 14
    num_hyper_nodes: int = 4
    router_temperature: float = 1.0
    router_type: str = "linear"
    use_local_rope: bool = True
    use_mixed_rope: bool = False
    block_pattern: Optional[str] = None
    num_steps: int = 100000
    batch_size: int = 1
    grad_accum: int = 16
    seq_len: int = 1024
    lr: float = 3e-4
    patience: int = 5
    dataset: str = "pg19"
    eval_seq_lens: Optional[List[int]] = None


# Define remaining experiments
ISOFLOP_EXPERIMENTS = [
    AblationConfig(
        name="isoflop_k4_baseline",
        description="K=4, 14 layers, 1024 seq (baseline for IsoFLOP)",
        num_hyper_nodes=4,
        num_layers=14,
        seq_len=1024,
    ),
    AblationConfig(
        name="isoflop_k6_deeper",
        description="K=6, 21 layers (IsoFLOP: more depth)",
        num_hyper_nodes=6,
        num_layers=21,
    ),
    AblationConfig(
        name="isoflop_k6_wider",
        description="K=6, dim=640 (IsoFLOP: more width)",
        num_hyper_nodes=6,
        num_layers=14,
        dim=640,
    ),
    AblationConfig(
        name="isoflop_k6_longer_ctx",
        description="K=6, seq_len=1536 (same tokens/timeline as K=4@1024)",
        num_hyper_nodes=6,
        num_layers=14,
        seq_len=1536,
    ),
]

EXTRAPOLATION_EXPERIMENTS = [
    AblationConfig(
        name="extrap_baseline_global",
        description="Baseline (all Full, global RoPE) - extrapolation test",
        block_pattern="F" * 14,
        use_local_rope=False,
        seq_len=1024,
        eval_seq_lens=[2048, 4096, 8192],
    ),
    AblationConfig(
        name="extrap_interlaced_local",
        description="Interlaced (FSSFSS, local RoPE) - extrapolation test",
        block_pattern="FSSFSSFSSFSSFF",
        num_hyper_nodes=4,
        use_local_rope=True,
        seq_len=1024,
        eval_seq_lens=[2048, 4096, 8192],
    ),
    AblationConfig(
        name="extrap_interlaced_mixed",
        description="Interlaced (FSSFSS, mixed RoPE) - extrapolation test",
        block_pattern="FSSFSSFSSFSSFF",
        num_hyper_nodes=4,
        use_mixed_rope=True,
        seq_len=1024,
        eval_seq_lens=[2048, 4096, 8192],
    ),
    AblationConfig(
        name="extrap_pure_sparse_local",
        description="Pure Sparse (all S, local RoPE) - extrapolation test",
        block_pattern="S" * 14,
        num_hyper_nodes=4,
        use_local_rope=True,
        seq_len=1024,
        eval_seq_lens=[2048, 4096, 8192],
    ),
]

# Queue of experiments to run (IsoFLOP first, then Extrapolation)
EXPERIMENT_QUEUE = ISOFLOP_EXPERIMENTS + EXTRAPOLATION_EXPERIMENTS


def get_gpu_utilization():
    """Get GPU utilization for each GPU."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used", 
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True
        )
        gpus = {}
        for line in result.stdout.strip().split('\n'):
            parts = line.split(',')
            gpu_id = int(parts[0].strip())
            util = int(parts[1].strip())
            mem = int(parts[2].strip())
            gpus[gpu_id] = {'util': util, 'mem': mem}
        return gpus
    except Exception as e:
        print(f"Error getting GPU info: {e}")
        return {}


def find_idle_gpus(threshold_util=10, threshold_mem=1000):
    """Find GPUs with low utilization and memory usage."""
    gpus = get_gpu_utilization()
    idle = []
    for gpu_id, stats in gpus.items():
        if stats['util'] < threshold_util and stats['mem'] < threshold_mem:
            idle.append(gpu_id)
    return idle


def check_experiment_running(exp_name, output_dir):
    """Check if an experiment is already running or completed."""
    log_pattern = Path(output_dir) / f"{exp_name}_gpu*.log"
    import glob
    logs = glob.glob(str(log_pattern))
    
    for log in logs:
        with open(log, 'r') as f:
            content = f.read()
            if "DONE" in content or "EARLY STOPPING" in content:
                return "completed"
            if "step " in content:
                return "running"
    return "not_started"


def start_experiment(gpu_id, config, output_dir):
    """Start a single experiment on specified GPU."""
    from scripts.run_ablation import train_single_ablation
    
    mp.set_start_method('spawn', force=True)
    result_queue = mp.Queue()
    
    p = mp.Process(
        target=train_single_ablation,
        args=(gpu_id, config, output_dir, result_queue, False, 0)
    )
    p.start()
    return p


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-interval", type=int, default=300, 
                        help="Seconds between checks")
    parser.add_argument("--output-dir", type=str, default="results/ablation_auto",
                        help="Output directory")
    parser.add_argument("--dry-run", action="store_true",
                        help="Don't actually start experiments")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Track which experiments have been started
    started = set()
    processes = {}
    
    print("=" * 60)
    print("AUTO-SCHEDULER FOR ABLATION EXPERIMENTS")
    print("=" * 60)
    print(f"Output: {output_dir}")
    print(f"Check interval: {args.check_interval}s")
    print(f"Experiments in queue: {len(EXPERIMENT_QUEUE)}")
    print("=" * 60)
    
    while True:
        # Find experiments not yet started
        pending = [e for e in EXPERIMENT_QUEUE if e.name not in started]
        
        if not pending:
            print("\n✅ All experiments have been started!")
            break
        
        # Find idle GPUs
        idle_gpus = find_idle_gpus()
        
        if idle_gpus:
            print(f"\n[{time.strftime('%H:%M:%S')}] Found {len(idle_gpus)} idle GPUs: {idle_gpus}")
            
            for gpu_id in idle_gpus:
                if not pending:
                    break
                
                # Get next experiment
                config = pending.pop(0)
                
                # Check if already running elsewhere
                status = check_experiment_running(config.name, str(output_dir))
                if status != "not_started":
                    print(f"  {config.name}: {status}, skipping")
                    started.add(config.name)
                    continue
                
                print(f"  Starting {config.name} on GPU {gpu_id}")
                
                if not args.dry_run:
                    p = start_experiment(gpu_id, config, str(output_dir))
                    processes[config.name] = p
                
                started.add(config.name)
        else:
            print(f"[{time.strftime('%H:%M:%S')}] No idle GPUs. "
                  f"Pending: {len(pending)} experiments. Waiting...")
        
        # Clean up completed processes
        for name, p in list(processes.items()):
            if not p.is_alive():
                print(f"  {name}: Process ended")
                del processes[name]
        
        time.sleep(args.check_interval)
    
    # Wait for all to complete
    print("\nWaiting for all experiments to complete...")
    for name, p in processes.items():
        p.join()
        print(f"  {name}: Done")
    
    print("\n✅ All experiments completed!")


if __name__ == "__main__":
    main()


