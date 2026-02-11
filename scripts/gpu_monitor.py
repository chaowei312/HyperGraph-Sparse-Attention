#!/usr/bin/env python3
"""
GPU Monitor - Automatically starts experiments on idle GPUs.

Usage:
    nohup python scripts/gpu_monitor.py > results/gpu_monitor.log 2>&1 &
"""

import subprocess
import time
import os
import sys
from pathlib import Path

# Experiment queue (name, description, k, layers, dim, seq_len, steps, pattern, use_mixed)
# NOTE: extrap_interlaced_local removed (F layers with local RoPE makes no sense)
# NOTE: extrap_pure_sparse_local removed (can reuse Phase 1 sparse_k4/rope_global_k4 models)
# NOTE: All IsoFLOP use 3.07B tokens for fair comparison
# NOTE: hybrid_rope_freq_explore handled by separate watcher (needs special params)
EXPERIMENT_QUEUE = [
    # IsoFLOP (already started - kept for reference)
    ("isoflop_k6_deeper", "K=6, 21 layers (IsoFLOP deeper)", 6, 21, 512, 1024, 187500, None, False),
    ("isoflop_k6_wider", "K=6, dim=640 (IsoFLOP wider)", 6, 14, 640, 1024, 187500, None, False),
    ("isoflop_k6_longer_ctx", "K=6, seq=1536 (IsoFLOP longer)", 6, 14, 512, 1536, 125000, None, False),
    # Extrapolation (only meaningful configs)
    ("extrap_baseline_global", "Full attn, global RoPE (baseline)", 4, 14, 512, 1024, 100000, "F"*14, False),
    ("extrap_interlaced_mixed", "FSSFSS, F=global S=local (hybrid)", 4, 14, 512, 1024, 100000, "FSSFSSFSSFSSFF", True),
]

OUTPUT_DIR = "results/ablation_auto"
STARTED_FILE = "/tmp/gpu_monitor_started.txt"
CHECK_INTERVAL = 300  # 5 minutes


def get_idle_gpus(util_threshold=10, mem_threshold=1500):  # 1.5GB threshold to avoid race conditions
    """Find GPUs with low utilization and memory."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True
        )
        idle = []
        for line in result.stdout.strip().split('\n'):
            parts = line.split(',')
            gpu_id = int(parts[0].strip())
            util = int(parts[1].strip())
            mem = int(parts[2].strip())
            if util < util_threshold and mem < mem_threshold:
                idle.append(gpu_id)
        return idle
    except Exception as e:
        print(f"Error getting GPU info: {e}")
        return []


def is_started(name):
    """Check if experiment was already started."""
    if not os.path.exists(STARTED_FILE):
        return False
    with open(STARTED_FILE, 'r') as f:
        return name in f.read().split('\n')


def mark_started(name):
    """Mark experiment as started."""
    with open(STARTED_FILE, 'a') as f:
        f.write(name + '\n')


def start_experiment(gpu_id, name, desc, k, layers, dim, seq_len, steps, pattern, use_mixed):
    """Start an experiment on the specified GPU."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Build config string
    pattern_str = f"block_pattern='{pattern}'," if pattern else ""
    mixed_str = "use_mixed_rope=True," if use_mixed else ""
    
    # Create experiment script
    # NOTE: All experiments use τ=1.0 (softer routing performs better)
    script = f'''
import sys
sys.path.insert(0, '/home/lopedg/project/HyperGraph-Sparse-Attention')
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)

from scripts.run_ablation import train_single_ablation, AblationConfig

config = AblationConfig(
    name='{name}',
    description='{desc}',
    num_hyper_nodes={k},
    num_layers={layers},
    dim={dim},
    seq_len={seq_len},
    num_steps={steps},
    patience=5,
    dataset='pg19',
    router_temperature=1.0,  # τ=1.0 performs better than τ=0.5
    eval_seq_lens=[2048, 4096, 8192] if 'extrap' in '{name}' else None,
    {pattern_str}
    {mixed_str}
)

result_queue = mp.Queue()
train_single_ablation(
    gpu_id={gpu_id},
    config=config,
    output_dir='{OUTPUT_DIR}',
    result_queue=result_queue,
    resume=False,
    num_workers=0,
)
'''
    
    # Write script to temp file
    script_path = f"/tmp/run_{name}.py"
    with open(script_path, 'w') as f:
        f.write(script)
    
    # Start process
    log_path = f"{OUTPUT_DIR}/{name}_main.log"
    subprocess.Popen(
        f"nohup python {script_path} > {log_path} 2>&1 &",
        shell=True
    )
    
    mark_started(name)
    print(f"[{time.strftime('%H:%M:%S')}] Started {name} on GPU {gpu_id}")


def main():
    print("=" * 60)
    print("GPU MONITOR - Auto Experiment Scheduler")
    print("=" * 60)
    print(f"Output: {OUTPUT_DIR}")
    print(f"Check interval: {CHECK_INTERVAL}s")
    print(f"Experiments in queue: {len(EXPERIMENT_QUEUE)}")
    print("=" * 60)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    while True:
        # Find pending experiments
        pending = [(name, *rest) for name, *rest in EXPERIMENT_QUEUE if not is_started(name)]
        
        if not pending:
            print(f"\n[{time.strftime('%H:%M:%S')}] All experiments started! Exiting.")
            break
        
        # Find idle GPUs
        idle_gpus = get_idle_gpus()
        
        if idle_gpus:
            print(f"\n[{time.strftime('%H:%M:%S')}] Found {len(idle_gpus)} idle GPU(s): {idle_gpus}")
            print(f"Pending experiments: {len(pending)}")
            
            for gpu_id in idle_gpus:
                if not pending:
                    break
                
                exp = pending.pop(0)
                name, desc, k, layers, dim, seq_len, steps, pattern, use_mixed = exp
                start_experiment(gpu_id, name, desc, k, layers, dim, seq_len, steps, pattern, use_mixed)
                time.sleep(30)  # Wait between starts
        else:
            print(f"[{time.strftime('%H:%M:%S')}] No idle GPUs. Pending: {len(pending)}. Waiting...")
        
        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    main()

