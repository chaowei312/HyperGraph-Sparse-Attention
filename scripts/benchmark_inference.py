"""
Benchmark inference speed comparison between baseline and sparse models.

Usage:
    python scripts/benchmark_inference.py --baseline results/arch_comparison_768/baseline_checkpoint.pt \
                                          --sparse results/arch_comparison_768/interlaced_fss_checkpoint.pt
"""
import argparse
import torch
import time
import numpy as np
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from model import CausalLM
from train.architectures import get_model_config


def load_model(checkpoint_path: str, arch_name: str, config: dict, device: torch.device):
    """Load a trained model from checkpoint."""
    model_cfg = get_model_config(
        arch_name=arch_name,
        dim=config['dim'],
        num_heads=config['num_heads'],
        num_hyper_nodes=config['num_hyper_nodes'],
        top_k=config['top_k'],
        max_seq_len=config['max_seq_len'],
    )
    model = CausalLM(model_cfg).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
    model.eval()
    return model


def benchmark_forward(model, seq_len: int, device: torch.device, warmup: int = 3, runs: int = 10):
    """Benchmark forward pass latency."""
    x = torch.randint(0, 1000, (1, seq_len), device=device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
    
    torch.cuda.synchronize()
    
    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(runs):
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(x)
            torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000)  # ms
    
    return np.mean(times), np.std(times)


def run_benchmark(baseline_path: str, sparse_path: str, sparse_arch: str, config: dict, 
                  seq_lengths: list, output_path: str = None):
    """Run full benchmark comparison."""
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    print("Loading models...")
    baseline = load_model(baseline_path, 'baseline', config, device)
    sparse = load_model(sparse_path, sparse_arch, config, device)
    
    print()
    print("=" * 80)
    print("INFERENCE SPEED COMPARISON: Baseline vs Sparse")
    print("=" * 80)
    print(f"Config: {config['dim']} dim, {config['num_heads']} heads")
    print(f"Sparse: K={config['num_hyper_nodes']} timelines, top_k={config['top_k']}")
    print("=" * 80)
    print()
    print(f"{'Seq Len':>10} {'Baseline (ms)':>15} {'Sparse (ms)':>15} {'Speedup':>12} {'Status':>10}")
    print("-" * 80)
    
    results = []
    
    for seq_len in seq_lengths:
        try:
            base_time, base_std = benchmark_forward(baseline, seq_len, device)
            sparse_time, sparse_std = benchmark_forward(sparse, seq_len, device)
            
            speedup = base_time / sparse_time
            
            if speedup >= 1.0:
                status = f"{speedup:.2f}x FASTER"
                emoji = "🚀"
            else:
                status = f"{1/speedup:.2f}x slower"
                emoji = "🐢"
            
            print(f"{seq_len:>10} {base_time:>12.2f}±{base_std:.1f} {sparse_time:>12.2f}±{sparse_std:.1f} {status:>12} {emoji}")
            
            results.append({
                'seq_len': seq_len,
                'baseline_ms': base_time,
                'baseline_std': base_std,
                'sparse_ms': sparse_time,
                'sparse_std': sparse_std,
                'speedup': speedup,
            })
            
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                print(f"{seq_len:>10} {'OOM':>15} {'OOM':>15}")
                torch.cuda.empty_cache()
                break
            raise
    
    # Find crossover point
    crossover = None
    for r in results:
        if r['speedup'] >= 1.0:
            crossover = r['seq_len']
            break
    
    print()
    print("=" * 80)
    print(f"CROSSOVER POINT: ~{crossover} tokens" if crossover else "No crossover found")
    print("=" * 80)
    
    # Save results
    if output_path:
        output = {
            'config': config,
            'results': results,
            'crossover': crossover,
        }
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"Results saved to {output_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark inference speed")
    parser.add_argument("--baseline", type=str, required=True, help="Path to baseline checkpoint")
    parser.add_argument("--sparse", type=str, required=True, help="Path to sparse checkpoint")
    parser.add_argument("--sparse_arch", type=str, default="interlaced_fss", help="Sparse architecture name")
    parser.add_argument("--dim", type=int, default=768, help="Model dimension")
    parser.add_argument("--num_heads", type=int, default=12, help="Number of heads")
    parser.add_argument("--num_hyper_nodes", type=int, default=6, help="Number of timelines (K)")
    parser.add_argument("--top_k", type=int, default=2, help="Top-k routing")
    parser.add_argument("--max_seq_len", type=int, default=20000, help="Max sequence length")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()
    
    config = {
        'dim': args.dim,
        'num_heads': args.num_heads,
        'num_hyper_nodes': args.num_hyper_nodes,
        'top_k': args.top_k,
        'max_seq_len': args.max_seq_len,
    }
    
    seq_lengths = [512, 1024, 2048, 4096, 6144, 8192, 10240, 12288, 14336, 16384]
    
    run_benchmark(
        args.baseline, args.sparse, args.sparse_arch,
        config, seq_lengths, args.output
    )


if __name__ == "__main__":
    main()

