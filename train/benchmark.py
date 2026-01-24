#!/usr/bin/env python3
"""
Benchmark utilities for HyperGraph Sparse Attention.

Runs forward pass and autoregressive decode benchmarks for trained models.
Results are appended to the training JSON files for visualization.

Usage:
    python train/benchmark.py --results_dir results/arch_comparison
    python train/benchmark.py --configs baseline late_full --seq_lengths 4096 6144 8192 12288
"""

import argparse
import json
import gc
import time
import sys
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import torch
import torch.nn.functional as F

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from model import ModelConfig, CausalLM
from data import FastTokenizer, create_dataloaders
from train.architectures import ARCHITECTURES, get_model_config


def load_model_from_checkpoint(checkpoint_path: Path, config: ModelConfig, device: torch.device) -> CausalLM:
    """Load model from checkpoint."""
    model = CausalLM(config).to(device)
    
    if checkpoint_path.exists():
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state['model_state_dict'])
        print(f"  Loaded checkpoint: {checkpoint_path.name}")
    else:
        print(f"  No checkpoint found, using random init")
    
    return model


def benchmark_forward_pass(
    model: CausalLM,
    seq_lengths: List[int],
    device: torch.device,
    n_warmup: int = 3,
    n_runs: int = 10,
) -> Dict:
    """Benchmark forward pass at various sequence lengths."""
    model.eval()
    results = {}
    
    for seq_len in seq_lengths:
        if seq_len > model.config.max_seq_len:
            print(f"    Skipping seq_len={seq_len} (exceeds max_seq_len={model.config.max_seq_len})")
            continue
            
        x = torch.randint(0, model.config.vocab_size, (1, seq_len), device=device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(n_warmup):
                _, _ = model(x)
        
        torch.cuda.synchronize()
        
        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(n_runs):
                torch.cuda.synchronize()
                start = time.perf_counter()
                _, _ = model(x)
                torch.cuda.synchronize()
                end = time.perf_counter()
                times.append((end - start) * 1000)  # ms
        
        results[seq_len] = {
            'forward_ms': float(np.median(times)),
            'forward_std': float(np.std(times)),
            'throughput_tok_per_sec': seq_len / np.median(times) * 1000,
        }
    
    return results


def benchmark_ar_decode(
    model: CausalLM,
    prefill_lens: List[int],
    gen_tokens: int = 100,
    device: torch.device = None,
    n_warmup: int = 2,
    n_runs: int = 3,
) -> Dict:
    """Benchmark autoregressive decode (simulated, no KV-cache)."""
    model.eval()
    results = {}
    
    for prefill_len in prefill_lens:
        total_len = prefill_len + gen_tokens
        if total_len > model.config.max_seq_len:
            print(f"    Skipping prefill={prefill_len} (total {total_len} exceeds max)")
            continue
        
        x = torch.randint(0, model.config.vocab_size, (1, prefill_len), device=device)
        
        # Warmup prefill
        with torch.no_grad():
            for _ in range(n_warmup):
                _, _ = model(x)
        
        torch.cuda.synchronize()
        
        # Benchmark prefill
        prefill_times = []
        with torch.no_grad():
            for _ in range(n_runs):
                torch.cuda.synchronize()
                start = time.perf_counter()
                _, _ = model(x)
                torch.cuda.synchronize()
                end = time.perf_counter()
                prefill_times.append((end - start) * 1000)
        
        # Benchmark decode (simulated - full forward each step, no KV cache)
        decode_times = []
        with torch.no_grad():
            for _ in range(n_runs):
                x_full = torch.randint(0, model.config.vocab_size, (1, total_len), device=device)
                torch.cuda.synchronize()
                start = time.perf_counter()
                
                # Simulate decode: each step processes increasing sequence
                for i in range(gen_tokens):
                    _, _ = model(x_full[:, :prefill_len + i + 1])
                
                torch.cuda.synchronize()
                end = time.perf_counter()
                decode_times.append((end - start) * 1000)
        
        total_decode_ms = float(np.median(decode_times))
        per_token_ms = total_decode_ms / gen_tokens
        
        results[prefill_len] = {
            'prefill_ms': float(np.median(prefill_times)),
            'total_decode_ms': total_decode_ms,
            'per_token_ms': per_token_ms,
            'tokens_per_sec': 1000 / per_token_ms,
            'gen_tokens': gen_tokens,
        }
    
    return results


def evaluate_test_set(
    model: CausalLM,
    test_loader,
    device: torch.device,
    max_batches: int = 100,
) -> Dict:
    """Evaluate model on test set."""
    model.eval()
    
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i >= max_batches:
                break
            
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            
            logits, _ = model(input_ids)
            loss = F.cross_entropy(
                logits.view(-1, model.config.vocab_size),
                labels.view(-1),
                reduction='sum'
            )
            
            total_loss += loss.item()
            total_tokens += labels.numel()
    
    avg_loss = total_loss / total_tokens
    perplexity = float(np.exp(avg_loss))
    
    return {
        'test_loss': avg_loss,
        'test_perplexity': perplexity,
        'num_batches': min(i + 1, max_batches),
        'num_tokens': total_tokens,
    }


def run_benchmarks(
    results_dir: Path,
    configs: List[str],
    seq_lengths: List[int],
    prefill_lens: List[int],
    dim: int = 512,
    num_heads: int = 8,
    num_hyper_nodes: int = 4,
    eval_test: bool = True,
    gpu_id: int = 0,
) -> Dict:
    """Run all benchmarks for specified configurations."""
    
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(gpu_id)}")
    
    # Load test data if needed
    test_loader = None
    if eval_test:
        print("\nLoading test data...")
        tokenizer = FastTokenizer(max_length=1024)
        loaders = create_dataloaders(
            tokenizer,
            dataset_name="gutenberg",
            max_length=1024,
            batch_size=1,
            num_workers=0,
        )
        test_loader = loaders.get("test")
        if test_loader:
            print(f"Test set: {len(test_loader)} batches")
    
    all_benchmark_results = {}
    
    for config_name in configs:
        print(f"\n{'='*60}")
        print(f"Benchmarking: {config_name}")
        print(f"{'='*60}")
        
        # Load existing results
        result_path = results_dir / f"{config_name}.json"
        if result_path.exists():
            with open(result_path, 'r') as f:
                results = json.load(f)
        else:
            results = {'config_name': config_name}
        
        # Create model config
        max_seq_len = max(seq_lengths) if seq_lengths else 4096
        config = get_model_config(
            config_name,
            dim=dim,
            num_heads=num_heads,
            num_hyper_nodes=num_hyper_nodes,
            max_seq_len=max_seq_len,
        )
        
        # Load model
        checkpoint_path = results_dir / f"{config_name}_checkpoint.pt"
        model = load_model_from_checkpoint(checkpoint_path, config, device)
        
        print(f"  Model: {model.num_parameters():,} params")
        
        # Forward pass benchmark
        if seq_lengths:
            print(f"  Running forward pass benchmark...")
            forward_results = benchmark_forward_pass(model, seq_lengths, device)
            results['benchmark_forward'] = forward_results
            
            print(f"    {'Seq Len':<10} {'Time (ms)':<15} {'Throughput':<15}")
            for seq_len, data in forward_results.items():
                print(f"    {seq_len:<10} {data['forward_ms']:<15.2f} {data['throughput_tok_per_sec']:<15,.0f} tok/s")
        
        # AR decode benchmark
        if prefill_lens:
            print(f"  Running AR decode benchmark...")
            ar_results = benchmark_ar_decode(model, prefill_lens, device=device)
            results['benchmark_ar_decode'] = ar_results
            
            print(f"    {'Prefill':<10} {'Prefill (ms)':<15} {'Decode (ms)':<15} {'Tok/s':<10}")
            for prefill, data in ar_results.items():
                print(f"    {prefill:<10} {data['prefill_ms']:<15.2f} {data['total_decode_ms']:<15.2f} {data['tokens_per_sec']:<10.1f}")
        
        # Test evaluation
        if eval_test and test_loader:
            print(f"  Running test evaluation...")
            test_results = evaluate_test_set(model, test_loader, device)
            results['test_loss'] = test_results['test_loss']
            results['test_perplexity'] = test_results['test_perplexity']
            print(f"    Test Loss: {test_results['test_loss']:.4f}")
            print(f"    Test Perplexity: {test_results['test_perplexity']:.2f}")
        
        # Save updated results
        with open(result_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"  Saved: {result_path}")
        
        all_benchmark_results[config_name] = results
        
        # Cleanup
        del model
        gc.collect()
        torch.cuda.empty_cache()
    
    # Update combined results
    combined_path = results_dir / "all_results.json"
    if combined_path.exists():
        with open(combined_path, 'r') as f:
            combined = json.load(f)
    else:
        combined = {}
    
    combined.update(all_benchmark_results)
    
    with open(combined_path, 'w') as f:
        json.dump(combined, f, indent=2)
    print(f"\nUpdated combined results: {combined_path}")
    
    return all_benchmark_results


def main():
    parser = argparse.ArgumentParser(description="Benchmark HyperGraph Models")
    parser.add_argument("--results_dir", type=str, default="results/arch_comparison",
                        help="Directory with training results")
    parser.add_argument("--configs", nargs="+", default=None,
                        help="Configs to benchmark (default: all in results_dir)")
    parser.add_argument("--seq_lengths", nargs="+", type=int, 
                        default=[4096, 6144, 8192, 12288],
                        help="Sequence lengths for forward benchmark")
    parser.add_argument("--prefill_lens", nargs="+", type=int,
                        default=[2048, 4096, 6144],
                        help="Prefill lengths for AR decode benchmark")
    parser.add_argument("--dim", type=int, default=512,
                        help="Model dimension")
    parser.add_argument("--num_heads", type=int, default=8,
                        help="Number of attention heads")
    parser.add_argument("--num_hyper_nodes", type=int, default=4,
                        help="Number of hyper nodes (K)")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU ID to use")
    parser.add_argument("--no_test", action="store_true",
                        help="Skip test set evaluation")
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        sys.exit(1)
    
    # Get configs to benchmark
    if args.configs:
        configs = args.configs
    else:
        # Find all JSON files in results dir
        configs = [p.stem for p in results_dir.glob("*.json") 
                   if p.stem != "all_results" and p.stem in ARCHITECTURES]
    
    if not configs:
        print("No configurations found to benchmark")
        sys.exit(1)
    
    print("=" * 70)
    print("HYPERGRAPH SPARSE ATTENTION BENCHMARK")
    print("=" * 70)
    print(f"Results dir: {results_dir}")
    print(f"Configs: {configs}")
    print(f"Seq lengths: {args.seq_lengths}")
    print(f"Prefill lengths: {args.prefill_lens}")
    print(f"Model: dim={args.dim}, heads={args.num_heads}, K={args.num_hyper_nodes}")
    print("=" * 70)
    
    run_benchmarks(
        results_dir=results_dir,
        configs=configs,
        seq_lengths=args.seq_lengths,
        prefill_lens=args.prefill_lens,
        dim=args.dim,
        num_heads=args.num_heads,
        num_hyper_nodes=args.num_hyper_nodes,
        eval_test=not args.no_test,
        gpu_id=args.gpu,
    )
    
    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()

