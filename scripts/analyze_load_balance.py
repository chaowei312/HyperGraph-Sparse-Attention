"""
Analyze load balance across timelines for sparse attention models.

Uses actual WikiText-103 test sequences for representative analysis.

Usage:
    # Single model
    python scripts/analyze_load_balance.py --checkpoint results/arch_comparison_768/interlaced_fss_checkpoint.pt \
                                           --arch interlaced_fss

    # All models in parallel
    python scripts/analyze_load_balance.py --all --results_dir results/arch_comparison_768 --gpus 0 2 6 7
"""
import argparse
import torch
import numpy as np
import json
from pathlib import Path
import sys
import multiprocessing as mp
from typing import List, Dict, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from model import CausalLM
from train.architectures import get_model_config, ARCHITECTURES
from data import create_dataloaders, FastTokenizer


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
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
    model.eval()
    return model


def analyze_load_balance_batch(model, dataloader, device: torch.device, max_batches: int = 100):
    """Analyze load balance across all sparse layers using real test data."""
    
    # Find sparse layers
    sparse_layers = [(i, l) for i, l in enumerate(model.layers) 
                     if hasattr(l, 'attention') and hasattr(l.attention, 'node_router')]
    
    if not sparse_layers:
        return []
    
    # Get K and H from first sparse layer
    K = sparse_layers[0][1].attention.num_hyper_nodes
    H = sparse_layers[0][1].attention.num_heads
    
    # Accumulate counts across batches
    layer_counts = {layer_idx: np.zeros((H, K)) for layer_idx, _ in sparse_layers}
    total_tokens = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= max_batches:
                break
            
            input_ids = batch['input_ids'].to(device)
            seq_len = input_ids.size(1)
            total_tokens += seq_len
            
            x = model.embedding(input_ids)
            
            for layer_idx, layer in enumerate(model.layers):
                if layer_idx in [idx for idx, _ in sparse_layers]:
                    attn = layer.attention
                    
                    # Get routing decisions
                    h = layer.attn_norm(x) if hasattr(layer, 'attn_norm') else x
                    node_logits = attn._compute_node_logits(h)
                    primary = node_logits.argmax(dim=-1)  # (batch, heads, seq)
                    
                    # Count per timeline per head
                    for head in range(H):
                        counts = torch.bincount(primary[0, head], minlength=K).cpu().numpy()
                        layer_counts[layer_idx][head] += counts
                
                # Forward through layer
                out = layer(x)
                x = out[0] if isinstance(out, tuple) else out
    
    # Compute statistics
    results = []
    for sparse_idx, (layer_idx, _) in enumerate(sparse_layers):
        counts = layer_counts[layer_idx]  # (H, K)
        avg_counts = counts.mean(axis=0)
        total = avg_counts.sum()
        avg_pct = avg_counts / total * 100
        
        ideal = 100 / K
        max_load = avg_pct.max()
        min_load = avg_pct.min()
        std_load = avg_pct.std()
        imbalance = max_load / min_load if min_load > 0 else float('inf')
        
        results.append({
            'layer_idx': layer_idx,
            'sparse_idx': sparse_idx,
            'distribution': avg_pct.tolist(),
            'max_load': float(max_load),
            'min_load': float(min_load),
            'imbalance': float(imbalance),
            'std': float(std_load),
            'ideal': float(ideal),
            'total_tokens': total_tokens,
        })
    
    return results


def print_results(results, K, arch_name: str = ""):
    """Print load balance analysis results."""
    print("=" * 75)
    print(f"LOAD BALANCE ANALYSIS: {arch_name} ({K} Timelines)")
    print("=" * 75)
    
    if not results:
        print("No sparse layers found.")
        return
    
    print(f"Analyzed {results[0].get('total_tokens', 'N/A')} tokens from WikiText-103 test set\n")
    
    for r in results:
        print(f"Layer {r['layer_idx']} (Sparse Layer #{r['sparse_idx'] + 1})")
        print("-" * 75)
        print("Timeline Distribution:")
        
        for t, pct in enumerate(r['distribution']):
            deviation = pct - r['ideal']
            sign = '+' if deviation >= 0 else ''
            bars = '█' * int(pct / 5)
            print(f"  T{t}: {pct:5.1f}% {bars:20s} ({sign}{deviation:.1f}%)")
        
        print(f"\nMetrics:")
        print(f"  Max load: {r['max_load']:.1f}%")
        print(f"  Min load: {r['min_load']:.1f}%")
        print(f"  Imbalance: {r['imbalance']:.2f}x")
        print(f"  Std: {r['std']:.2f}%")
        
        if r['imbalance'] < 1.5:
            print("  Status: ✅ Good balance")
        elif r['imbalance'] < 2.0:
            print("  Status: ⚠️ Moderate imbalance")
        else:
            print("  Status: ❌ High imbalance")
        print()
    
    print("=" * 75)
    avg_imbalance = np.mean([r['imbalance'] for r in results])
    print(f"OVERALL: Average imbalance = {avg_imbalance:.2f}x")
    print("=" * 75)
    
    return avg_imbalance


def analyze_single_model(
    checkpoint_path: str,
    arch_name: str,
    config: dict,
    gpu_id: int,
    max_batches: int = 100,
    num_workers: int = 0,
) -> Dict:
    """Analyze a single model on a specific GPU."""
    device = torch.device(f'cuda:{gpu_id}')
    torch.cuda.set_device(device)
    
    print(f"[GPU {gpu_id}] Loading {arch_name}...")
    model = load_model(checkpoint_path, arch_name, config, device)
    
    # Load WikiText-103 test data
    tokenizer = FastTokenizer(max_length=1024)
    loaders = create_dataloaders(
        tokenizer,
        dataset_name='wikitext-103',
        max_length=1024,
        batch_size=1,
        num_workers=num_workers,  # 0 for parallel mode to avoid daemon issues
    )
    
    print(f"[GPU {gpu_id}] Analyzing {arch_name} on {max_batches} test batches...")
    results = analyze_load_balance_batch(model, loaders['test'], device, max_batches)
    
    avg_imbalance = np.mean([r['imbalance'] for r in results]) if results else 0
    print(f"[GPU {gpu_id}] {arch_name}: avg imbalance = {avg_imbalance:.2f}x")
    
    return {
        'arch': arch_name,
        'results': results,
        'avg_imbalance': float(avg_imbalance),
    }


def analyze_all_models_parallel(
    results_dir: str,
    gpus: List[int],
    config: dict,
    max_batches: int = 100,
) -> Dict[str, Dict]:
    """Analyze all models in parallel across GPUs."""
    results_path = Path(results_dir)
    
    # Find all checkpoints
    checkpoints = list(results_path.glob('*_checkpoint.pt'))
    models = []
    for ckpt in checkpoints:
        arch_name = ckpt.stem.replace('_checkpoint', '')
        if arch_name == 'baseline':
            continue  # Skip baseline (no sparse layers)
        if arch_name in ARCHITECTURES:
            models.append((str(ckpt), arch_name))
    
    print(f"Found {len(models)} sparse models to analyze")
    print(f"Using GPUs: {gpus}")
    print()
    
    # Run in parallel
    all_results = {}
    
    if len(gpus) == 1:
        # Sequential on single GPU
        for ckpt_path, arch_name in models:
            result = analyze_single_model(ckpt_path, arch_name, config, gpus[0], max_batches, num_workers=2)
            all_results[arch_name] = result
    else:
        # Parallel across GPUs
        mp.set_start_method('spawn', force=True)
        
        # Process in batches matching GPU count
        for batch_start in range(0, len(models), len(gpus)):
            batch = models[batch_start:batch_start + len(gpus)]
            
            with mp.Pool(len(batch)) as pool:
                args = [
                    (ckpt_path, arch_name, config, gpus[i % len(gpus)], max_batches, 0)  # num_workers=0 for parallel
                    for i, (ckpt_path, arch_name) in enumerate(batch)
                ]
                results = pool.starmap(analyze_single_model, args)
                
                for result in results:
                    all_results[result['arch']] = result
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Analyze load balance using WikiText-103 test data")
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint (single model mode)")
    parser.add_argument("--arch", type=str, help="Architecture name (single model mode)")
    parser.add_argument("--all", action="store_true", help="Analyze all models in results_dir")
    parser.add_argument("--results_dir", type=str, default="results/arch_comparison_768", help="Results directory")
    parser.add_argument("--gpus", type=int, nargs="+", default=[0], help="GPU IDs to use")
    parser.add_argument("--dim", type=int, default=768, help="Model dimension")
    parser.add_argument("--num_heads", type=int, default=12, help="Number of heads")
    parser.add_argument("--num_hyper_nodes", type=int, default=6, help="Number of timelines")
    parser.add_argument("--top_k", type=int, default=2, help="Top-k routing")
    parser.add_argument("--max_seq_len", type=int, default=2048, help="Max sequence length")
    parser.add_argument("--max_batches", type=int, default=100, help="Max test batches to analyze")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()
    
    config = {
        'dim': args.dim,
        'num_heads': args.num_heads,
        'num_hyper_nodes': args.num_hyper_nodes,
        'top_k': args.top_k,
        'max_seq_len': args.max_seq_len,
    }
    
    if args.all:
        # Analyze all models
        all_results = analyze_all_models_parallel(
            args.results_dir, args.gpus, config, args.max_batches
        )
        
        # Print summary
        print("\n" + "=" * 80)
        print("LOAD BALANCE SUMMARY (WikiText-103 Test Set)")
        print("=" * 80)
        print(f"{'Model':<20} {'Pattern':<18} {'Avg Imbalance':<15} {'Status'}")
        print("-" * 80)
        
        sorted_results = sorted(all_results.items(), key=lambda x: x[1]['avg_imbalance'])
        for arch_name, data in sorted_results:
            pattern = ''.join(ARCHITECTURES.get(arch_name, {}).get('block_pattern', ['?']))
            imb = data['avg_imbalance']
            status = "✅ Good" if imb < 1.5 else ("⚠️ Moderate" if imb < 2.0 else "❌ High")
            print(f"{arch_name:<20} {pattern:<18} {imb:.2f}x{'':<10} {status}")
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(all_results, f, indent=2)
            print(f"\nResults saved to {args.output}")
    
    else:
        # Single model mode
        if not args.checkpoint or not args.arch:
            parser.error("--checkpoint and --arch required for single model mode")
        
        device = torch.device(f'cuda:{args.gpus[0]}' if torch.cuda.is_available() else 'cpu')
        
        print(f"Loading model from {args.checkpoint}...")
        model = load_model(args.checkpoint, args.arch, config, device)
        
        # Load WikiText-103 test data
        tokenizer = FastTokenizer(max_length=1024)
        loaders = create_dataloaders(
            tokenizer,
            dataset_name='wikitext-103',
            max_length=1024,
            batch_size=1,
            num_workers=2,
        )
        
        print(f"Analyzing {args.max_batches} test batches...")
        results = analyze_load_balance_batch(model, loaders['test'], device, args.max_batches)
        
        print_results(results, args.num_hyper_nodes, args.arch)
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
