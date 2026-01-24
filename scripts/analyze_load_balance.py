"""
Analyze load balance across timelines for sparse attention models.

Usage:
    python scripts/analyze_load_balance.py --checkpoint results/arch_comparison_768/interlaced_fss_checkpoint.pt \
                                           --arch interlaced_fss
"""
import argparse
import torch
import numpy as np
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from model import CausalLM
from train.architectures import get_model_config
from data.fast_tokenizer import FastTokenizer


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


def analyze_load_balance(model, text: str, tokenizer, device: torch.device):
    """Analyze load balance across all sparse layers."""
    token_ids = tokenizer.encode(text)
    input_ids = torch.tensor([token_ids], dtype=torch.long).to(device)
    
    # Find sparse layers
    sparse_layers = [(i, l) for i, l in enumerate(model.layers) 
                     if hasattr(l, 'attention') and hasattr(l.attention, 'node_router')]
    
    results = []
    
    with torch.no_grad():
        x = model.embedding(input_ids)
        
        for sparse_idx, (layer_idx, layer) in enumerate(sparse_layers):
            # Forward through previous layers
            for i in range(layer_idx):
                if i < layer_idx:
                    out = model.layers[i](x)
                    x = out[0] if isinstance(out, tuple) else out
            
            attn = layer.attention
            K = attn.num_hyper_nodes
            H = attn.num_heads
            
            # Get routing
            node_logits = attn._compute_node_logits(x)
            probs = torch.softmax(node_logits / attn.router_temperature, dim=-1)
            primary = probs.argmax(dim=-1)  # (batch, heads, seq)
            
            # Count per timeline per head
            all_counts = []
            for h in range(H):
                counts = torch.bincount(primary[0, h], minlength=K).cpu().numpy()
                all_counts.append(counts)
            
            all_counts = np.array(all_counts)  # (H, K)
            avg_counts = all_counts.mean(axis=0)
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
                'max_load': max_load,
                'min_load': min_load,
                'imbalance': imbalance,
                'std': std_load,
                'ideal': ideal,
            })
            
            # Forward through this layer
            out = layer(x)
            x = out[0] if isinstance(out, tuple) else out
    
    return results


def print_results(results, K):
    """Print load balance analysis results."""
    print("=" * 75)
    print(f"LOAD BALANCE ANALYSIS ({K} Timelines)")
    print("=" * 75)
    
    for r in results:
        print(f"\nLayer {r['layer_idx']} (Sparse Layer #{r['sparse_idx'] + 1})")
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
    
    print("\n" + "=" * 75)
    avg_imbalance = np.mean([r['imbalance'] for r in results])
    print(f"OVERALL: Average imbalance = {avg_imbalance:.2f}x")
    print("=" * 75)


def main():
    parser = argparse.ArgumentParser(description="Analyze load balance")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--arch", type=str, required=True, help="Architecture name")
    parser.add_argument("--dim", type=int, default=768, help="Model dimension")
    parser.add_argument("--num_heads", type=int, default=12, help="Number of heads")
    parser.add_argument("--num_hyper_nodes", type=int, default=6, help="Number of timelines")
    parser.add_argument("--top_k", type=int, default=2, help="Top-k routing")
    parser.add_argument("--max_seq_len", type=int, default=2048, help="Max sequence length")
    parser.add_argument("--text", type=str, default=None, help="Custom text to analyze")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    tokenizer = FastTokenizer()
    
    config = {
        'dim': args.dim,
        'num_heads': args.num_heads,
        'num_hyper_nodes': args.num_hyper_nodes,
        'top_k': args.top_k,
        'max_seq_len': args.max_seq_len,
    }
    
    # Default test text
    if args.text:
        text = args.text
    else:
        text = """The quick brown fox jumps over the lazy dog in the park. 
Scientists have discovered that artificial intelligence can learn complex patterns.
The future of technology will transform how we live and work.
Natural language processing enables machines to understand human communication."""
    
    print(f"Loading model from {args.checkpoint}...")
    model = load_model(args.checkpoint, args.arch, config, device)
    
    print(f"Analyzing text ({len(tokenizer.encode(text))} tokens)...")
    results = analyze_load_balance(model, text, tokenizer, device)
    
    print_results(results, args.num_hyper_nodes)
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()

