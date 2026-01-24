"""
Simple timeline routing visualization for one head.

Usage:
    python scripts/visualize_routing_simple.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from data.fast_tokenizer import FastTokenizer
from model import CausalLM
from train.architectures import get_model_config

device = torch.device("cuda:0")
tokenizer = FastTokenizer()

def load_sparse_model(checkpoint_path: str, arch_name: str):
    """Load a trained sparse model."""
    model_cfg = get_model_config(
        arch_name=arch_name,
        dim=768, num_heads=12, num_hyper_nodes=6, top_k=2, max_seq_len=2048,
    )
    model = CausalLM(model_cfg).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    return model

def get_routing(model, text: str, sparse_layer_idx: int = 0, head_idx: int = 0):
    """Get routing assignments."""
    token_ids = tokenizer.encode(text)
    input_ids = torch.tensor([token_ids], dtype=torch.long).to(device)
    tokens = [tokenizer.decode([tid]) for tid in token_ids]
    
    # Find sparse layers
    sparse_layers = [(i, l) for i, l in enumerate(model.layers) 
                     if hasattr(l, 'attention') and hasattr(l.attention, 'node_router')]
    
    actual_idx, sparse_layer = sparse_layers[sparse_layer_idx]
    attn = sparse_layer.attention
    
    # Get routing
    with torch.no_grad():
        # Run through embedding and layers up to the target sparse layer
        x = model.embedding(input_ids)  # (batch, seq, dim)
        
        for i, layer in enumerate(model.layers):
            if i == actual_idx:
                # Get routing from this layer
                node_logits = attn._compute_node_logits(x)  # (batch, heads, seq, K)
                probs = torch.softmax(node_logits / attn.router_temperature, dim=-1)
                probs = probs[0, head_idx].cpu().numpy()  # (seq, K)
                top_k_idx = np.argsort(-probs, axis=-1)[:, :2]  # (seq, 2)
                break
            # Forward through layer
            out = layer(x)
            if isinstance(out, tuple):
                x = out[0]
            else:
                x = out
    
    return tokens, probs, top_k_idx, actual_idx

def visualize(tokens, probs, top_k_idx, layer_idx, head_idx):
    """Create simple visualization."""
    seq_len, K = probs.shape
    
    # 6 distinct colors for timelines
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#a65628']
    
    fig, ax = plt.subplots(figsize=(12, 4))
    
    # Draw tokens with timeline colors
    for i, token in enumerate(tokens):
        t1, t2 = top_k_idx[i]
        w1 = probs[i, t1]
        w2 = probs[i, t2]
        
        # Token box - primary timeline color
        rect = mpatches.FancyBboxPatch(
            (i - 0.4, 0), 0.8, 1.5,
            boxstyle="round,pad=0.02",
            facecolor=colors[t1],
            edgecolor='black',
            linewidth=1,
            alpha=0.7
        )
        ax.add_patch(rect)
        
        # Secondary timeline indicator (small circle)
        circle = plt.Circle((i + 0.25, 1.3), 0.12, color=colors[t2], ec='black', lw=0.5)
        ax.add_patch(circle)
        
        # Token text
        display_token = token.replace('\n', '\\n')[:6]
        ax.text(i, 0.75, display_token, ha='center', va='center', 
                fontsize=8, fontweight='bold', color='white')
        
        # Timeline labels
        ax.text(i, 0.2, f"T{t1}", ha='center', va='center', fontsize=7, color='white')
    
    ax.set_xlim(-0.6, seq_len - 0.4)
    ax.set_ylim(-0.2, 1.8)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Title
    ax.set_title(f"Token → Timeline Routing (Layer {layer_idx}, Head {head_idx})\n"
                 f"Box color = primary timeline, small circle = secondary timeline",
                 fontsize=11)
    
    # Legend
    legend_patches = [mpatches.Patch(color=colors[i], label=f"Timeline {i}") for i in range(K)]
    ax.legend(handles=legend_patches, loc='upper right', ncol=3, fontsize=8)
    
    plt.tight_layout()
    plt.savefig('timeline_routing.png', dpi=120, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved to timeline_routing.png")

def main():
    text = "The quick brown fox jumps over the lazy dog"
    print(f"Text: {text}")
    print("Loading model...")
    model = load_sparse_model(
        "results/arch_comparison_768/interlaced_fss_checkpoint.pt",
        "interlaced_fss"
    )
    
    print("Getting routing assignments...")
    tokens, probs, top_k_idx, layer_idx = get_routing(model, text, sparse_layer_idx=0, head_idx=0)
    
    print("Creating visualization...")
    visualize(tokens, probs, top_k_idx, layer_idx, head_idx=0)
    
    # Also print text summary
    print("\n" + "=" * 60)
    print("ROUTING SUMMARY (Head 0, First Sparse Layer)")
    print("=" * 60)
    colors = ['🔴', '🔵', '🟢', '🟣', '🟠', '🟤']
    for i, token in enumerate(tokens):
        t1, t2 = top_k_idx[i]
        w1, w2 = probs[i, t1], probs[i, t2]
        print(f"{token:10s} → {colors[t1]} T{t1} ({w1:.2f}) + {colors[t2]} T{t2} ({w2:.2f})")

if __name__ == "__main__":
    main()

