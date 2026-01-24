"""
Combined timeline routing visualization showing multiple heads/layers.
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
        x = model.embedding(input_ids)
        for i, layer in enumerate(model.layers):
            if i == actual_idx:
                node_logits = attn._compute_node_logits(x)
                probs = torch.softmax(node_logits / attn.router_temperature, dim=-1)
                probs = probs[0, head_idx].cpu().numpy()  # (seq, K)
                top_k_idx = np.argsort(-probs, axis=-1)[:, :2]  # (seq, 2)
                top_k_weights = np.take_along_axis(probs, top_k_idx, axis=-1)
                break
            out = model.layers[i](x)
            x = out[0] if isinstance(out, tuple) else out
    
    return tokens, probs, top_k_idx, top_k_weights, actual_idx


def visualize_combined(model, text: str, save_path: str = "routing_combined.png"):
    """Create combined visualization showing multiple heads."""
    
    # Get routing for different layers/heads
    configs = [
        (0, 0, "Layer 1, Head 0"),
        (0, 5, "Layer 1, Head 5"),
        (3, 0, "Layer 5, Head 0"),
        (3, 5, "Layer 5, Head 5"),
    ]
    
    tokens, _, _, _, _ = get_routing(model, text, 0, 0)
    seq_len = len(tokens)
    K = 6  # Number of timelines
    
    # Colors for timelines
    colors = ['#ff6b6b', '#9b59b6', '#2ecc71', '#3498db', '#f39c12', '#95a5a6']
    
    fig, axes = plt.subplots(len(configs), 1, figsize=(max(14, seq_len * 0.9), len(configs) * 1.8))
    
    for ax_idx, (sparse_layer_idx, head_idx, title) in enumerate(configs):
        ax = axes[ax_idx]
        tokens, probs, top_k_idx, top_k_weights, actual_layer = get_routing(
            model, text, sparse_layer_idx, head_idx
        )
        
        # Draw tokens with timeline colors (only Top-1)
        for i, token in enumerate(tokens):
            t1 = top_k_idx[i, 0]
            w1 = top_k_weights[i, 0]
            
            # Token box - primary timeline color
            rect = mpatches.FancyBboxPatch(
                (i - 0.45, 0.1), 0.9, 0.8,
                boxstyle="round,pad=0.02",
                facecolor=colors[t1],
                edgecolor='#333',
                linewidth=1,
                alpha=0.7 + 0.3 * w1
            )
            ax.add_patch(rect)
            
            # Token text
            display_token = token.replace('\n', '\\n').strip()[:8]
            ax.text(i, 0.5, display_token, ha='center', va='center', 
                    fontsize=8, fontweight='bold', color='white')
            
            # Timeline label below
            ax.text(i, 0.2, f"T{t1}", ha='center', va='center', 
                    fontsize=6, color='white', alpha=0.8)
        
        ax.set_xlim(-0.6, seq_len - 0.4)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(title, fontsize=10, loc='left', pad=2)
    
    # Add legend at bottom
    legend_patches = [mpatches.Patch(color=colors[i], label=f"Timeline {i}") for i in range(K)]
    fig.legend(handles=legend_patches, loc='lower center', ncol=K, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))
    
    # Title
    fig.suptitle(f'Token → Timeline Routing (Top-1 Assignment)\nText: "{text}"', 
                 fontsize=11, y=1.02)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved combined visualization to {save_path}")


def visualize_single_detailed(model, text: str, sparse_layer_idx: int, head_idx: int, 
                               save_path: str = "routing_detailed.png"):
    """Create detailed visualization for a single head showing both top-1 and top-2."""
    
    tokens, probs, top_k_idx, top_k_weights, actual_layer = get_routing(
        model, text, sparse_layer_idx, head_idx
    )
    seq_len = len(tokens)
    K = 6
    
    # Colors
    colors = ['#ff6b6b', '#9b59b6', '#2ecc71', '#3498db', '#f39c12', '#95a5a6']
    
    fig, ax = plt.subplots(figsize=(max(14, seq_len * 1.0), 3.5))
    
    # Draw two rows: Top-1 and Top-2
    for row, label in enumerate(['Top-1', 'Top-2']):
        y_base = 1.5 - row * 1.2
        
        for i, token in enumerate(tokens):
            t_idx = top_k_idx[i, row]
            weight = top_k_weights[i, row]
            
            # Token box
            rect = mpatches.FancyBboxPatch(
                (i - 0.45, y_base), 0.9, 0.9,
                boxstyle="round,pad=0.02",
                facecolor=colors[t_idx],
                edgecolor='#333',
                linewidth=1,
                alpha=0.5 + 0.5 * weight
            )
            ax.add_patch(rect)
            
            # Token text
            display_token = token.replace('\n', '\\n').strip()[:7]
            ax.text(i, y_base + 0.55, display_token, ha='center', va='center', 
                    fontsize=8, fontweight='bold', color='white')
            
            # Timeline label
            ax.text(i, y_base + 0.2, f"T{t_idx}", ha='center', va='center', 
                    fontsize=7, color='white', alpha=0.9)
        
        # Row label
        ax.text(-1, y_base + 0.45, label, ha='right', va='center', fontsize=10, fontweight='bold')
    
    ax.set_xlim(-1.5, seq_len - 0.3)
    ax.set_ylim(-0.1, 2.6)
    ax.set_xlabel('Token Position', fontsize=10)
    ax.set_xticks(range(seq_len))
    ax.set_xticklabels(range(seq_len))
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # Legend on top right (outside plot area)
    legend_patches = [mpatches.Patch(color=colors[i], label=f"Timeline {i}") for i in range(K)]
    ax.legend(handles=legend_patches, loc='upper left', bbox_to_anchor=(1.01, 1), 
              fontsize=8, framealpha=0.9)
    
    ax.set_title(f'Each token routes to top-2 of {K} timelines\nLayer {actual_layer}, Head {head_idx}', 
                 fontsize=11)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved detailed visualization to {save_path}")


def main():
    text = "The quick brown fox jumps over the lazy dog in the park."
    
    print(f"Text: {text}")
    print("Loading model...")
    model = load_sparse_model(
        "results/arch_comparison_768/interlaced_fss_checkpoint.pt",
        "interlaced_fss"
    )
    
    # Combined view (4 heads, top-1 only)
    print("\nGenerating combined visualization...")
    visualize_combined(model, text, "results/figures/routing_combined.png")
    
    # Detailed view for Layer 1, Head 0
    print("\nGenerating detailed visualization...")
    visualize_single_detailed(model, text, 0, 0, "results/figures/routing_detailed_L1H0.png")
    
    # Detailed view for Layer 5, Head 0
    visualize_single_detailed(model, text, 3, 0, "results/figures/routing_detailed_L5H0.png")
    
    print("\nDone!")


if __name__ == "__main__":
    main()

