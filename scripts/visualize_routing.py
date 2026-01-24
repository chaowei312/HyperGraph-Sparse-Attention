"""
Visualize timeline routing for one attention head.

Usage:
    python scripts/visualize_routing.py
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

# Load tokenizer
tokenizer = FastTokenizer()

def load_sparse_model(checkpoint_path: str, arch_name: str):
    """Load a trained sparse model."""
    model_cfg = get_model_config(
        arch_name=arch_name,
        dim=768,
        num_heads=12,
        num_hyper_nodes=6,
        top_k=2,
        max_seq_len=2048,
    )
    model = CausalLM(model_cfg).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    return model

def get_routing_assignments(model, text: str, layer_idx: int = 0, head_idx: int = 0):
    """
    Get timeline routing assignments for a specific layer and head.
    
    Returns:
        tokens: list of token strings
        assignments: tensor of shape (seq_len, top_k) with timeline indices
        weights: tensor of shape (seq_len, top_k) with routing weights
    """
    # Tokenize
    token_ids = tokenizer.encode(text)
    input_ids = torch.tensor([token_ids], dtype=torch.long).to(device)
    seq_len = len(token_ids)
    
    # Get token strings for display
    tokens = [tokenizer.decode([tid]) for tid in token_ids]
    
    # Find the sparse attention layer
    sparse_layers = []
    for i, layer in enumerate(model.layers):
        if hasattr(layer, 'attention') and hasattr(layer.attention, 'node_router'):
            sparse_layers.append((i, layer))
    
    if not sparse_layers:
        raise ValueError("No sparse layers found in model")
    
    # Get the requested layer
    if layer_idx >= len(sparse_layers):
        layer_idx = 0
    actual_layer_idx, sparse_layer = sparse_layers[layer_idx]
    
    # Hook to capture routing
    routing_info = {}
    
    def hook_fn(module, input, output):
        # Get the hidden states
        x = input[0]  # (batch, seq_len, dim)
        batch, seq, dim = x.shape
        
        # Compute router logits using the module's method
        num_heads = module.num_heads
        K = module.num_hyper_nodes
        
        # Use the module's routing computation
        node_logits = module._compute_node_logits(x)  # (batch, heads, seq, K)
        
        # Get softmax probs
        probs = torch.softmax(node_logits / module.router_temperature, dim=-1)
        
        # Store for the specific head: transpose to (batch, seq, heads, K)
        probs_transposed = probs.transpose(1, 2)  # (batch, seq, heads, K)
        logits_transposed = node_logits.transpose(1, 2)  # (batch, seq, heads, K)
        
        routing_info['logits'] = logits_transposed[0, :, head_idx, :].detach().cpu()  # (seq, K)
        routing_info['probs'] = probs_transposed[0, :, head_idx, :].detach().cpu()  # (seq, K)
        
        # Get top-k indices and weights
        top_k = module.top_k
        top_k_probs, top_k_indices = torch.topk(routing_info['probs'], top_k, dim=-1)
        routing_info['top_k_indices'] = top_k_indices  # (seq, top_k)
        routing_info['top_k_weights'] = top_k_probs  # (seq, top_k)
    
    # Register hook
    handle = sparse_layer.attention.register_forward_hook(hook_fn)
    
    # Forward pass
    with torch.no_grad():
        model(input_ids)
    
    # Remove hook
    handle.remove()
    
    return tokens, routing_info, actual_layer_idx

def visualize_routing(tokens, routing_info, layer_idx, head_idx, save_path="routing_viz.png"):
    """Create a visualization of timeline routing."""
    
    K = routing_info['probs'].shape[1]  # number of timelines
    seq_len = len(tokens)
    top_k = routing_info['top_k_indices'].shape[1]
    
    # Colors for each timeline (6 distinct colors)
    colors = plt.cm.Set2(np.linspace(0, 1, K))[:, :3]  # RGB only
    timeline_names = [f"T{i}" for i in range(K)]
    
    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(max(14, seq_len * 0.5), 8), 
                              gridspec_kw={'height_ratios': [1, 2]})
    
    # === Top plot: Token sequence with color-coded timelines ===
    ax1 = axes[0]
    ax1.set_xlim(-0.5, seq_len - 0.5)
    ax1.set_ylim(-0.5, top_k + 0.5)
    
    for i, token in enumerate(tokens):
        for k in range(top_k):
            timeline_idx = routing_info['top_k_indices'][i, k].item()
            weight = routing_info['top_k_weights'][i, k].item()
            color = colors[timeline_idx]
            
            # Draw colored rectangle
            rect = mpatches.FancyBboxPatch(
                (i - 0.45, k), 0.9, 0.8,
                boxstyle="round,pad=0.02",
                facecolor=(*color, 0.3 + 0.7 * weight),  # alpha based on weight
                edgecolor=color,
                linewidth=2
            )
            ax1.add_patch(rect)
            
            # Add token text
            display_token = token.replace('\n', '\\n')[:8]  # Truncate long tokens
            ax1.text(i, k + 0.4, display_token, ha='center', va='center', 
                    fontsize=7, fontweight='bold')
            
            # Add timeline label
            ax1.text(i, k + 0.1, f"T{timeline_idx}", ha='center', va='center',
                    fontsize=6, color=color)
    
    ax1.set_yticks(range(top_k))
    ax1.set_yticklabels([f"Top-{k+1}" for k in range(top_k)])
    ax1.set_xlabel("Token Position")
    ax1.set_title(f"Token → Timeline Routing (Layer {layer_idx}, Head {head_idx})\n"
                  f"Each token routes to top-{top_k} of {K} timelines", fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # === Bottom plot: Heatmap of routing probabilities ===
    ax2 = axes[1]
    probs = routing_info['probs'].numpy()  # (seq, K)
    
    im = ax2.imshow(probs.T, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)
    ax2.set_xlabel("Token Position")
    ax2.set_ylabel("Timeline")
    ax2.set_yticks(range(K))
    ax2.set_yticklabels([f"Timeline {i}" for i in range(K)])
    ax2.set_title("Routing Probability Heatmap (darker = higher probability)")
    
    # Add token labels on x-axis (sample if too many)
    if seq_len <= 30:
        ax2.set_xticks(range(seq_len))
        ax2.set_xticklabels([t.replace('\n', '\\n')[:6] for t in tokens], 
                           rotation=45, ha='right', fontsize=7)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax2, shrink=0.8)
    cbar.set_label("Routing Probability")
    
    # Legend for timeline colors
    legend_patches = [mpatches.Patch(color=colors[i], label=f"Timeline {i}") 
                      for i in range(K)]
    fig.legend(handles=legend_patches, loc='upper right', bbox_to_anchor=(0.99, 0.99))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Saved visualization to {save_path}")
    return save_path

def main():
    # Sample text
    text = "The quick brown fox jumps over the lazy dog in the park."
    
    print("=" * 60)
    print("TIMELINE ROUTING VISUALIZATION")
    print("=" * 60)
    print(f"Text: {text}")
    print(f"Config: 768 dim, 12 heads, K=6 timelines, top_k=2")
    print("=" * 60)
    
    # Load sparse model
    print("\nLoading sparse model (interlaced_fss)...")
    model = load_sparse_model(
        "results/arch_comparison_768/interlaced_fss_checkpoint.pt",
        "interlaced_fss"
    )
    
    # Get routing for different layers/heads
    for layer_idx in [0, 3]:  # First sparse layer and a later one
        for head_idx in [0, 5]:  # Different heads
            print(f"\nExtracting routing for layer {layer_idx}, head {head_idx}...")
            tokens, routing_info, actual_layer = get_routing_assignments(
                model, text, layer_idx=layer_idx, head_idx=head_idx
            )
            
            save_path = f"routing_L{actual_layer}_H{head_idx}.png"
            visualize_routing(tokens, routing_info, actual_layer, head_idx, save_path)
    
    print("\n" + "=" * 60)
    print("Done! Generated routing visualizations.")
    print("=" * 60)

if __name__ == "__main__":
    main()

