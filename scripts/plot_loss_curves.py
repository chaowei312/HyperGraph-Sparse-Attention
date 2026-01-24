"""
Plot training and validation loss curves from benchmark results.

Usage:
    python scripts/plot_loss_curves.py --results_dir results/arch_comparison_768
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import matplotlib.pyplot as plt
import numpy as np
import argparse
from glob import glob


def load_results(results_dir: str):
    """Load all JSON result files from directory."""
    results = {}
    for json_file in glob(f"{results_dir}/*.json"):
        with open(json_file) as f:
            data = json.load(f)
            name = data.get('config_name', Path(json_file).stem)
            results[name] = data
    return results


def plot_loss_curves(results: dict, output_path: str = "loss_curves.png"):
    """Plot training and validation loss curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Colors for different models
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    
    # Sort by final val loss
    sorted_names = sorted(results.keys(), 
                          key=lambda x: results[x].get('final_val_loss', float('inf')))
    
    for idx, name in enumerate(sorted_names):
        data = results[name]
        steps = data.get('step', [])
        train_loss = data.get('train_loss', [])
        val_loss = data.get('val_loss', [])
        
        if not steps or not train_loss:
            continue
        
        color = colors[idx]
        label = f"{name} ({data.get('final_val_loss', 0):.4f})"
        
        # Training loss
        axes[0].plot(steps[:len(train_loss)], train_loss, 
                     color=color, alpha=0.7, linewidth=1.5, label=label)
        
        # Validation loss (usually recorded less frequently)
        if val_loss:
            val_steps = steps[:len(val_loss)]
            axes[1].plot(val_steps, val_loss, 
                        color=color, alpha=0.7, linewidth=2, label=label, marker='o', markersize=3)
    
    # Formatting
    axes[0].set_xlabel('Training Steps')
    axes[0].set_ylabel('Training Loss')
    axes[0].set_title('Training Loss Curves')
    axes[0].legend(loc='upper right', fontsize=8)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel('Training Steps')
    axes[1].set_ylabel('Validation Loss')
    axes[1].set_title('Validation Loss Curves')
    axes[1].legend(loc='upper right', fontsize=8)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved loss curves to {output_path}")


def plot_comparison_bar(results: dict, output_path: str = "model_comparison.png"):
    """Plot bar chart comparing final validation loss."""
    # Sort by val loss
    sorted_items = sorted(results.items(), 
                          key=lambda x: x[1].get('final_val_loss', float('inf')))
    
    names = [item[0] for item in sorted_items]
    val_losses = [item[1].get('final_val_loss', 0) for item in sorted_items]
    
    fig, ax = plt.subplots(figsize=(12, 5))
    
    # Color baseline differently
    colors = ['#2ecc71' if 'baseline' in name else '#3498db' for name in names]
    
    bars = ax.barh(names, val_losses, color=colors, edgecolor='black', linewidth=0.5)
    
    # Add value labels
    for bar, val in zip(bars, val_losses):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, 
                f'{val:.4f}', va='center', fontsize=9)
    
    ax.set_xlabel('Validation Loss (lower is better)')
    ax.set_title('Model Comparison: Final Validation Loss')
    ax.set_xlim(min(val_losses) * 0.99, max(val_losses) * 1.01)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved comparison chart to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results/arch_comparison_768")
    parser.add_argument("--output_dir", type=str, default="results/figures")
    args = parser.parse_args()
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"Loading results from {args.results_dir}...")
    results = load_results(args.results_dir)
    print(f"Found {len(results)} models: {list(results.keys())}")
    
    plot_loss_curves(results, f"{args.output_dir}/loss_curves.png")
    plot_comparison_bar(results, f"{args.output_dir}/model_comparison.png")


if __name__ == "__main__":
    main()

