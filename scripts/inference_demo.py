"""
Quick inference demo for baseline vs sparse models.

Usage:
    python scripts/inference_demo.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
from data.fast_tokenizer import FastTokenizer
from model import CausalLM, ModelConfig
from train.architectures import get_model_config

device = torch.device("cuda:0")

# Load tokenizer (tiktoken-based, no network needed)
tokenizer = FastTokenizer()

def load_model(checkpoint_path: str, arch_name: str):
    """Load a trained model from checkpoint."""
    # Get model config (matching training config)
    model_cfg = get_model_config(
        arch_name=arch_name,
        dim=768,
        num_heads=12,
        num_hyper_nodes=6,
        top_k=2,
        max_seq_len=2048,
    )
    
    model = CausalLM(model_cfg).to(device)
    
    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    
    model.eval()
    return model

@torch.no_grad()
def generate(model, prompt: str, max_new_tokens: int = 50, temperature: float = 0.8, top_p: float = 0.9):
    """Generate text from a prompt."""
    # Tokenize prompt
    tokens = tokenizer.encode(prompt)
    input_ids = torch.tensor([tokens], dtype=torch.long).to(device)
    
    generated = input_ids
    
    for _ in range(max_new_tokens):
        # Forward pass
        outputs = model(generated)
        if isinstance(outputs, tuple):
            logits = outputs[0]
        else:
            logits = outputs
        
        # Get next token logits
        next_logits = logits[:, -1, :] / temperature
        
        # Top-p (nucleus) sampling
        sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative probability above threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        next_logits[indices_to_remove] = float('-inf')
        
        # Sample
        probs = F.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        # Append
        generated = torch.cat([generated, next_token], dim=1)
        
        # Stop at EOS
        if next_token.item() == tokenizer.eos_token_id:
            break
    
    return tokenizer.decode(generated[0], skip_special_tokens=True)

def main():
    prompts = [
        "The history of artificial intelligence",
        "In the beginning, there was",
        "Scientists have discovered that",
        "The future of technology will",
    ]
    
    print("=" * 70)
    print("INFERENCE COMPARISON: Baseline vs Best Sparse (interlaced_fss)")
    print("=" * 70)
    print(f"Config: 768 dim, 12 heads, K=6 timelines, top_k=2")
    print("=" * 70)
    
    # Load models
    print("\nLoading baseline model...")
    baseline = load_model(
        "results/arch_comparison_768/baseline_checkpoint.pt",
        "baseline"
    )
    
    print("Loading interlaced_fss model (best sparse)...")
    sparse = load_model(
        "results/arch_comparison_768/interlaced_fss_checkpoint.pt",
        "interlaced_fss"
    )
    
    # Generate
    for prompt in prompts:
        print("\n" + "=" * 70)
        print(f"PROMPT: {prompt}")
        print("=" * 70)
        
        print("\n[BASELINE]:")
        baseline_output = generate(baseline, prompt, max_new_tokens=60)
        print(baseline_output)
        
        print("\n[SPARSE (interlaced_fss)]:")
        sparse_output = generate(sparse, prompt, max_new_tokens=60)
        print(sparse_output)

if __name__ == "__main__":
    main()

