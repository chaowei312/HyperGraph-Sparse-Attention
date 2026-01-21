"""
Memory safety test for Sparse Attention model.
Tests forward/backward pass with minimal settings.
"""

import gc
import torch
import torch.nn.functional as F


def log_memory(label: str):
    """Log GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"[{label}] GPU: {allocated:.3f} GB allocated, {reserved:.3f} GB reserved")


def test_model_memory():
    """Test model forward/backward with production-like settings."""
    print("=" * 60)
    print("MEMORY SAFETY TEST (Production-like)")
    print("=" * 60)
    
    from model import ModelConfig, CausalLM
    
    # Production-like settings (same as hybrid_4_4.yaml debug)
    config = ModelConfig(
        vocab_size=50257,  # GPT-2 vocab
        dim=512,           # As in config
        num_heads=8,
        n_standard_blocks=4,
        n_sparse_blocks=4,
        num_hyper_nodes=4,
        max_seq_len=128,   # Reduced for debug
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Check initial memory
    log_memory("initial")
    
    # Create model
    print("\n[1] Creating model...")
    model = CausalLM(config).to(device)
    print(f"    Parameters: {model.num_parameters():,}")
    log_memory("after model")
    
    # Test forward with debug batch
    print("\n[2] Testing forward pass (batch=1, seq=128)...")
    x = torch.randint(0, config.vocab_size, (1, 128), device=device)
    
    with torch.no_grad():
        logits = model(x)
        print(f"    Output shape: {logits.shape}")
    log_memory("after forward")
    
    # Clear cache
    del logits
    torch.cuda.empty_cache()
    gc.collect()
    log_memory("after cleanup")
    
    # Test training forward + backward
    print("\n[3] Testing forward+backward (batch=1, seq=128)...")
    model.train()
    x = torch.randint(0, config.vocab_size, (1, 128), device=device)
    labels = torch.randint(0, config.vocab_size, (1, 128), device=device)
    
    logits = model(x)
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1)
    )
    print(f"    Loss: {loss.item():.4f}")
    log_memory("after loss")
    
    loss.backward()
    log_memory("after backward")
    
    # Clear
    del loss, logits, x, labels
    torch.cuda.empty_cache()
    gc.collect()
    log_memory("after cleanup 2")
    
    # Test with training batch size
    print("\n[4] Testing forward+backward (batch=2, seq=128) - debug training settings...")
    x = torch.randint(0, config.vocab_size, (2, 128), device=device)
    labels = torch.randint(0, config.vocab_size, (2, 128), device=device)
    
    logits = model(x)
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1)
    )
    print(f"    Loss: {loss.item():.4f}")
    log_memory("after loss")
    
    loss.backward()
    log_memory("after backward")
    
    # Clear
    del loss, logits, x, labels
    torch.cuda.empty_cache()
    gc.collect()
    log_memory("cleanup 4")
    
    # Test multiple iterations (simulating training loop)
    print("\n[5] Testing 5 training iterations...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    for i in range(5):
        x = torch.randint(0, config.vocab_size, (2, 128), device=device)
        labels = torch.randint(0, config.vocab_size, (2, 128), device=device)
        
        optimizer.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1)
        )
        loss.backward()
        optimizer.step()
        
        if i == 0:
            log_memory(f"iter {i}")
        if i == 4:
            log_memory(f"iter {i}")
        
        del x, labels, logits, loss
    
    print(f"    Completed 5 iterations successfully")
    
    # Final cleanup
    del model, optimizer
    torch.cuda.empty_cache()
    gc.collect()
    log_memory("final cleanup")
    
    print("\n" + "=" * 60)
    print("MEMORY TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    test_model_memory()

