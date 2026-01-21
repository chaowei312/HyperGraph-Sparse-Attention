"""
End-to-end pipeline test for Sparse Attention models.

Run with: python test_pipeline.py
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

print("=" * 60)
print("SPARSE ATTENTION PIPELINE TEST")
print("=" * 60)

# ============================================================
# 1. Test Imports
# ============================================================
print("\n[1/7] Testing imports...")

try:
    from data import Tokenizer
    from model import CausalLM, ModelConfig
    from model.module import (
        BlockConfig,
        DecoderBlock,
        SparseBlockConfig,
        SparseDecoderBlock,
        RoPEAttention,
        HyperGraphSparseAttention,
        SwiGLU,
    )
    from model.train import (
        Trainer,
        TrainingConfig,
        create_optimizer,
        create_scheduler,
    )
    print("  [OK] All imports successful")
except ImportError as e:
    print(f"  [FAIL] Import error: {e}")
    raise

# ============================================================
# 2. Test Tokenizer
# ============================================================
print("\n[2/7] Testing tokenizer...")

try:
    tokenizer = Tokenizer(max_length=128)
    
    # Encode/decode
    text = "Hello, this is a test of the sparse attention model."
    tokens = tokenizer.encode(text)
    decoded = tokenizer.decode(tokens)
    
    assert len(tokens) > 0, "Tokenization failed"
    assert isinstance(tokens, list), "Expected list of tokens"
    
    # Batch encoding
    batch = tokenizer(["First text", "Second longer text here"], padding=True)
    assert "input_ids" in batch, "Missing input_ids"
    assert "attention_mask" in batch, "Missing attention_mask"
    
    print(f"  [OK] Tokenizer working (vocab_size={tokenizer.vocab_size})")
    print(f"    Sample: '{text[:30]}...' -> {len(tokens)} tokens")
except Exception as e:
    print(f"  [FAIL] Tokenizer error: {e}")
    raise

# ============================================================
# 3. Test Individual Modules
# ============================================================
print("\n[3/7] Testing individual modules...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Using device: {device}")

try:
    batch_size, seq_len, dim = 2, 32, 256
    num_heads = 4
    x = torch.randn(batch_size, seq_len, dim, device=device)
    
    # Test RoPEAttention
    rope_attn = RoPEAttention(
        embed_dim=dim,
        num_heads=num_heads,
        max_seq_len=128,
        causal=True,
    ).to(device)
    out = rope_attn(x)
    assert out.shape == x.shape, f"RoPEAttention shape mismatch: {out.shape}"
    print(f"  [OK] RoPEAttention: {x.shape} -> {out.shape}")
    
    # Test HyperGraphSparseAttention
    sparse_attn = HyperGraphSparseAttention(
        embed_dim=dim,
        num_heads=num_heads,
        num_hyper_nodes=4,
        max_seq_len=128,
    ).to(device)
    out, node_counts = sparse_attn(x)
    assert out.shape == x.shape, f"HyperGraphSparseAttention shape mismatch"
    print(f"  [OK] HyperGraphSparseAttention: {x.shape} -> {out.shape}")
    
    # Test SwiGLU
    swiglu = SwiGLU(in_features=dim, hidden_features=dim * 4).to(device)
    out = swiglu(x)
    assert out.shape == x.shape, f"SwiGLU shape mismatch"
    print(f"  [OK] SwiGLU: {x.shape} -> {out.shape}")
    
except Exception as e:
    print(f"  [FAIL] Module error: {e}")
    raise

# ============================================================
# 4. Test Blocks
# ============================================================
print("\n[4/7] Testing blocks...")

try:
    # Standard DecoderBlock
    block_config = BlockConfig(dim=dim, num_heads=num_heads, max_seq_len=128)
    decoder_block = DecoderBlock(block_config).to(device)
    out = decoder_block(x)
    assert out.shape == x.shape, "DecoderBlock shape mismatch"
    print(f"  [OK] DecoderBlock: {x.shape} -> {out.shape}")
    
    # SparseDecoderBlock
    sparse_config = SparseBlockConfig.from_block_config(block_config, num_hyper_nodes=4)
    sparse_block = SparseDecoderBlock(sparse_config).to(device)
    out, node_counts = sparse_block(x)
    assert out.shape == x.shape, "SparseDecoderBlock shape mismatch"
    print(f"  [OK] SparseDecoderBlock: {x.shape} -> {out.shape}")
    
except Exception as e:
    print(f"  [FAIL] Block error: {e}")
    raise

# ============================================================
# 5. Test Full Model
# ============================================================
print("\n[5/7] Testing full model...")

try:
    # Test different block compositions
    compositions = [
        ("baseline", 4, 0),      # All standard
        ("sparse_only", 0, 4),   # All sparse
        ("hybrid_2_2", 2, 2),    # Hybrid
    ]
    
    for name, n_std, n_sparse in compositions:
        config = ModelConfig(
            dim=256,
            num_heads=4,
            n_standard_blocks=n_std,
            n_sparse_blocks=n_sparse,
            vocab_size=tokenizer.vocab_size,
            max_seq_len=128,
            num_hyper_nodes=4,
        )
        model = CausalLM(config).to(device)
        
        # Forward pass
        input_ids = torch.randint(0, tokenizer.vocab_size, (2, 32), device=device)
        logits = model(input_ids)
        
        assert logits.shape == (2, 32, tokenizer.vocab_size), f"Logits shape mismatch for {name}"
        print(f"  [OK] CausalLM ({name}: {n_std}std+{n_sparse}sparse): input {input_ids.shape} -> logits {logits.shape}")
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"    Model parameters: {num_params:,}")
    
except Exception as e:
    print(f"  [FAIL] Model error: {e}")
    raise

# ============================================================
# 6. Test Training Step
# ============================================================
print("\n[6/7] Testing training step...")

try:
    # Simple dataset
    class DummyDataset(Dataset):
        def __init__(self, vocab_size, seq_len, size=100):
            self.data = torch.randint(0, vocab_size, (size, seq_len))
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            return {"input_ids": self.data[idx], "labels": self.data[idx]}
    
    # Create small model and data
    config = ModelConfig(
        dim=128,
        num_heads=4,
        n_standard_blocks=1,
        n_sparse_blocks=1,
        vocab_size=tokenizer.vocab_size,
        max_seq_len=64,
    )
    model = CausalLM(config).to(device)
    
    dataset = DummyDataset(tokenizer.vocab_size, 64, size=16)
    dataloader = DataLoader(dataset, batch_size=4)
    
    # Training config
    train_config = TrainingConfig(
        learning_rate=1e-4,
        batch_size=4,
        num_epochs=1,
        mixed_precision=False,  # Disable for CPU compatibility
        log_every_n_steps=100,
        save_every_n_steps=0,  # Disable saving for test
        checkpoint_dir="checkpoints",
    )
    
    # Manual training step
    optimizer = create_optimizer(model, train_config)
    
    batch = next(iter(dataloader))
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    
    # Forward
    logits = model(input_ids)
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    
    loss = nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    )
    
    # Backward
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    print(f"  [OK] Training step complete")
    print(f"    Loss: {loss.item():.4f}")
    
except Exception as e:
    print(f"  [FAIL] Training error: {e}")
    raise

# ============================================================
# 7. Test Inference
# ============================================================
print("\n[7/7] Testing inference...")

try:
    model.eval()
    
    # Encode prompt
    prompt = "The quick brown fox"
    input_ids = torch.tensor([tokenizer.encode(prompt)], device=device)
    
    # Generate a few tokens (greedy)
    generated = input_ids.clone()
    
    with torch.no_grad():
        for _ in range(10):
            logits = model(generated)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
    
    output_text = tokenizer.decode(generated[0])
    print(f"  [OK] Generation complete")
    print(f"    Prompt: '{prompt}'")
    print(f"    Output: '{output_text}'")
    
except Exception as e:
    print(f"  [FAIL] Inference error: {e}")
    raise

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print("ALL TESTS PASSED [OK]")
print("=" * 60)
print(f"""
Pipeline ready for training:

1. Tokenizer: GPT-2 (vocab_size={tokenizer.vocab_size})
2. Model: CausalLM with hybrid attention patterns
3. Training: Trainer with AMP, checkpointing, logging

Next steps:
  - Prepare your dataset (DataLoader with input_ids)
  - Configure ModelConfig and TrainingConfig
  - Run: trainer.train(train_dataloader, eval_dataloader)
""")

