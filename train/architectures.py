"""
Architecture configurations for HyperGraph Sparse Attention experiments.

This module provides a single source of truth for architecture definitions
used by both training (parallel_train.py) and benchmarking (benchmark.py).

All sparse architectures use fixed ratio: 6 Full (F) + 8 Sparse (S) = 14 layers
This enables comparing PLACEMENT patterns rather than sparsity density.
"""

from model import ModelConfig


# Architecture constants
DEPTH = 14          # Total number of layers
N_FULL = 6          # Number of full attention layers
N_SPARSE = 8        # Number of sparse attention layers
SPARSITY_PCT = N_SPARSE / DEPTH * 100  # 57.1%


ARCHITECTURES = {
    # =========================================================================
    # BASELINE: All full attention (0% sparse) - quality reference
    # =========================================================================
    "baseline": {
        "block_pattern": ["F"] * DEPTH,
        "description": "All Full attention (0% sparse) - quality reference",
    },
    
    # =========================================================================
    # PLACEMENT COMPARISONS (all 6F + 8S = 57% sparse)
    # =========================================================================
    
    # Full attention early, sparse late (global context first)
    "early_full": {
        "block_pattern": ["F"] * N_FULL + ["S"] * N_SPARSE,  # FFFFFFSSSSSSSS
        "description": "Full early: FFFFFF + SSSSSSSS (global context first)",
    },
    
    # Sparse early, full attention late (sparse encoding, full decode)  
    "late_full": {
        "block_pattern": ["S"] * N_SPARSE + ["F"] * N_FULL,  # SSSSSSSSFFFFFF
        "description": "Full late: SSSSSSSS + FFFFFF (sparse encode, full decode)",
    },
    
    # Bookend pattern: Full attention at input/output, sparse in middle
    "bookend": {
        "block_pattern": ["F"] * 3 + ["S"] * 8 + ["F"] * 3,  # FFFSSSSSSSSFFF
        "description": "Bookend: FFF + SSSSSSSS + FFF (full I/O, sparse middle)",
    },
    
    # Interlaced: alternating SF with extra S at end for 8S+6F
    "interlaced_sf": {
        "block_pattern": list("SFSFSFSFSFSFSS"),  # SF*6 + SS = 8S + 6F
        "description": "Interlaced SF: SFSFSFSFSFSFSS (alternating, sparse-heavy end)",
    },
    
    # Interlaced: FSS pattern repeated  
    "interlaced_fss": {
        "block_pattern": list("FSSFSSFSSFSSFF"),  # (FSS)*4 + FF = 8S + 6F
        "description": "Interlaced FSS: FSSFSSFSSFSSFF (full-first variant)",
    },
    
    # Chunked: groups of 4S then 2F
    "chunked_4s2f": {
        "block_pattern": list("SSSSFFSSSSFFFF"),  # 8S + 6F
        "description": "Chunked 4+2: SSSSFF + SSSSFF + FF (local sparse blocks)",
    },
    
    # Chunked: 2F then 4S groups
    "chunked_2f4s": {
        "block_pattern": list("FFSSSSFFSSSSFF"),  # 6F + 8S
        "description": "Chunked 2+4: FF + SSSS + FF + SSSS + FF (full boundaries)",
    },
    
    # Reverse bookend: Sparse at edges, full in middle
    "reverse_bookend": {
        "block_pattern": list("SSSFFFFFFSSSSS"),  # 3S + 6F + 5S = 8S + 6F
        "description": "Reverse bookend: SSS + FFFFFF + SSSSS (sparse I/O, full middle)",
    },
    
    # Custom: 12 layers for toy experiments with MoH comparison
    "custom_12l": {
        "block_pattern": list("FFSSFSSFSSFF"),  # 4F + 8S = 12 layers (67% sparse)
        "description": "Custom 12L: FFSSFSSFSSFF for MoH comparison (67% sparse)",
    },
    
    # Baseline 12 layers (all full attention)
    "baseline_12l": {
        "block_pattern": ["F"] * 12,  # All full attention
        "description": "Baseline 12L: All full attention (0% sparse)",
    },
}


def get_model_config(
    arch_name: str,
    dim: int = 512,
    num_heads: int = 8,
    num_hyper_nodes: int = 4,
    top_k: int = 1,
    max_seq_len: int = 8192,
    vocab_size: int = 50257,
) -> ModelConfig:
    """
    Create ModelConfig for a given architecture.
    
    Args:
        arch_name: Architecture name from ARCHITECTURES dict
        dim: Model dimension (d_model)
        num_heads: Number of attention heads
        num_hyper_nodes: Number of hyper nodes (K timelines) per head
        top_k: Number of timelines each token routes to
        max_seq_len: Maximum sequence length
        vocab_size: Vocabulary size (default: GPT-2 tokenizer size)
    
    Returns:
        ModelConfig with specified architecture
        
    Raises:
        KeyError: If arch_name not in ARCHITECTURES
    """
    if arch_name not in ARCHITECTURES:
        available = ", ".join(ARCHITECTURES.keys())
        raise KeyError(f"Unknown architecture '{arch_name}'. Available: {available}")
    
    arch = ARCHITECTURES[arch_name]
    return ModelConfig(
        vocab_size=vocab_size,
        dim=dim,
        num_heads=num_heads,
        num_hyper_nodes=num_hyper_nodes,
        top_k=top_k,
        block_pattern=arch["block_pattern"],
        max_seq_len=max_seq_len,
    )


def get_architecture_info(arch_name: str) -> dict:
    """
    Get architecture information including computed metrics.
    
    Args:
        arch_name: Architecture name from ARCHITECTURES dict
        
    Returns:
        Dict with pattern, description, and computed metrics
    """
    arch = ARCHITECTURES[arch_name]
    pattern = arch["block_pattern"]
    pattern_str = "".join(pattern)
    n_sparse = pattern_str.count("S")
    n_full = pattern_str.count("F")
    
    return {
        "name": arch_name,
        "pattern": pattern_str,
        "description": arch["description"],
        "n_layers": len(pattern),
        "n_sparse": n_sparse,
        "n_full": n_full,
        "sparse_pct": n_sparse / len(pattern) * 100,
    }


def list_architectures() -> None:
    """Print all available architectures."""
    print(f"{'Name':<18} {'Pattern':<16} {'Sparse%':<8} Description")
    print("-" * 80)
    for name in ARCHITECTURES:
        info = get_architecture_info(name)
        print(f"{info['name']:<18} {info['pattern']:<16} {info['sparse_pct']:<8.1f} {info['description']}")


# Validate all architectures on import
def _validate_architectures():
    """Validate that all architectures have correct layer counts."""
    # Architectures that can have different layer counts
    special_archs = {"baseline_12l", "custom_12l"}
    
    for name, arch in ARCHITECTURES.items():
        pattern = arch["block_pattern"]
        n_s = sum(1 for x in pattern if x == "S")
        n_f = sum(1 for x in pattern if x == "F")
        total = len(pattern)
        
        # Skip validation for special architectures
        if name in special_archs:
            continue
        
        if total != DEPTH:
            raise ValueError(f"Architecture '{name}' has {total} layers, expected {DEPTH}")
        
        # Baseline is all-full, others should be 6F+8S
        if name != "baseline":
            if n_s != N_SPARSE or n_f != N_FULL:
                raise ValueError(
                    f"Architecture '{name}' has {n_f}F+{n_s}S, "
                    f"expected {N_FULL}F+{N_SPARSE}S"
                )

_validate_architectures()

