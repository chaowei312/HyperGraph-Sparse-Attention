"""
Data utilities for Sparse Attention experiments.

Provides:
    - Tokenizer: GPT-2 tokenizer (standard for research)
    - Datasets: WikiText-2/103, OpenWebText
    
Example:
    ```python
    from data import Tokenizer, create_dataloaders
    
    tokenizer = Tokenizer()
    loaders = create_dataloaders(tokenizer, "wikitext-103", batch_size=8)
    
    for batch in loaders["train"]:
        input_ids = batch["input_ids"]  # (batch, seq_len)
        labels = batch["labels"]        # (batch, seq_len)
    ```
"""

from .tokenizer import Tokenizer
from .dataset import (
    TextDataset,
    load_wikitext,
    load_openwebtext,
    create_dataloaders,
)

__all__ = [
    "Tokenizer",
    "TextDataset",
    "load_wikitext",
    "load_openwebtext",
    "create_dataloaders",
]
