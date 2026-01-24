"""
Data utilities for Sparse Attention experiments.

Provides:
    - FastTokenizer: GPT-2 tokenizer using tiktoken (recommended)
    - Datasets: Gutenberg, WikiText-2/103, OpenWebText, SlimPajama
    
Example:
    ```python
    from data import FastTokenizer, create_dataloaders
    
    tokenizer = FastTokenizer()
    loaders = create_dataloaders(tokenizer, "wikitext-103-small", batch_size=1)
    
    for batch in loaders["train"]:
        input_ids = batch["input_ids"]  # (batch, seq_len)
        labels = batch["labels"]        # (batch, seq_len)
    ```
"""

from .fast_tokenizer import FastTokenizer
from .dataset import (
    TextDataset,
    StreamingTextDataset,
    load_gutenberg,
    load_wikitext,
    load_openwebtext,
    create_dataloaders,
)

__all__ = [
    "FastTokenizer",
    "TextDataset",
    "StreamingTextDataset",
    "load_gutenberg",
    "load_wikitext",
    "load_openwebtext",
    "create_dataloaders",
]
