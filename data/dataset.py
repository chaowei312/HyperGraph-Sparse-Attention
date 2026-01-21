"""
Dataset utilities for language model training.

Standard benchmarks:
    - WikiText-2: Small (2M tokens) - quick iteration
    - WikiText-103: Medium (103M tokens) - standard benchmark
    - OpenWebText: Large - full pretraining
"""

from typing import Optional, Dict, List
import torch
from torch.utils.data import Dataset, DataLoader


class TextDataset(Dataset):
    """
    Simple text dataset for causal language modeling.
    
    Tokenizes text and stores as single tensor (memory efficient).
    
    Args:
        texts: List of text strings or single text
        tokenizer: Tokenizer instance
        max_length: Maximum sequence length
        stride: Stride for sliding window (default: max_length, no overlap)
    """
    
    def __init__(
        self,
        texts: List[str],
        tokenizer,
        max_length: int = 512,
        stride: Optional[int] = None,
    ):
        self.max_length = max_length
        self.stride = stride or max_length
        
        # Tokenize all texts and concatenate
        all_tokens = []
        for text in texts:
            tokens = tokenizer.encode(text, truncation=False, max_length=None)
            all_tokens.extend(tokens)
        
        # Store as single contiguous tensor (much more memory efficient than list of lists)
        self.tokens = torch.tensor(all_tokens, dtype=torch.long)
        
        # Number of valid sequences
        self.num_sequences = max(0, (len(self.tokens) - max_length) // self.stride)
    
    def __len__(self) -> int:
        return self.num_sequences
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        start = idx * self.stride
        end = start + self.max_length + 1  # +1 for labels
        tokens = self.tokens[start:end]
        return {"input_ids": tokens[:-1], "labels": tokens[1:]}


def load_wikitext(
    tokenizer,
    split: str = "train",
    version: str = "wikitext-103-v1",
    max_length: int = 512,
    stride: Optional[int] = None,
) -> TextDataset:
    """
    Load WikiText dataset.
    
    Args:
        tokenizer: Tokenizer instance
        split: "train", "validation", or "test"
        version: "wikitext-2-v1" (small) or "wikitext-103-v1" (standard)
        max_length: Sequence length
        stride: Sliding window stride
        
    Returns:
        TextDataset ready for DataLoader
        
    Example:
        ```python
        from data import Tokenizer, load_wikitext
        
        tokenizer = Tokenizer()
        train_dataset = load_wikitext(tokenizer, split="train")
        val_dataset = load_wikitext(tokenizer, split="validation")
        
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        ```
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "datasets package required. Install with: pip install datasets"
        )
    
    # Map split names
    split_map = {"train": "train", "validation": "validation", "val": "validation", "test": "test"}
    hf_split = split_map.get(split, split)
    
    # Load from HuggingFace
    dataset = load_dataset("wikitext", version, split=hf_split)
    
    # Filter empty lines and concatenate
    texts = [item["text"] for item in dataset if item["text"].strip()]
    
    return TextDataset(texts, tokenizer, max_length=max_length, stride=stride)


def load_openwebtext(
    tokenizer,
    split: str = "train",
    max_length: int = 512,
    stride: Optional[int] = None,
    num_samples: Optional[int] = None,
) -> TextDataset:
    """
    Load OpenWebText dataset (GPT-2's training data recreation).
    
    Note: This is a large dataset (~38GB). Use num_samples to limit.
    
    Args:
        tokenizer: Tokenizer instance
        split: "train" (only split available)
        max_length: Sequence length
        stride: Sliding window stride
        num_samples: Limit number of samples (for quick testing)
        
    Returns:
        TextDataset ready for DataLoader
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "datasets package required. Install with: pip install datasets"
        )
    
    dataset = load_dataset("openwebtext", split=split)
    
    if num_samples:
        dataset = dataset.select(range(min(num_samples, len(dataset))))
    
    texts = [item["text"] for item in dataset if item["text"].strip()]
    
    return TextDataset(texts, tokenizer, max_length=max_length, stride=stride)


def create_dataloaders(
    tokenizer,
    dataset_name: str = "wikitext-103",
    max_length: int = 512,
    batch_size: int = 8,
    num_workers: int = 4,
    stride: Optional[int] = None,
    max_train_samples: Optional[int] = None,
    max_eval_samples: Optional[int] = None,
) -> Dict[str, DataLoader]:
    """
    Create train/val/test DataLoaders for a dataset.
    
    Args:
        tokenizer: Tokenizer instance
        dataset_name: "wikitext-2", "wikitext-103", or "openwebtext"
        max_length: Sequence length
        batch_size: Batch size
        num_workers: DataLoader workers
        stride: Sliding window stride
        
    Returns:
        Dict with "train", "val", "test" DataLoaders
        
    Example:
        ```python
        from data import Tokenizer, create_dataloaders
        
        tokenizer = Tokenizer()
        loaders = create_dataloaders(tokenizer, "wikitext-103", batch_size=8)
        
        for batch in loaders["train"]:
            input_ids = batch["input_ids"]
            labels = batch["labels"]
            ...
        ```
    """
    # Map dataset names to versions
    version_map = {
        "wikitext-2": "wikitext-2-v1",
        "wikitext-103": "wikitext-103-v1",
        "wikitext-2-v1": "wikitext-2-v1",
        "wikitext-103-v1": "wikitext-103-v1",
    }
    
    if dataset_name in version_map:
        version = version_map[dataset_name]
        train_ds = load_wikitext(tokenizer, "train", version, max_length, stride)
        val_ds = load_wikitext(tokenizer, "validation", version, max_length, stride)
        test_ds = load_wikitext(tokenizer, "test", version, max_length, stride)
    elif dataset_name == "openwebtext":
        train_ds = load_openwebtext(tokenizer, "train", max_length, stride)
        val_ds = None  # OpenWebText has no val/test split
        test_ds = None
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Limit samples if requested (for quick testing)
    if max_train_samples and len(train_ds) > max_train_samples:
        train_ds = torch.utils.data.Subset(train_ds, range(max_train_samples))
    if max_eval_samples and val_ds and len(val_ds) > max_eval_samples:
        val_ds = torch.utils.data.Subset(val_ds, range(max_eval_samples))
    if max_eval_samples and test_ds and len(test_ds) > max_eval_samples:
        test_ds = torch.utils.data.Subset(test_ds, range(max_eval_samples))
    
    # Windows multiprocessing can cause memory issues - use 0 workers by default
    import sys
    if sys.platform == "win32" and num_workers > 0:
        import warnings
        warnings.warn("Reducing num_workers to 0 on Windows to avoid memory issues")
        num_workers = 0
    
    loaders = {
        "train": DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=False,  # Avoid pinned memory overhead
        ),
    }
    
    if val_ds:
        loaders["val"] = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
        )
    
    if test_ds:
        loaders["test"] = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
        )
    
    return loaders

