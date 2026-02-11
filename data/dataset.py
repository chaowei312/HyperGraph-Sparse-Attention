"""
Dataset utilities for language model training.

Supports:
- NLTK Gutenberg: ~2.4M tokens (small, built-in)
- WikiText-103: ~100M tokens (medium, via HuggingFace)
- OpenWebText: ~8B tokens (large, via HuggingFace)
- SlimPajama: ~627B tokens (very large, via HuggingFace)
"""

from typing import Optional, Dict, Union
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset


class TextDataset(Dataset):
    """
    Simple text dataset for causal language modeling.
    Stores pre-tokenized data as a single tensor.
    """
    
    def __init__(
        self,
        tokens: torch.Tensor,
        max_length: int = 512,
        stride: Optional[int] = None,
    ):
        self.max_length = max_length
        self.stride = stride or max_length
        self.tokens = tokens
        self.num_sequences = max(0, (len(self.tokens) - max_length) // self.stride)
    
    def __len__(self) -> int:
        return self.num_sequences
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        start = idx * self.stride
        end = start + self.max_length + 1
        tokens = self.tokens[start:end]
        return {"input_ids": tokens[:-1], "labels": tokens[1:]}


def load_gutenberg(
    tokenizer,
    split: str = "train",
    max_length: int = 512,
    stride: Optional[int] = None,
) -> TextDataset:
    """
    Load NLTK Gutenberg corpus (classic literature).
    
    Includes works by:
    - Shakespeare (Hamlet, Macbeth, Caesar)
    - Milton (Paradise Lost)
    - Austen, Melville, Whitman, etc.
    
    Args:
        tokenizer: Tokenizer instance with encode() method
        split: "train", "valid"/"validation", or "test"
        max_length: Sequence length
        stride: Sliding window stride
        
    Returns:
        TextDataset ready for DataLoader
    """
    from nltk.corpus import gutenberg
    
    # Get all file IDs
    file_ids = gutenberg.fileids()
    
    # Split: 70% train, 15% val, 15% test
    n = len(file_ids)
    train_ids = file_ids[:int(n * 0.7)]
    val_ids = file_ids[int(n * 0.7):int(n * 0.85)]
    test_ids = file_ids[int(n * 0.85):]
    
    # Map splits
    split_map = {
        "train": train_ids,
        "validation": val_ids,
        "valid": val_ids,
        "val": val_ids,
        "test": test_ids,
    }
    selected_ids = split_map.get(split, train_ids)
    
    # Concatenate all text from selected files
    all_text = "\n\n".join([gutenberg.raw(fid) for fid in selected_ids])
    
    # Tokenize
    tokens = tokenizer.encode(all_text, truncation=False, max_length=None)
    tokens = torch.tensor(tokens, dtype=torch.long)
    
    return TextDataset(tokens, max_length, stride)


def load_wikitext(
    tokenizer,
    split: str = "train",
    max_length: int = 512,
    stride: Optional[int] = None,
    variant: str = "103",  # "2" or "103"
) -> TextDataset:
    """
    Load WikiText-2 or WikiText-103 dataset.
    
    First tries to load from local files, then falls back to HuggingFace.
    
    WikiText-2: ~2M tokens (similar to Gutenberg)
    WikiText-103: ~100M tokens (50x larger)
    
    Args:
        tokenizer: Tokenizer instance
        split: "train", "validation", or "test"
        max_length: Sequence length
        stride: Sliding window stride
        variant: "2" for WikiText-2, "103" for WikiText-103
    """
    import os
    
    # Try local files first
    local_paths = [
        f"/home/lopedg/project/data/data/wikitext-{variant}",
        f"/home/lopedg/project/HyperGraph-Sparse-Attention/data/wikitext-{variant}",
        f"./data/wikitext-{variant}",
    ]
    
    split_file = f"{split}.txt" if split != "val" else "validation.txt"
    if split == "val":
        split_file = "validation.txt"
    
    for local_dir in local_paths:
        file_path = os.path.join(local_dir, split_file)
        if os.path.exists(file_path):
            print(f"Loading WikiText-{variant} ({split}) from local file: {file_path}")
            with open(file_path, 'r', encoding='utf-8') as f:
                all_text = f.read()
            
            # Tokenize
            print(f"  Tokenizing {len(all_text):,} characters...")
            tokens = tokenizer.encode(all_text, truncation=False, max_length=None)
            tokens = torch.tensor(tokens, dtype=torch.long)
            print(f"  Generated {len(tokens):,} tokens")
            
            return TextDataset(tokens, max_length, stride)
    
    # Fall back to HuggingFace
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Please install datasets: pip install datasets")
    
    dataset_name = f"wikitext-{variant}-raw-v1"
    print(f"Loading {dataset_name} ({split}) from HuggingFace...")
    
    ds = load_dataset("wikitext", dataset_name, split=split, trust_remote_code=True)
    
    # Concatenate all text
    all_text = "\n".join([x["text"] for x in ds if x["text"].strip()])
    
    # Tokenize
    tokens = tokenizer.encode(all_text, truncation=False, max_length=None)
    tokens = torch.tensor(tokens, dtype=torch.long)
    
    return TextDataset(tokens, max_length, stride)


def load_local_wikitext(
    tokenizer,
    split: str = "train",
    max_length: int = 512,
    stride: Optional[int] = None,
    data_dir: str = "/home/lopedg/project/data/data/wikitext-103-small",
) -> TextDataset:
    """
    Load WikiText from local directory.
    
    Args:
        tokenizer: Tokenizer instance
        split: "train", "validation", or "test"
        max_length: Sequence length
        stride: Sliding window stride
        data_dir: Directory containing train.txt, validation.txt, test.txt
    """
    import os
    
    split_file = f"{split}.txt"
    file_path = os.path.join(data_dir, split_file)
    
    print(f"Loading {split} from local file: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        all_text = f.read()
    
    # Tokenize
    print(f"  Tokenizing {len(all_text):,} characters...")
    if hasattr(tokenizer, 'encode'):
        tokens = tokenizer.encode(all_text)
    else:
        # tiktoken
        tokens = tokenizer.encode(all_text)
    tokens = torch.tensor(tokens, dtype=torch.long)
    print(f"  Generated {len(tokens):,} tokens")
    
    return TextDataset(tokens, max_length, stride)


def load_openwebtext(
    tokenizer,
    split: str = "train",
    max_length: int = 512,
    stride: Optional[int] = None,
    max_samples: Optional[int] = None,
) -> TextDataset:
    """
    Load OpenWebText dataset from HuggingFace (~8B tokens).
    
    Note: This is a large dataset. Consider using max_samples for testing.
    Full dataset requires significant disk space and download time.
    
    Args:
        tokenizer: Tokenizer instance
        split: "train" (only split available, we create val/test from it)
        max_length: Sequence length
        stride: Sliding window stride
        max_samples: Limit number of documents to load
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Please install datasets: pip install datasets")
    
    print(f"Loading OpenWebText ({split})...")
    
    # OpenWebText only has train split, we'll create our own splits
    ds = load_dataset("openwebtext", split="train", trust_remote_code=True)
    
    # Create splits: 98% train, 1% val, 1% test
    n = len(ds)
    if split == "train":
        indices = range(0, int(n * 0.98))
    elif split in ("validation", "valid", "val"):
        indices = range(int(n * 0.98), int(n * 0.99))
    else:  # test
        indices = range(int(n * 0.99), n)
    
    # Limit samples if requested
    if max_samples:
        indices = list(indices)[:max_samples]
    
    # Concatenate text
    all_text = "\n\n".join([ds[i]["text"] for i in indices])
    
    # Tokenize
    tokens = tokenizer.encode(all_text, truncation=False, max_length=None)
    tokens = torch.tensor(tokens, dtype=torch.long)
    
    return TextDataset(tokens, max_length, stride)


def load_pg19_static(
    tokenizer,
    split: str = "train",
    max_length: int = 512,
    stride: Optional[int] = None,
    max_samples: Optional[int] = None,
    max_tokens: Optional[int] = None,
) -> TextDataset:
    """
    Load PG-19 with token limit (for validation/test where we need fixed size).
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Please install datasets: pip install datasets")
    
    print(f"Loading PG-19 ({split}) - static mode...")
    
    # Set default token limits for val/test
    if max_tokens is None:
        max_tokens = 10_000_000  # 10M tokens
    
    ds = load_dataset("emozilla/pg19", split=split)
    
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))
    
    print(f"  Loaded {len(ds):,} documents, tokenizing up to {max_tokens:,} tokens...")
    
    all_tokens = []
    for i, doc in enumerate(ds):
        tokens = tokenizer.encode(doc["text"], truncation=False, max_length=None)
        all_tokens.extend(tokens)
        
        if len(all_tokens) >= max_tokens:
            all_tokens = all_tokens[:max_tokens]
            break
    
    tokens = torch.tensor(all_tokens, dtype=torch.long)
    print(f"  Generated {len(tokens):,} tokens")
    
    return TextDataset(tokens, max_length, stride)


class PG19StreamingDataset(IterableDataset):
    """
    Streaming dataset for PG-19 that processes the ENTIRE dataset
    without loading all tokens into memory.
    
    Memory usage: ~100MB buffer (not 20GB+ for full dataset)
    Coverage: ALL 2.8B tokens processed over training
    """
    
    def __init__(
        self,
        tokenizer,
        max_length: int = 1024,
        split: str = "train",
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = split
        
    def __iter__(self):
        from datasets import load_dataset
        
        # Load from cache (memory-mapped, not in RAM)
        ds = load_dataset("emozilla/pg19", split=self.split)
        
        buffer = []
        total_seqs = 0
        
        for doc_idx, doc in enumerate(ds):
            text = doc["text"]
            if not text.strip():
                continue
            
            # Tokenize this document
            tokens = self.tokenizer.encode(text, truncation=False, max_length=None)
            buffer.extend(tokens)
            
            # Yield sequences as buffer fills
            while len(buffer) >= self.max_length + 1:
                seq = buffer[:self.max_length + 1]
                buffer = buffer[self.max_length:]
                total_seqs += 1
                yield {
                    "input_ids": torch.tensor(seq[:-1], dtype=torch.long),
                    "labels": torch.tensor(seq[1:], dtype=torch.long),
                }
            
            # Log progress periodically
            if (doc_idx + 1) % 1000 == 0:
                print(f"  [PG-19] Processed {doc_idx+1} docs, yielded {total_seqs:,} sequences")
        
        # Yield remaining buffer
        while len(buffer) >= self.max_length + 1:
            seq = buffer[:self.max_length + 1]
            buffer = buffer[self.max_length:]
            yield {
                "input_ids": torch.tensor(seq[:-1], dtype=torch.long),
                "labels": torch.tensor(seq[1:], dtype=torch.long),
            }


class StreamingTextDataset(IterableDataset):
    """
    Streaming dataset for very large corpora.
    Tokenizes on-the-fly to avoid memory issues while covering entire dataset.
    
    Key features:
    - Loads documents one at a time from disk/cache
    - Tokenizes incrementally 
    - Never holds more than buffer_size tokens in memory
    - Iterates through ENTIRE dataset
    """
    
    def __init__(
        self,
        dataset_name: str,
        tokenizer,
        max_length: int = 512,
        split: str = "train",
        subset: Optional[str] = None,
        buffer_size: int = 100_000,  # Max tokens to hold in memory
        shuffle_buffer: int = 10_000,  # Shuffle window for randomization
    ):
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = split
        self.subset = subset
        self.buffer_size = buffer_size
        self.shuffle_buffer = shuffle_buffer
        self._total_tokens = 0
        
    def __iter__(self):
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Please install datasets: pip install datasets")
        
        # Load dataset (uses memory-mapped files from cache, not RAM)
        if self.subset:
            ds = load_dataset(
                self.dataset_name,
                self.subset,
                split=self.split,
                trust_remote_code=True,
            )
        else:
            ds = load_dataset(
                self.dataset_name,
                split=self.split,
                trust_remote_code=True,
            )
        
        buffer = []
        self._total_tokens = 0
        
        for example in ds:
            text = example.get("text", example.get("content", ""))
            if not text.strip():
                continue
                
            tokens = self.tokenizer.encode(text, truncation=False, max_length=None)
            buffer.extend(tokens)
            self._total_tokens += len(tokens)
            
            # Yield sequences when buffer is large enough
            while len(buffer) >= self.max_length + 1:
                seq = buffer[:self.max_length + 1]
                buffer = buffer[self.max_length:]
                yield {
                    "input_ids": torch.tensor(seq[:-1], dtype=torch.long),
                    "labels": torch.tensor(seq[1:], dtype=torch.long),
                }
        
        # Yield remaining tokens in buffer
            while len(buffer) >= self.max_length + 1:
                seq = buffer[:self.max_length + 1]
                buffer = buffer[self.max_length:]
                yield {
                    "input_ids": torch.tensor(seq[:-1], dtype=torch.long),
                    "labels": torch.tensor(seq[1:], dtype=torch.long),
                }


def create_dataloaders(
    tokenizer,
    dataset_name: str = "gutenberg",
    max_length: int = 512,
    batch_size: int = 8,
    num_workers: int = 0,
    stride: Optional[int] = None,
    max_train_samples: Optional[int] = None,
    max_eval_samples: Optional[int] = None,
    num_proc: int = 1,  # Ignored (for API compatibility)
    streaming: bool = False,  # Use streaming for very large datasets
) -> Dict[str, DataLoader]:
    """
    Create train/val/test DataLoaders.
    
    Supported datasets:
        - "gutenberg": NLTK classic literature (~2.4M tokens)
        - "wikitext-2": Wikipedia articles (~2M tokens)
        - "wikitext-103": Wikipedia articles (~100M tokens)
        - "pg19": Project Gutenberg books (~2.8B tokens)
        - "openwebtext": Web text (~8B tokens)
        - "slimpajama": Large web corpus, streaming (~627B tokens)
    
    Args:
        tokenizer: Tokenizer instance with encode() method
        dataset_name: Dataset to use
        max_length: Sequence length
        batch_size: Batch size
        num_workers: DataLoader workers
        stride: Sliding window stride
        max_train_samples: Limit training samples
        max_eval_samples: Limit eval samples
        num_proc: Ignored (for API compatibility)
        streaming: Use streaming for very large datasets
        
    Returns:
        Dict with "train", "val", "test" DataLoaders
    """
    
    if dataset_name == "gutenberg":
        print(f"Loading NLTK Gutenberg corpus...")
        train_ds = load_gutenberg(tokenizer, "train", max_length, stride)
        val_ds = load_gutenberg(tokenizer, "validation", max_length, stride)
        test_ds = load_gutenberg(tokenizer, "test", max_length, stride)
        
    elif dataset_name in ("wikitext-2", "wikitext2"):
        train_ds = load_wikitext(tokenizer, "train", max_length, stride, "2")
        val_ds = load_wikitext(tokenizer, "validation", max_length, stride, "2")
        test_ds = load_wikitext(tokenizer, "test", max_length, stride, "2")
        
    elif dataset_name in ("wikitext-103", "wikitext103"):
        train_ds = load_wikitext(tokenizer, "train", max_length, stride, "103")
        val_ds = load_wikitext(tokenizer, "validation", max_length, stride, "103")
        test_ds = load_wikitext(tokenizer, "test", max_length, stride, "103")
    
    elif dataset_name in ("wikitext-103-small", "wikitext103-small", "wiki103-small"):
        # Small subset of WikiText-103 for toy experiments (~11M tokens)
        data_dir = "/home/lopedg/project/data/data/wikitext-103-small"
        train_ds = load_local_wikitext(tokenizer, "train", max_length, stride, data_dir)
        val_ds = load_local_wikitext(tokenizer, "validation", max_length, stride, data_dir)
        test_ds = load_local_wikitext(tokenizer, "test", max_length, stride, data_dir)
        
    elif dataset_name == "openwebtext":
        train_ds = load_openwebtext(tokenizer, "train", max_length, stride, max_train_samples)
        val_ds = load_openwebtext(tokenizer, "validation", max_length, stride, max_eval_samples)
        test_ds = load_openwebtext(tokenizer, "test", max_length, stride, max_eval_samples)
    
    elif dataset_name == "pg19":
        # Use STREAMING for training (entire 2.8B tokens without OOM)
        # Use static loading for val/test (fixed evaluation set)
        print("PG-19: Using streaming for train (full 2.8B tokens)")
        train_ds = PG19StreamingDataset(tokenizer, max_length, "train")
        val_ds = load_pg19_static(tokenizer, "validation", max_length, stride, max_eval_samples, max_tokens=10_000_000)
        test_ds = load_pg19_static(tokenizer, "test", max_length, stride, max_eval_samples, max_tokens=10_000_000)
        
        # Streaming train loader (no shuffle - data is streamed in order)
        loaders = {
            "train": DataLoader(train_ds, batch_size=batch_size, num_workers=0),  # num_workers=0 for IterableDataset
            "val": DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False),
            "test": DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False),
        }
        return loaders
        
    elif dataset_name == "slimpajama" or streaming:
        # Use streaming for very large datasets
        print(f"Loading {dataset_name} in streaming mode...")
        train_ds = StreamingTextDataset(
            "cerebras/SlimPajama-627B",
            tokenizer,
            max_length,
            split="train",
        )
        val_ds = StreamingTextDataset(
            "cerebras/SlimPajama-627B",
            tokenizer,
            max_length,
            split="validation",
        )
        # SlimPajama doesn't have test split, use validation
        test_ds = val_ds
        
        # Streaming datasets don't support length
        loaders = {
            "train": DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers),
            "val": DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers),
            "test": DataLoader(test_ds, batch_size=batch_size, num_workers=num_workers),
        }
        return loaders
        
    else:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Supported: gutenberg, wikitext-2, wikitext-103, pg19, openwebtext, slimpajama"
        )
    
    # Print stats for non-streaming datasets
    if hasattr(train_ds, 'tokens'):
        print(f"  Train: {len(train_ds)} sequences ({len(train_ds.tokens):,} tokens)")
        print(f"  Val:   {len(val_ds)} sequences ({len(val_ds.tokens):,} tokens)")
        print(f"  Test:  {len(test_ds)} sequences ({len(test_ds.tokens):,} tokens)")
    
    # Limit samples if requested
    if max_train_samples and hasattr(train_ds, '__len__') and len(train_ds) > max_train_samples:
        train_ds = torch.utils.data.Subset(train_ds, range(max_train_samples))
    if max_eval_samples:
        if hasattr(val_ds, '__len__') and len(val_ds) > max_eval_samples:
            val_ds = torch.utils.data.Subset(val_ds, range(max_eval_samples))
        if hasattr(test_ds, '__len__') and len(test_ds) > max_eval_samples:
            test_ds = torch.utils.data.Subset(test_ds, range(max_eval_samples))
    
    loaders = {
        "train": DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=False,
        ),
        "val": DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
        ),
        "test": DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
        ),
    }
    
    return loaders
