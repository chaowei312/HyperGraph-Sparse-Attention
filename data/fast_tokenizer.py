"""
Fast tiktoken-based tokenizer for academic research.

tiktoken is 10-100x faster than transformers and used in GPT-3/4.
Perfect for demos and academic benchmarking.
"""

from typing import Optional, List, Union, Dict, Any
import torch


class FastTokenizer:
    """
    Fast GPT-2 compatible tokenizer using tiktoken.
    
    ~10-100x faster than transformers.GPT2TokenizerFast
    Perfect for academic demos where speed matters.
    
    Example:
        ```python
        from data.tokenizer_fast import FastTokenizer
        
        tokenizer = FastTokenizer()
        
        # Encode text
        tokens = tokenizer.encode("Hello, world!")
        
        # Decode tokens
        text = tokenizer.decode(tokens)
        ```
    """
    
    VOCAB_SIZE = 50257  # GPT-2 vocabulary size
    
    def __init__(
        self,
        max_length: int = 2048,
        padding_side: str = "right",
        truncation_side: str = "right",
    ):
        """
        Initialize fast tokenizer.
        
        Args:
            max_length: Default max sequence length
            padding_side: Side to pad ("left" or "right")
            truncation_side: Side to truncate ("left" or "right")
        """
        try:
            import tiktoken
        except ImportError:
            raise ImportError(
                "tiktoken package required. Install with: pip install tiktoken"
            )
        
        # Use GPT-2 encoding (same vocab as transformers GPT2)
        self._tokenizer = tiktoken.get_encoding("gpt2")
        
        self.max_length = max_length
        self.padding_side = padding_side
        self.truncation_side = truncation_side
        
        # GPT-2 special tokens
        self._eos_token_id = 50256
        self._bos_token_id = 50256
        self._pad_token_id = 50256
    
    @property
    def vocab_size(self) -> int:
        """Vocabulary size (50257 for GPT-2)."""
        return self.VOCAB_SIZE
    
    @property
    def pad_token_id(self) -> int:
        """Padding token ID."""
        return self._pad_token_id
    
    @property
    def eos_token_id(self) -> int:
        """End of sequence token ID."""
        return self._eos_token_id
    
    @property
    def bos_token_id(self) -> int:
        """Beginning of sequence token ID."""
        return self._bos_token_id
    
    def encode(
        self,
        text: str,
        max_length: Optional[int] = None,
        truncation: bool = True,
        add_special_tokens: bool = True,
    ) -> List[int]:
        """
        Encode text to token IDs.
        
        Args:
            text: Input text
            max_length: Maximum length (uses default if None)
            truncation: Whether to truncate
            add_special_tokens: Add special tokens (ignored for compatibility)
            
        Returns:
            List of token IDs
        """
        tokens = self._tokenizer.encode(text, allowed_special="all")
        
        if truncation and max_length:
            max_len = max_length or self.max_length
            if self.truncation_side == "right":
                tokens = tokens[:max_len]
            else:
                tokens = tokens[-max_len:]
        
        return tokens
    
    def decode(
        self,
        token_ids: Union[List[int], torch.Tensor],
        skip_special_tokens: bool = True,
    ) -> str:
        """
        Decode token IDs to text.
        
        Args:
            token_ids: Token IDs (list or tensor)
            skip_special_tokens: Skip special tokens in output
            
        Returns:
            Decoded text
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        
        return self._tokenizer.decode(token_ids)
    
    def __call__(
        self,
        texts: Union[str, List[str]],
        max_length: Optional[int] = None,
        padding: Union[bool, str] = True,
        truncation: bool = True,
        return_tensors: Optional[str] = "pt",
        return_attention_mask: bool = True,
    ) -> Dict[str, Any]:
        """
        Tokenize text(s) for model input.
        
        Args:
            texts: Single text or list of texts
            max_length: Maximum sequence length
            padding: Padding strategy (True, "longest", "max_length")
            truncation: Whether to truncate
            return_tensors: Return type ("pt" for PyTorch, None for lists)
            return_attention_mask: Include attention mask
            
        Returns:
            Dictionary with input_ids, attention_mask
        """
        if isinstance(texts, str):
            texts = [texts]
        
        max_len = max_length or self.max_length
        
        # Encode all texts
        all_ids = [self.encode(text, max_len, truncation) for text in texts]
        
        # Pad if needed
        if padding:
            if padding == "longest" or padding is True:
                target_len = max(len(ids) for ids in all_ids)
            else:  # "max_length"
                target_len = max_len
            
            pad_id = self.pad_token_id
            for i, ids in enumerate(all_ids):
                pad_len = target_len - len(ids)
                if pad_len > 0:
                    if self.padding_side == "right":
                        all_ids[i] = ids + [pad_id] * pad_len
                    else:
                        all_ids[i] = [pad_id] * pad_len + ids
        
        result = {}
        
        if return_tensors == "pt":
            result["input_ids"] = torch.tensor(all_ids, dtype=torch.long)
            if return_attention_mask:
                result["attention_mask"] = (result["input_ids"] != self.pad_token_id).long()
        else:
            result["input_ids"] = all_ids
            if return_attention_mask:
                result["attention_mask"] = [[1 if id != self.pad_token_id else 0 for id in ids] for ids in all_ids]
        
        return result
    
    def __len__(self) -> int:
        """Return vocabulary size."""
        return self.vocab_size
    
    def __repr__(self) -> str:
        return f"FastTokenizer(tiktoken-gpt2, vocab_size={self.vocab_size}, max_length={self.max_length})"

