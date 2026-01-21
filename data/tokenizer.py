"""
Standard GPT-2 tokenizer for academic research.

GPT-2 BPE tokenizer (50257 vocab) is the most common baseline in 
language modeling research, ensuring fair comparisons across experiments.
"""

from typing import Optional, List, Union, Dict, Any
import torch


class Tokenizer:
    """
    GPT-2 tokenizer wrapper for research use.
    
    Uses the standard GPT-2 BPE tokenizer (50257 vocab) which is
    the most common baseline in academic language modeling research.
    
    Example:
        ```python
        from data import Tokenizer
        
        tokenizer = Tokenizer()
        
        # Encode text
        tokens = tokenizer.encode("Hello, world!")
        
        # Decode tokens
        text = tokenizer.decode(tokens)
        
        # Batch encode for training
        batch = tokenizer(
            ["First sequence", "Second sequence"],
            max_length=512,
            padding=True,
            return_tensors="pt",
        )
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
        Initialize GPT-2 tokenizer.
        
        Args:
            max_length: Default max sequence length
            padding_side: Side to pad ("left" or "right")
            truncation_side: Side to truncate ("left" or "right")
        """
        try:
            from transformers import GPT2TokenizerFast
        except ImportError:
            raise ImportError(
                "transformers package required. Install with: pip install transformers"
            )
        
        self._tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        
        # Configure tokenizer
        self._tokenizer.padding_side = padding_side
        self._tokenizer.truncation_side = truncation_side
        
        # GPT-2 doesn't have pad token by default - use EOS
        self._tokenizer.pad_token = self._tokenizer.eos_token
        self._tokenizer.pad_token_id = self._tokenizer.eos_token_id
        
        self.max_length = max_length
    
    @property
    def vocab_size(self) -> int:
        """Vocabulary size (50257 for GPT-2)."""
        return self.VOCAB_SIZE
    
    @property
    def pad_token_id(self) -> int:
        """Padding token ID."""
        return self._tokenizer.pad_token_id
    
    @property
    def eos_token_id(self) -> int:
        """End of sequence token ID."""
        return self._tokenizer.eos_token_id
    
    @property
    def bos_token_id(self) -> int:
        """Beginning of sequence token ID (same as EOS for GPT-2)."""
        return self._tokenizer.bos_token_id
    
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
            add_special_tokens: Add special tokens
            
        Returns:
            List of token IDs
        """
        return self._tokenizer.encode(
            text,
            max_length=max_length or self.max_length,
            truncation=truncation,
            add_special_tokens=add_special_tokens,
        )
    
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
        
        return self._tokenizer.decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
        )
    
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
        
        return dict(self._tokenizer(
            texts,
            max_length=max_length or self.max_length,
            padding=padding,
            truncation=truncation,
            return_tensors=return_tensors,
            return_attention_mask=return_attention_mask,
        ))
    
    def __len__(self) -> int:
        """Return vocabulary size."""
        return self.vocab_size
    
    def __repr__(self) -> str:
        return f"Tokenizer(gpt2, vocab_size={self.vocab_size}, max_length={self.max_length})"
