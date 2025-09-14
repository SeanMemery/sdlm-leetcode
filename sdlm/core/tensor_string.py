import builtins
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Optional, Union, List, Dict, Iterable

# Store original string class
_original_str = builtins.str

# Early declaration of TensorString to resolve circular dependency
TensorString = None

class TensorString(_original_str):
    """
    Tensor-centric string class where PyTorch tensors are the core representation.
    
    This class can be initialized from either text or tensors, with all operations
    preserving gradient flow through the underlying tensor representation.
    """
    
    # Class-level configuration
    _default_model_name = "distilbert-base-uncased"
    _tokenizer_cache = {}  
    _model_cache = {}      
    _global_device = torch.device('cpu')
    _recursion_guard = False 

    def __new__(
        cls, 
        value: Optional[str] = None, 
        tensor: Optional[torch.Tensor] = None, 
        model_name: Optional[str] = None, 
        attention_mask: Optional[torch.Tensor] = None, 
        device: Optional[torch.device] = None,
    ):
        """
        Create a new TensorString instance.
        
        Args:
            value: String value (optional if tensor provided)
            tensor: PyTorch tensor representation (seq_len, vocab_size) or (seq_len,) for token IDs
            model_name: Model name for tokenization
            attention_mask: Attention mask for the tensor
            device: Device to place tensors on
        """
        if cls._recursion_guard:
            return _original_str.__new__(cls, value or "")

        try: 
            cls._recursion_guard = True 

            if tensor is not None and value is None:
                # Tensor-first initialization: derive string from tensor
                instance = _original_str.__new__(cls, "")  # Placeholder
                instance._init_from_tensor(tensor, model_name, attention_mask, device)
            elif value is not None:
                # String-first initialization: derive tensor from string
                instance = _original_str.__new__(cls, value)
                instance._init_from_string(value, model_name, device)
            else:
                # Empty initialization
                instance = _original_str.__new__(cls, "")
                instance._init_empty(model_name, device)
        finally:
            cls._recursion_guard = False
        return instance
    
    def _init_from_tensor(
        self, 
        tensor: torch.Tensor, 
        model_name: Optional[str] = None, 
        attention_mask: Optional[torch.Tensor] = None, 
        device: Optional[torch.device] = None,
    ):
        """
        Initialize from tensor representation.
        """
        self._model_name = model_name or self._default_model_name
        self._device = device or self._global_device
        
        # Move tensor to specified device
        tensor = tensor.to(self._device)
        
        if tensor.dim() == 1:
            # Token ID tensor - convert to one-hot
            self._core_tensor = F.one_hot(tensor.long(), num_classes=self.vocab_size).float()
            if tensor.requires_grad:
                self._core_tensor.requires_grad_(True)
        elif tensor.dim() == 2:
            # Already one-hot encoded or soft probabilities
            # Maintains gradient flow :
            self._core_tensor = tensor.clone()
            if tensor.requires_grad:
                self._core_tensor.requires_grad_(True)
        else:
            raise ValueError(f"Tensor must be 1D (token IDs) or 2D (one-hot), got shape {tensor.shape}")
        
        self._attention_mask = attention_mask.to(self._device) if attention_mask is not None else None
        self._is_tensor_primary = True
        
        # Derive string representation from tensor
        self._sync_string_from_tensor()
    
    def _init_from_string(
        self, 
        value: str, 
        model_name: Optional[str] = None, 
        device: Optional[torch.device] = None,
    ):
        """
        Initialize from string representation.
        """
        self._model_name = model_name or self._default_model_name
        self._device = device or self._global_device
        self._is_tensor_primary = False
        self._attention_mask = None
        
        # Create tensor representation from string
        self._sync_tensor_from_string()
    
    def _init_empty(
        self, 
        model_name: Optional[str] = None, 
        device: Optional[torch.device] = None,
    ):
        """
        Initialize empty instance.
        """
        self._model_name = model_name or self._default_model_name
        self._device = device or self._global_device
        self._core_tensor = torch.zeros((0, self.vocab_size), requires_grad=True, device=self._device)
        self._attention_mask = None
        self._is_tensor_primary = True
    
    @classmethod
    def configure_model(
        cls, 
        model_name: str,
        model: Optional[AutoModelForCausalLM] = None,
    ) -> None:
        """
        Configure the default model for all TensorString instances.
        
        Args:
            model_name: Name of the model to configure (e.g., 'distilbert-base-uncased')
        """
        cls._default_model_name = model_name
        if model_name not in cls._tokenizer_cache:
            cls._tokenizer_cache[model_name] = AutoTokenizer.from_pretrained(
                model_name, 
                use_fast=True
            )
        if model_name not in cls._model_cache:
            if model is None:
                cls._model_cache[model_name] = AutoModelForCausalLM.from_pretrained(
                    model_name,
                ).to(cls._global_device)
            else:
                cls._model_cache[model_name] = model.to(cls._global_device)
   
    @classmethod
    def set_global_device(
        cls, 
        device: Union[str, torch.device],
    ):
        """
        Set the global device for all new TensorString instances.
        """
        cls._global_device = torch.device(device)
    
    @classmethod
    def clear_model_cache(cls, model_name: Optional[str] = None) -> None:
        """
        Clear the model cache to free up memory.
        
        Args:
            model_name: If provided, only clear this specific model from the cache.
                       If None, clear all models from the cache.
        """
        if model_name is not None:
            if model_name in cls._model_cache:
                del cls._model_cache[model_name]
            if model_name in cls._tokenizer_cache:
                del cls._tokenizer_cache[model_name]
        else:
            cls._model_cache.clear()
            cls._tokenizer_cache.clear()
    
    @classmethod
    def list_cached_models(cls) -> List[str]:
        """
        Get a list of model names currently in the cache.
        
        Returns:
            List of model names that are currently cached
        """
        tokenizer_models = set(cls._tokenizer_cache.keys())
        model_models = set(cls._model_cache.keys())
        return list(tokenizer_models.union(model_models))
    
    @classmethod
    def get_cache_size(cls) -> Dict[str, int]:
        """
        Get the current size of the caches.
        
        Returns:
            Dictionary with 'tokenizer_cache_size' and 'model_cache_size'
        """
        return {
            'tokenizer_cache_size': len(cls._tokenizer_cache),
            'model_cache_size': len(cls._model_cache)
        }
    
    @property
    def tokenizer(self):
        """
        Get the tokenizer for this string's model.
        """
        if self._model_name not in self._tokenizer_cache:
            self._tokenizer_cache[self._model_name] = AutoTokenizer.from_pretrained(
                self._model_name, 
                use_fast=True
            )
            if self._tokenizer_cache[self._model_name].pad_token is None:
                self._tokenizer_cache[self._model_name].pad_token = self._tokenizer_cache[self._model_name].eos_token
        return self._tokenizer_cache[self._model_name]
    
    @classmethod
    def fluency_model(cls):
        """
        Get the fluency model for this string.
        """
        if cls._default_model_name not in cls._model_cache:
            cls._model_cache[cls._default_model_name] = AutoModelForCausalLM.from_pretrained(
                cls._default_model_name,
            ).to(cls._global_device)
        return cls._model_cache[cls._default_model_name]
    
    @classmethod
    def fluency_tokenizer(cls):
        if cls._default_model_name not in cls._tokenizer_cache:
            cls._tokenizer_cache[cls._default_model_name] = AutoTokenizer.from_pretrained(
                cls._default_model_name, 
                use_fast=True
            )
            if cls._tokenizer_cache[cls._default_model_name].pad_token is None:
                cls._tokenizer_cache[cls._default_model_name].pad_token = cls._tokenizer_cache[cls._default_model_name].eos_token
        return cls._tokenizer_cache[cls._default_model_name]

    @property
    def vocab_size(self) -> int:
        """
        Get the vocabulary size of the tokenizer.
        """
        return self.tokenizer.vocab_size
    
    @property
    def tensor(self) -> torch.Tensor:
        """
        Get the core tensor representation with gradient flow.
        """
        return self._core_tensor
    
    @property
    def attention_mask(self) -> Optional[torch.Tensor]:
        """
        Get the attention mask if available.
        """
        return self._attention_mask
    
    @property
    def requires_grad(self) -> bool:
        """
        Check if the tensor requires gradients.
        """
        return self._core_tensor.requires_grad
    
    @property
    def device(self) -> torch.device:
        """
        Get the device of the core tensor.
        """
        return self._core_tensor.device
    
    def requires_grad_(
        self, 
        requires_grad: bool = True,
    ):
        """
        Set gradient requirement for the core tensor.
        """
        self._core_tensor.requires_grad_(requires_grad)
        return self
    
    def to(
        self, 
        device: Union[str, torch.device],
    ):
        """
        Move tensor to specified device.
        """
        device = torch.device(device)
        new_tensor = self._core_tensor.to(device)
        new_attention_mask = self._attention_mask.to(device) if self._attention_mask is not None else None
        
        return TensorString(
            tensor=new_tensor,
            model_name=self._model_name,
            attention_mask=new_attention_mask,
            device=device
        )
    
    def _sync_tensor_from_string(self):
        """
        Update tensor representation from string content.
        """
        if len(_original_str(self)) == 0:
            self._core_tensor = torch.zeros((0, self.vocab_size), requires_grad=True, device=self._device)
            return
            
        tokens = self.tokenizer(
            _original_str(self),
            return_tensors="pt",
            padding=False,
            truncation=False
        )
        
        input_ids = tokens['input_ids'].squeeze(0).to(self._device)
        self._core_tensor = F.one_hot(input_ids, num_classes=self.vocab_size).float()
        self._core_tensor.requires_grad_(True)
        
        self._attention_mask = tokens.get('attention_mask', None)
        if self._attention_mask is not None:
            self._attention_mask = self._attention_mask.squeeze(0).to(self._device)
    
    def _sync_string_from_tensor(self):
        """
        Update string representation from tensor content.
        """
        if self._core_tensor.numel() == 0:
            object.__setattr__(self, '_cached_str', "")
            return
        
        # Convert one-hot/soft to token IDs
        if self._core_tensor.dtype == torch.float:
            # For soft distributions, use argmax
            token_ids = self._core_tensor.argmax(dim=1)
        else:
            # Already token IDs
            token_ids = self._core_tensor
        
        # Decode to string - use detach to avoid affecting gradients
        try:
            decoded = self.tokenizer.decode(token_ids.detach().cpu(), skip_special_tokens=True)
        except:
            # Fallback for invalid token IDs
            decoded = "[INVALID TOKENS]"
        
        # Update the string content without creating new instance
        object.__setattr__(self, '_cached_str', decoded)
    
    def __str__(self):
        """
        String representation.
        """
        if self._is_tensor_primary and hasattr(self, '_cached_str'):
            return self._cached_str
        return super().__str__()
    
    def to_tensor(self) -> torch.Tensor:
        """
        Get the tensor representation (for API compatibility).
        """
        return self._core_tensor
    
    def get_tokens(self) -> list:
        """
        Get the list of tokens for this string.
        """
        if self._is_tensor_primary:
            token_ids = self._core_tensor.argmax(dim=1).detach().cpu()
            return self.tokenizer.convert_ids_to_tokens(token_ids.tolist())
        return self.tokenizer.tokenize(_original_str(self))
    
    def get_token_ids(self) -> torch.Tensor:
        """
        Get the tensor of token IDs.
        """
        return self._core_tensor.argmax(dim=1)
    
    def join(
        self, 
        tensor_strings: Iterable[TensorString],
        separator: Optional[Union[str, TensorString]] = None,
    ) -> TensorString:
        """
        Join multiple TensorString objects with the specified separator.
        
        Args:
            tensor_strings: Iterable of TensorString objects to join
            separator: String or TensorString to use as a separator
            
        Returns:
            A new TensorString containing the joined strings
            
        Example 1 (using TensorString separator):
            >>> ts1 = TensorString("Hello")
            >>> ts2 = TensorString("world")
            >>> result = ts1.join([ts1, ts2], separator=TensorString(" "))
            >>> str(result)
            'Hello world'
        
        Example 2 (using self as separator):
            >>> ts1 = TensorString("Hello")
            >>> ts2 = TensorString("world")
            >>> result = TensorString("\n").join([ts1, ts2])
            >>> str(result)
            'Hello\nworld'
        """
        tensor_strings = [
            ts if isinstance(ts, TensorString) else TensorString(ts) 
            for ts in tensor_strings
        ]
        
        if not tensor_strings:
            return TensorString("")
            
        # Convert separator to TensorString if needed
        separator = self if separator is None else separator
        if not isinstance(separator, TensorString):
            separator = TensorString(
                separator,
                model_name=self._model_name,
                device=self._device,
            )
        
        # Convert all tensor strings to tensors
        tensors = [ts.tensor for ts in tensor_strings]
        
        # Interleave with separator tensors
        joined_tensors = []
        for i, tensor in enumerate(tensors):
            if i > 0:
                joined_tensors.append(separator.tensor)
            joined_tensors.append(tensor)
        
        # Handle attention masks if present
        attention_masks = []
        all_have_masks = all(ts.attention_mask is not None for ts in tensor_strings)
        if all_have_masks and separator.attention_mask is not None:
            for i, ts in enumerate(tensor_strings):
                if i > 0:
                    attention_masks.append(separator.attention_mask)
                attention_masks.append(ts.attention_mask)
            attention_mask = torch.cat(attention_masks, dim=-1)
        else:
            attention_mask = None
        
        # Concatenate tensors along sequence dimension
        if not joined_tensors:
            return TensorString("")
            
        result_tensor = torch.cat(joined_tensors, dim=0)
        
        # Create new TensorString
        ret_ts = TensorString(
            tensor=result_tensor,
            attention_mask=attention_mask,
            model_name=self._model_name,
            device=self._device
        )
        
        return ret_ts

    def __add__(
        self, 
        other: Union[str, TensorString],
    ) -> TensorString:
        """
        Concatenation with gradient flow preservation.
        """
        if isinstance(other, TensorString):
            # Tensor concatenation
            if self._core_tensor.shape[1] != other._core_tensor.shape[1]:
                raise ValueError("Cannot concatenate TensorStrings with different vocab sizes")
            
            concatenated_tensor = torch.cat([self._core_tensor, other._core_tensor], dim=0)
            
            # Concatenate attention masks if both exist
            new_attention_mask = None
            if self._attention_mask is not None and other._attention_mask is not None:
                new_attention_mask = torch.cat([self._attention_mask, other._attention_mask], dim=0)
            elif self._attention_mask is not None:
                new_attention_mask = torch.cat([
                    self._attention_mask, 
                    torch.ones(other._core_tensor.shape[0], dtype=self._attention_mask.dtype, device=self._device)
                ], dim=0)
            elif other._attention_mask is not None:
                new_attention_mask = torch.cat([
                    torch.ones(self._core_tensor.shape[0], dtype=other._attention_mask.dtype, device=self._device),
                    other._attention_mask
                ], dim=0)
            
            return TensorString(
                tensor=concatenated_tensor,
                model_name=self._model_name,
                attention_mask=new_attention_mask,
                device=self._device
            )
        else:
            # Convert other to TensorString first
            other_tensor_str = TensorString(_original_str(other), model_name=self._model_name, device=self._device)
            return self + other_tensor_str
    
    def __radd__(
        self, 
        other: Union[str, TensorString],
    ) -> TensorString:
        """
        Right-side concatenation.
        """
        other_tensor_str = TensorString(_original_str(other), model_name=self._model_name, device=self._device)
        return other_tensor_str + self
    
    def __mul__(
        self, 
        other: int,
    ) -> TensorString:
        """
        Repetition with gradient flow preservation.
        """
        if isinstance(other, int):
            repeated_tensor = self._core_tensor.repeat(other, 1)
            
            new_attention_mask = None
            if self._attention_mask is not None:
                new_attention_mask = self._attention_mask.repeat(other)
            
            return TensorString(
                tensor=repeated_tensor,
                model_name=self._model_name,
                attention_mask=new_attention_mask,
                device=self._device
            )
        return NotImplemented
    
    def __rmul__(
        self, 
        other: int,
    ) -> TensorString:
        """
        Right-side repetition.
        """
        return self.__mul__(other)
    
    # Override string methods to return TensorString instances
    def upper(self):
        """
        Convert to uppercase - returns new TensorString.
        WARNING: this will not preserve gradient flow.
        """
        upper_str = super().upper()
        return TensorString(upper_str, model_name=self._model_name, device=self._device)
    
    def lower(self):
        """
        Convert to lowercase - returns new TensorString.
        WARNING: this will not preserve gradient flow.
        """
        lower_str = super().lower()
        return TensorString(lower_str, model_name=self._model_name, device=self._device)
    
    def strip(
        self, 
        chars: Optional[str] = None,
        preserve_grad: bool = True,
    ) -> TensorString:
        """
        Strip whitespace - returns new TensorString if preserve_grad is False.
        
        Args:
            chars: Characters to strip
            preserve_grad: Whether to preserve gradient flow. If True, then the cached string will
                be updated without creating a new instance. If False, then the cached string will
                be updated with the new string. With respect to the core_tensor, this method
                corresponds to a no-op if preserve_grad is True.
        """
        stripped_str = super().strip(chars)
        #TODO: verify that this won't break everything
        if preserve_grad:
            object.__setattr__(self, '_cached_str', stripped_str)
            return self
        return TensorString(stripped_str, model_name=self._model_name, device=self._device)
    
    def replace(
        self, 
        old: str, 
        new: str, 
        count: int = -1,
        preserve_grad: bool = True,
    ) -> TensorString:
        """
        Replace substring - returns new TensorString if preserve_grad is False.
        
        Args:
            old: Substring to replace
            new: Substring to replace with
            count: Number of occurrences to replace
            preserve_grad: Whether to preserve gradient flow. If True, then the cached string will
                be updated without creating a new instance. If False, then the cached string will
                be updated with the new string. With respect to the core_tensor, this method
                corresponds to a no-op if preserve_grad is True.
        """
        replaced_str = super().replace(old, new, count)
        #TODO: verify that this won't break everything
        if preserve_grad:
            object.__setattr__(self, '_cached_str', replaced_str)
            return self
        return TensorString(replaced_str, model_name=self._model_name, device=self._device)
   
    def gradient_based_replace(
        self, 
        old_pattern: TensorString, 
        new_pattern: TensorString, 
        temperature: float = 1.0,
    ) -> TensorString:
        """
        Replace pattern using differentiable matching.
        """
        if old_pattern._core_tensor.shape[0] == 0:
            return self
        
        pattern_len = old_pattern._core_tensor.shape[0]
        text_len = self._core_tensor.shape[0]
        
        if pattern_len > text_len:
            return self
        
        # Sliding window similarity computation
        similarities = []
        for i in range(text_len - pattern_len + 1):
            window = self._core_tensor[i:i + pattern_len]
            sim = F.cosine_similarity(
                window.flatten().unsqueeze(0),
                old_pattern._core_tensor.flatten().unsqueeze(0)
            )
            similarities.append(sim)
        
        if not similarities:
            return self
        
        similarities = torch.stack(similarities)
        
        # Soft replacement using attention weights
        replacement_weights = F.softmax(similarities / temperature, dim=0)
        
        # Create replacement tensor
        result_parts = []
        used_positions = torch.zeros(text_len, dtype=torch.bool)
        
        for i, weight in enumerate(replacement_weights):
            if weight > 0.1:  # Threshold for replacement
                # Add text before pattern
                if i > 0 and not used_positions[:i].any():
                    result_parts.append(self._core_tensor[:i])
                    used_positions[:i] = True
                
                # Add weighted combination of old and new pattern
                old_weighted = (1 - weight) * self._core_tensor[i:i + pattern_len]
                new_weighted = weight * new_pattern._core_tensor
                
                if new_pattern._core_tensor.shape[0] == pattern_len:
                    combined = old_weighted + new_weighted
                else:
                    # Handle different lengths
                    combined = new_pattern._core_tensor * weight.unsqueeze(0).unsqueeze(1)
                
                result_parts.append(combined)
                used_positions[i:i + pattern_len] = True
        
        # Add remaining text
        remaining_indices = ~used_positions
        if remaining_indices.any():
            remaining_text = self._core_tensor[remaining_indices]
            result_parts.append(remaining_text)
        
        if result_parts:
            result_tensor = torch.cat(result_parts, dim=0)
        else:
            result_tensor = self._core_tensor
        
        return TensorString(
            tensor=result_tensor,
            model_name=self._model_name
        )

    def split(
        self, 
        sep: Optional[str] = None, 
        maxsplit: int = -1,
    ) -> List['TensorString']:
        """
        Split string - returns list of TensorStrings slicing the core_tensor.
        
        Args:
            sep: Separator to split on
            maxsplit: Maximum number of splits
        """
        parts = super().split(sep, maxsplit)
        tokenized_parts = [
            self.tokenizer(
                part, 
                return_tensors='pt', 
                device=self._device, 
                add_special_tokens=False
            ).input_ids 
            for part in parts
        ]
        #TODO: verify that this won't break gradient flow
        slices = []
        start = 0
        for tpidx, tokenized_part in enumerate(tokenized_parts):
            slices.append(self.slice_tokens(start, start + len(tokenized_part)))
            start += len(tokenized_part)
        return slices
    
    def slice_tokens(
        self, 
        start: int, 
        end: Optional[int] = None,
    ) -> TensorString:
        """
        Slice the tensor at token level with gradient preservation.
        """
        sliced_tensor = self._core_tensor[start:end]
        
        sliced_attention_mask = None
        if self._attention_mask is not None:
            sliced_attention_mask = self._attention_mask[start:end]
        
        return TensorString(
            tensor=sliced_tensor,
            model_name=self._model_name,
            attention_mask=sliced_attention_mask,
            device=self._device
        )
    
    def pad_to_length(
        self, 
        length: int, 
        pad_token_id: Optional[int] = None,
        padding_side: Optional[str] = "right",
    ) -> TensorString:
        """
        Pad tensor to specified length with gradient preservation.
        """
        current_length = self._core_tensor.shape[0]
        
        if current_length >= length:
            return self.slice_tokens(0, length)
        
        if pad_token_id is None:
            pad_token_id = self.tokenizer.pad_token_id or 0
        
        pad_length = length - current_length
        pad_tensor = torch.zeros((pad_length, self.vocab_size), device=self._device)
        pad_tensor[:, pad_token_id] = 1.0
        
        # Preserve gradient requirements
        if self._core_tensor.requires_grad:
            pad_tensor.requires_grad_(True)
        
        if padding_side == "right":
            padded_tensor = torch.cat([self._core_tensor, pad_tensor], dim=0) 
        elif padding_side == "left" :
            padded_tensor = torch.cat([pad_tensor, self._core_tensor], dim=0)
        else:
            raise ValueError(f"Invalid padding side: {padding_side}")
        
        # Extend attention mask
        new_attention_mask = None
        if self._attention_mask is not None:
            pad_mask = torch.zeros(pad_length, dtype=self._attention_mask.dtype, device=self._device)
            if padding_side == "right":
                new_attention_mask = torch.cat([self._attention_mask, pad_mask], dim=0)
            elif padding_side == "left":
                new_attention_mask = torch.cat([pad_mask, self._attention_mask], dim=0)
        
        return TensorString(
            tensor=padded_tensor,
            model_name=self._model_name,
            attention_mask=new_attention_mask,
            device=self._device
        )
    
    def apply_softmax_temperature(
        self, 
        temperature: float = 1.0,
    ) -> TensorString:
        """
        Apply softmax with temperature to make the tensor differentiable.
        """
        if temperature <= 0:
            raise ValueError("Temperature must be positive")
        
        # Apply temperature scaling and softmax
        logits = self._core_tensor / temperature
        soft_tensor = F.softmax(logits, dim=1)
        
        return TensorString(
            tensor=soft_tensor,
            model_name=self._model_name,
            attention_mask=self._attention_mask,
            device=self._device
        )
    
    def gumbel_softmax(
        self, 
        temperature: float = 1.0, 
        hard: bool = False,
    ) -> TensorString:
        """
        Apply Gumbel softmax for differentiable discrete sampling.
        """
        # Convert to logits if needed (assume uniform if one-hot)
        if torch.allclose(self._core_tensor.sum(dim=1), torch.ones(self._core_tensor.shape[0], device=self._device)):
            # Already probability distribution, convert to logits
            logits = torch.log(self._core_tensor + 1e-8)
        else:
            logits = self._core_tensor
        
        gumbel_tensor = F.gumbel_softmax(logits, tau=temperature, hard=hard, dim=1)
        
        return TensorString(
            tensor=gumbel_tensor,
            model_name=self._model_name,
            attention_mask=self._attention_mask,
            device=self._device
        )
    
    def similarity_to(
        self, 
        other: TensorString,
    ) -> torch.Tensor:
        """
        Calculate similarity with gradient flow.
        """
        # Ensure same length for comparison
        max_len = max(self._core_tensor.shape[0], other._core_tensor.shape[0])
        self_padded = self.pad_to_length(max_len, padding_side="right")
        other_padded = other.pad_to_length(max_len, padding_side="right")
        
        # Calculate cosine similarity
        return F.cosine_similarity(
            self_padded._core_tensor.flatten().unsqueeze(0),
            other_padded._core_tensor.flatten().unsqueeze(0)
        )
    
    def differentiable_edit_distance(
        self, 
        other: TensorString, 
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Compute differentiable edit distance using soft alignment.
        """
        self_tensor = self._core_tensor
        other_tensor = other._core_tensor
        
        # Compute pairwise similarities
        similarities = torch.matmul(self_tensor, other_tensor.transpose(0, 1))
        
        # Apply temperature and softmax for soft alignment
        soft_alignment = F.softmax(similarities / temperature, dim=1)
        
        # Compute alignment score (higher = more similar)
        alignment_score = (soft_alignment * similarities).sum()
        
        # Convert to distance (lower = more similar)
        max_possible_score = min(self_tensor.shape[0], other_tensor.shape[0])
        distance = max_possible_score - alignment_score
        
        return distance
    
    def __repr__(self) -> str:
        """
        Enhanced representation showing tensor information.
        """
        grad_info = " (requires_grad)" if self.requires_grad else ""
        device_info = f" on {self.device}" if self.device != torch.device('cpu') else ""
        return f"TensorString('{_original_str(self)}', shape={self._core_tensor.shape}, model='{self._model_name}'{grad_info}{device_info})"
