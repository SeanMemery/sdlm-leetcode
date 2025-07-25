# ==========================================
# Factory Functions
# ==========================================

from typing import Optional, List
import torch
import torch.nn.functional as F
import builtins
from transformers import AutoTokenizer

from ..core.tensor_string import TensorString

# Store original string class
_original_str = builtins.str

def TensorStr(
    value=None, 
    tensor=None, 
    model_name=None, 
    requires_grad=False, 
    device=None,
) -> TensorString:
    """
    Factory function to create TensorString instances.
    
    Args:
        value: String value
        tensor: PyTorch tensor representation
        model_name: Model name to use
        requires_grad: Whether tensor should require gradients
        device: Device to place tensor on
        
    Returns:
        TensorString: Enhanced tensor-centric string
    """
    result = TensorString(value=value, tensor=tensor, model_name=model_name, device=device)
    if requires_grad:
        result.requires_grad_(True)
    return result

def from_string(text: str, model_name: Optional[str] = None, 
                requires_grad: bool = False, device: Optional[torch.device] = None) -> TensorString:
    """
    Create a TensorString from text.
    
    Args:
        text: Input text string
        model_name: Model to use for tokenization (uses default if None)
        requires_grad: Whether the tensor should require gradients
        device: Device to place tensor on (uses default if None)
        
    Returns:
        TensorString: Tensor-centric string with gradient capabilities
    """
    result = create_tensor_string_safe(text, model_name, device)
    if requires_grad:
        result.requires_grad_(True)
    return result

def from_tensor(tensor: torch.Tensor, model_name: Optional[str] = None, 
                requires_grad: Optional[bool] = None, 
                device: Optional[torch.device] = None) -> TensorString:
    """
    Create a TensorString from a PyTorch tensor.
    
    Args:
        tensor: Input tensor (1D token IDs or 2D one-hot/soft probabilities)
        model_name: Model to use for decoding (uses default if None)
        requires_grad: Whether to enable gradients (preserves tensor setting if None)
        device: Device to place tensor on (preserves tensor device if None)
        
    Returns:
        TensorString: Tensor-centric string
    """
    if requires_grad is not None:
        tensor = tensor.clone().requires_grad_(requires_grad)
    
    if device is not None:
        tensor = tensor.to(device)
    
    return create_tensor_string_from_tensor_safe(tensor, model_name)

def from_token_ids(token_ids: List[int], model_name: Optional[str] = None, 
                   requires_grad: bool = False, 
                   device: Optional[torch.device] = None) -> TensorString:
    """
    Create a TensorString from token IDs.
    
    Args:
        token_ids: List of token IDs
        model_name: Model to use (uses default if None)
        requires_grad: Whether tensor should require gradients
        device: Device to place tensor on (uses default if None)
        
    Returns:
        TensorString: Tensor-centric string
    """
    device = device or _sdlm_instance.device
    tensor = torch.tensor(token_ids, dtype=torch.long, device=device)
    
    if requires_grad:
        # Convert to one-hot for gradient support
        vocab_size = 50257  # Default, will be updated when TensorString is created
        tensor = F.one_hot(tensor, num_classes=vocab_size).float()
        tensor.requires_grad_(True)
    
    return create_tensor_string_from_tensor_safe(tensor, model_name)

def empty_tensor_string(length: int = 0, model_name: Optional[str] = None, 
                       requires_grad: bool = True, 
                       device: Optional[torch.device] = None) -> TensorString:
    """
    Create an empty TensorString for gradient-based generation.
    
    Args:
        length: Sequence length (0 for empty)
        model_name: Model to use (uses default if None)
        requires_grad: Whether tensor should require gradients
        device: Device to place tensor on (uses default if None)
        
    Returns:
        TensorString: Empty learnable string
    """
    device = device or _sdlm_instance.device
    model_name = model_name or _sdlm_instance.default_model
    
    # Create temporary instance to get vocab size
    temp_instance = create_tensor_string_safe("", model_name, device)
    vocab_size = temp_instance.vocab_size
    
    if length > 0:
        # Create random initialization
        tensor = torch.randn(length, vocab_size, requires_grad=requires_grad, device=device)
        tensor = F.softmax(tensor, dim=1)  # Initialize as probability distribution
    else:
        tensor = torch.zeros(0, vocab_size, requires_grad=requires_grad, device=device)
    
    return create_tensor_string_from_tensor_safe(tensor, model_name)
