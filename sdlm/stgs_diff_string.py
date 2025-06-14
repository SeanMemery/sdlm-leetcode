import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from .stgs import STGS

class STGSDiffString(nn.Module):
    """
    A class that represents a differentiable string using the STGS (Straight-Through Gumbel-Softmax)
    operation. It maintains both a one-hot encoded tensor representation of tokens and the 
    corresponding string.
    """
    
    def __init__(self, 
                 initial_string: str,
                 tokenizer,
                 temperature: float = 1.0,
                 hard: bool = False,
                 learnable_temperature: bool = False,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the STGSDiffString.
        
        Args:
            initial_string: The initial string to represent
            tokenizer: A tokenizer with encode() and decode() methods
            temperature: Initial temperature for Gumbel-Softmax
            hard: If True, uses straight-through estimator with hard samples
            learnable_temperature: If True, makes temperature a learnable parameter
            device: Device to store tensors on
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.device = device
        
        # Encode the initial string to get token IDs
        self.input_ids = torch.tensor(
            tokenizer.encode(
                initial_string, 
                return_tensors='pt',
                skip_special_tokens=True,
            ).to(device),
            device=device
        )
        
        # Initialize parameters
        self.vocab_size = tokenizer.vocab_size
        self.seq_len = len(self.input_ids[0])
        
        # Initialize logits for the one-hot distribution
        # Start with one-hot encoding of the input_ids
        self.logits = nn.Parameter(
            F.one_hot(self.input_ids, num_classes=self.vocab_size).float().to(device),
            requires_grad=True
        )
        
        # Initialize STGS module
        self.stgs = STGS(
            vocab_size=self.vocab_size,
            stgs_hard=hard,
            init_temperature=temperature,
            learnable_temperature=learnable_temperature,
            device=device
        )
    
    def forward(self) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """
        Forward pass that returns both the differentiable input_ids and the decoded string.
        
        Returns:
            Tuple containing:
                - input_ids: Differentiable token IDs using STGS
                - one_hot: Differentiable one-hot encoding of the input_ids
                - decoded_string: The decoded string from the current distribution
        """
        # Apply STGS to get differentiable samples
        diff_output_ids, diff_one_hot, _, _ = self.stgs(self.logits)
        
        input_ids = diff_output_ids#.long()
        one_hot = diff_one_hot

        # Decode the string
        decoded_string = self.tokenizer.decode(input_ids.long()[0].tolist())

        return input_ids, one_hot, decoded_string
    
    def get_string(self) -> str:
        """Get the current string representation."""
        _, _, decoded_string = self.forward()
        return decoded_string
    
    def get_input_ids(self) -> torch.Tensor:
        """Get the current differentiable input_ids."""
        input_ids, _, _ = self.forward()
        return input_ids
    
    def __str__(self) -> str:
        return self.get_string()
    
    def __len__(self) -> int:
        return self.seq_len
    
    def to(self, device, *args, **kwargs):
        """Move the module to the specified device."""
        super().to(device, *args, **kwargs)
        self.device = device
        self.logits.data = self.logits.data.to(device)
        self.input_ids = self.input_ids.to(device)
        self.stgs = self.stgs.to(device)
        return self
    
    def parameters(self, recurse: bool = True):
        """Return the parameters of the model."""
        return [self.logits]
    
    def state_dict(self, *args, **kwargs):
        """Return the state dictionary."""
        return {'logits': self.logits}
    
    def load_state_dict(self, state_dict, strict: bool = True):
        """Load the state dictionary."""
        self.logits.data.copy_(state_dict['logits'])
        
    def extra_repr(self) -> str:
        return f'seq_len={self.seq_len}, vocab_size={self.vocab_size}, device={self.device}'
