"""
TextGrad model wrapper for SDLM.
"""

from typing import List, Optional, Dict, Any, Union
import torch
import torch.nn as nn

from sdlm.stgs_diff_model import STGSDiffModel
from .variables import Variable


class TextGradModel(STGSDiffModel):
    """
    A model wrapper that supports gradient-based text optimization.
    
    This class extends STGSDiffModel to add support for tracking and optimizing
    TextGradVariables through the generation process.
    """
    
    def __init__(self, *args, **kwargs):
        """
        Initialize the TextGradModel.
        
        Args:
            *args: Positional arguments passed to STGSDiffModel
            **kwargs: Keyword arguments passed to STGSDiffModel
        """
        super().__init__(*args, **kwargs)
        self._grad_vars = []
        
    def register_variable(self, var: Variable) -> Variable:
        """
        Register a variable for gradient tracking.
        
        Args:
            var: Variable to register
            
        Returns:
            The registered variable
        """
        self._grad_vars.append(var)
        return var
        
    def compute_gradients(self, loss: torch.Tensor) -> None:
        """
        Compute gradients for all registered variables.
        
        Args:
            loss: Scalar loss tensor
        """
        if not self._grad_vars:
            return
            
        # Compute gradients for all variables
        grads = torch.autograd.grad(
            loss, 
            [var.logits for var in self._grad_vars],
            create_graph=True,
            allow_unused=True
        )
        
        # Store gradients in variables
        for var, grad in zip(self._grad_vars, grads):
            if grad is not None:
                var.backward(grad)
        
    def optimize_step(self, lr: float = 0.1) -> None:
        """
        Perform one optimization step for all registered variables.
        
        Args:
            lr: Learning rate for the update
        """
        for var in self._grad_vars:
            var.apply_gradient(lr)
    
    def clear_variables(self) -> None:
        """Clear all registered variables."""
        self._grad_vars = []

    def generate(
        self,
        input_one_hot: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        max_length: int = 50,
        num_beams: int = 1,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate text with gradient tracking for registered variables.
        
        Args:
            input_one_hot: Input (differentiable) one-hot representation of tokens
            attention_mask: Attention mask
            max_length: Maximum length of generated sequence
            num_beams: Number of beams for beam search
            **kwargs: Additional arguments for generation
            
        Returns:
            Generated differentiable one-hot representation of the text
        """
        # Clear any previous gradients
        for var in self._grad_vars:
            var.zero_grad()
            
        # Call parent generate
        return super().generate(
            input_one_hot=input_one_hot,
            attention_mask=attention_mask,
            max_length=max_length,
            num_beams=num_beams,
            **kwargs
        )
