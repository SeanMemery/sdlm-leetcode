"""
Optimization utilities for TextGrad.
"""

from typing import Callable, List, Optional, Dict, Any
import torch
from tqdm import tqdm

from .variables import Variable


def textgrad_optimize(
    loss_fn: Callable[[], torch.Tensor],
    variables: List[Variable],
    optimizer: Optional[torch.optim.Optimizer] = None,
    num_steps: int = 100,
    lr: float = 0.01,
    verbose: bool = True,
    progress_bar: bool = True,
    early_stopping: Optional[float] = None,
    clip_grad_norm: Optional[float] = None,
    log_freq: int = 10,
) -> Dict[str, Any]:
    """
    Optimize text variables using gradient descent.
    
    Args:
        loss_fn: Function that computes the loss (must be callable with no arguments)
        variables: List of Variables to optimize
        optimizer: Optional PyTorch optimizer (defaults to Adam if None)
        num_steps: Number of optimization steps
        lr: Learning rate (if optimizer is None)
        verbose: Whether to print progress
        progress_bar: Whether to show a progress bar
        early_stopping: Stop early if loss falls below this value
        clip_grad_norm: Maximum gradient norm for gradient clipping
        log_freq: Frequency of logging (in steps)
        
    Returns:
        Dictionary containing optimization results
    """
    # Initialize optimizer if not provided
    if optimizer is None:
        params = []
        for var in variables:
            params.append({'params': [var.logits]})
            if hasattr(var, 'temperature_param') and var.temperature_param is not None:
                #params.append({'params': [var.temperature_param], 'lr': lr * 0.1})
                params.append({'params': [var.temperature_param]})
                
        optimizer = torch.optim.Adam(params, lr=lr)
    
    # Track best results
    best_loss = float('inf')
    best_vars = [var.clone() for var in variables]
    losses = []
    
    # Optimization loop
    iterator = range(num_steps)
    if progress_bar:
        iterator = tqdm(iterator, desc="Optimizing")
    
    for step in iterator:
        # Zero gradients
        optimizer.zero_grad()
        
        # Compute loss
        loss = loss_fn()
        
        # Backward pass
        loss.backward()
        
        # Clip gradients if specified
        if clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                [v for var in variables for v in [var.logits] + 
                 ([var.temperature_param] if hasattr(var, 'temperature_param') and var.temperature_param is not None else [])],
                clip_grad_norm
            )
        
        # Optimization step
        optimizer.step()
        
        # Track best variables
        current_loss = loss.item()
        losses.append(current_loss)
        
        if current_loss < best_loss:
            best_loss = current_loss
            best_vars = [var.clone() for var in variables]
        
        # Log progress
        if verbose and (step % log_freq == 0 or step == num_steps - 1):
            log_str = f"Step {step:4d}/{num_steps}: Loss = {current_loss:.6f}"
            if hasattr(variables[0], 'temperature_param') and variables[0].temperature_param is not None:
                temp = variables[0].temperature_param.detach().cpu().item()
                log_str += f" | Temp = {temp:.4f}"
            print(log_str)
            
            if verbose > 1:  # More verbose output
                for i, var in enumerate(variables):
                    print(f"  {var.name}: {var()}")
        
        # Early stopping
        if early_stopping is not None and current_loss <= early_stopping:
            if verbose:
                print(f"Early stopping at step {step} with loss {current_loss:.6f}")
            break
    
    # Restore best variables
    for var, best_var in zip(variables, best_vars):
        var.logits.data.copy_(best_var.logits.data)
        if hasattr(var, 'temperature_param') and var.temperature_param is not None:
            var.temperature_param.data.copy_(best_var.temperature_param.data)
    
    return {
        'best_loss': best_loss,
        'loss_history': losses,
        'final_loss': current_loss,
        'variables': variables
    }
