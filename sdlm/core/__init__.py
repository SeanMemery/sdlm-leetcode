"""
sdlm/core/__init__.py

Core components for SDLM - tensor-centric strings with gradient flow.
"""

from .tensor_string import TensorString
from .patching import DSPyPatcher, TensorStringContext
from .differentiable_lm import DifferentiableLM

__all__ = [
    'TensorString',
    'DSPyPatcher', 
    'TensorStringContext',
    'DifferentiableLM'
]