"""
TextGrad-style differentiable text optimization for SDLM.

This module provides tools for gradient-based text optimization using SDLM's
differentiable text generation capabilities.
"""

from .variables import Variable
from .losses import TextLoss
from .model import TextGradModel
from .optimizer import textgrad_optimize

__all__ = [
    'Variable',
    'TextLoss',
    'TextGradModel',
    'textgrad_optimize',
]
