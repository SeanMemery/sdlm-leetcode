"""
SDLM (Straight-Through Gumbel-Softmax Differentiable Language Modelling)

A library for differentiable text generation using Straight-Through Gumbel-Softmax.
"""

from .stgs import STGS
from .stgs_diff_model import STGSDiffModel
from .stgs_diff_string import STGSDiffString

__version__ = "0.1.0"
__all__ = ["STGS", "STGSDiffModel", "STGSDiffString"]
