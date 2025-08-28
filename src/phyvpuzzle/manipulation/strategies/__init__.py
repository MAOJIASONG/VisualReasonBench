"""
Holding strategies for different puzzle types.
"""

from .base_strategy import HoldingStrategy
from .luban_strategy import LubanStrategy

__all__ = ['HoldingStrategy', 'LubanStrategy']