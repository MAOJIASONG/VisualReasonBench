"""
Manipulation components for automatic puzzle piece holding and stabilization.
"""

from .second_hand_manager import SecondHandManager
from .strategies.base_strategy import HoldingStrategy

__all__ = ['SecondHandManager', 'HoldingStrategy']