"""
Base class for puzzle-specific holding strategies.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any


class HoldingStrategy(ABC):
    """Abstract base class for puzzle-specific holding strategies."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the holding strategy with configuration."""
        self.config = config or {}
        
    @abstractmethod
    def select_holding_piece(self, target_piece: str, all_pieces: List[str], 
                           action_type: str, puzzle_state: Dict[str, Any]) -> Optional[str]:
        """
        Select the best piece to hold during manipulation of target piece.
        
        Args:
            target_piece: Name of piece being manipulated
            all_pieces: List of all available pieces
            action_type: Type of action ("move", "rotate", "push", etc.)
            puzzle_state: Current state information including positions, contacts, etc.
            
        Returns:
            Name of piece to hold, or None if no holding needed
        """
        pass
    
    @abstractmethod
    def get_hold_strength(self, piece_name: str, action_type: str) -> float:
        """
        Get appropriate constraint stiffness for holding this piece.
        
        Args:
            piece_name: Name of piece to hold
            action_type: Type of action being performed
            
        Returns:
            Constraint stiffness value (higher = more rigid)
        """
        pass
    
    def should_skip_holding(self, target_piece: str, puzzle_state: Dict[str, Any]) -> bool:
        """
        Quick check if holding can be skipped for performance.
        
        Args:
            target_piece: Name of piece being manipulated
            puzzle_state: Current puzzle state
            
        Returns:
            True if holding should be skipped
        """
        # Default implementation: skip if very few pieces remain
        total_pieces = len(puzzle_state.get('all_pieces', []))
        return total_pieces <= 2
    
    def get_constraint_params(self, piece_name: str, action_type: str) -> Dict[str, Any]:
        """
        Get constraint parameters for holding this piece.
        
        Args:
            piece_name: Name of piece to hold
            action_type: Type of action being performed
            
        Returns:
            Dictionary of constraint parameters for PyBullet
        """
        base_stiffness = self.get_hold_strength(piece_name, action_type)
        
        return {
            'erp': min(0.8, base_stiffness / 1000.0),  # Error Reduction Parameter
            'cfm': max(0.0001, 1.0 / base_stiffness),  # Constraint Force Mixing  
            'maxForce': min(500.0, base_stiffness * 0.5),  # Maximum constraint force
            'jointDamping': 0.7  # Damping to reduce oscillations
        }
    
    def should_release_hold(self, constraint_info: Dict[str, Any], 
                          current_state: Dict[str, Any]) -> bool:
        """
        Determine if hold should be released based on current conditions.
        
        Args:
            constraint_info: Information about the current constraint
            current_state: Current puzzle state
            
        Returns:
            True if hold should be released
        """
        # Default: release if constraint force exceeds limits
        max_force = constraint_info.get('max_force', 500.0)
        current_force = current_state.get('constraint_force', 0.0)
        
        return current_force > max_force * 1.2  # 20% safety margin