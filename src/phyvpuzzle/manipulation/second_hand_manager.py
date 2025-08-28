"""
Automatic "second hand" holding system for 3D puzzle manipulation.
"""

import time
import logging
from typing import Dict, List, Optional, Any, Tuple
from functools import lru_cache

# Conditional pybullet import (matching base_env.py pattern)
try:
    import pybullet as p
    PYBULLET_AVAILABLE = True
except ImportError:
    PYBULLET_AVAILABLE = False
    class MockPyBullet:
        JOINT_FIXED = 0
        @staticmethod
        def saveState(*args, **kwargs): return 0
        @staticmethod 
        def restoreState(*args, **kwargs): pass
        @staticmethod
        def createConstraint(*args, **kwargs): return 1
        @staticmethod
        def removeConstraint(*args, **kwargs): pass
        @staticmethod
        def changeConstraint(*args, **kwargs): pass
        @staticmethod
        def stepSimulation(*args, **kwargs): pass
        @staticmethod
        def getBasePositionAndOrientation(*args, **kwargs): return ([0,0,0], [0,0,0,1])
        @staticmethod
        def getContactPoints(*args, **kwargs): return []
        @staticmethod
        def getBaseVelocity(*args, **kwargs): return ([0,0,0], [0,0,0])
    p = MockPyBullet()

from .strategies.base_strategy import HoldingStrategy


class SecondHandManager:
    """
    Automatic "second hand" manager that transparently holds puzzle pieces
    during manipulation to prevent instability and collapse.
    """
    
    def __init__(self, strategy: HoldingStrategy, physics_client: int, config: Dict[str, Any] = None):
        """
        Initialize the second hand manager.
        
        Args:
            strategy: Puzzle-specific holding strategy
            physics_client: PyBullet physics client ID
            config: Configuration parameters
        """
        self.strategy = strategy
        self.physics_client = physics_client
        self.config = config or {}
        
        # State tracking
        self.active_constraints: Dict[int, Dict[str, Any]] = {}
        self.hold_history: List[Dict[str, Any]] = []
        
        # Performance tracking
        self.stats = {
            'total_holds_applied': 0,
            'total_holds_skipped': 0,
            'average_selection_time_ms': 0.0,
            'failed_holds': 0
        }
        
        # Logger
        self.logger = logging.getLogger(f"{__name__}.SecondHandManager")
        
        # Configuration defaults
        self.auto_hold_threshold = self.config.get('auto_hold_threshold', 0.3)
        self.min_piece_count = self.config.get('min_piece_count', 2)
        self.hold_for_actions = set(self.config.get('hold_for_actions', ['move', 'rotate', 'push']))
        self.max_selection_time_ms = self.config.get('max_selection_time_ms', 100)
        self.enable_rollout_validation = self.config.get('enable_rollout_validation', False)
        
    def auto_hold_for_action(self, action: str, target_piece: str,
                             action_params: Dict[str, Any],
                             puzzle_state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Automatically select and apply holding for an action.

        Args:
            action: Type of action being performed
            target_piece: Piece being manipulated
            action_params: Parameters of the action
            puzzle_state: Current puzzle state

        Returns:
            Hold information dict or None if no holding applied
        """
        start_time = time.time()

        try:
            # Quick exit conditions
            if not self.needs_holding(action, target_piece, puzzle_state):
                self.stats['total_holds_skipped'] += 1
                # Include skip in timing stats
                self._update_average_time((time.time() - start_time) * 1000)
                return None

            # Select holding piece using strategy
            hold_piece = self.strategy.select_holding_piece(
                target_piece=target_piece,
                all_pieces=puzzle_state.get('all_pieces', []),
                action_type=action,
                puzzle_state=puzzle_state
            )

            if not hold_piece:
                self.stats['total_holds_skipped'] += 1
                # Include skip in timing stats
                self._update_average_time((time.time() - start_time) * 1000)
                return None

            # Optional: Validate with quick rollout
            if self.enable_rollout_validation:
                if not self._validate_hold_with_rollout(hold_piece, target_piece, action, puzzle_state):
                    self.logger.debug(f"Hold validation failed for {hold_piece}, skipping")
                    self.stats['total_holds_skipped'] += 1
                    # Include skip in timing stats
                    self._update_average_time((time.time() - start_time) * 1000)
                    return None

            # Enforce selection time budget before applying constraint
            selection_time_ms = (time.time() - start_time) * 1000
            if selection_time_ms > self.max_selection_time_ms:
                self.logger.debug(
                    f"Skipping hold due to timeout ({selection_time_ms:.1f}ms > {self.max_selection_time_ms}ms)"
                )
                self.stats['total_holds_skipped'] += 1
                self._update_average_time(selection_time_ms)
                return None

            # Apply soft constraint
            constraint_id = self._apply_soft_constraint(hold_piece, action, puzzle_state)
            
            if constraint_id is None:
                self.stats['failed_holds'] += 1
                return None
            
            # Track the hold
            hold_info = {
                'hold_id': constraint_id,
                'piece': hold_piece,
                'action_type': action,
                'target_piece': target_piece,
                'applied_time': time.time(),
                'constraint_params': self.strategy.get_constraint_params(hold_piece, action)
            }
            
            self.active_constraints[constraint_id] = hold_info
            self.hold_history.append(hold_info.copy())
            self.stats['total_holds_applied'] += 1
            
            # Update performance stats
            selection_time_ms = (time.time() - start_time) * 1000
            self._update_average_time(selection_time_ms)
            
            self.logger.debug(f"Applied automatic hold: {hold_piece} for {action} on {target_piece} "
                            f"(took {selection_time_ms:.1f}ms)")
            
            return {
                'hold_id': constraint_id,
                'piece': hold_piece,
                'type': 'auto',
                'selection_time_ms': selection_time_ms,
                'used': True,
                'mode': 'intelligent',
                'hold_strength': self.strategy.get_hold_strength(hold_piece, action)
            }
            
        except Exception as e:
            self.logger.warning(f"Automatic holding failed for {action} on {target_piece}: {e}")
            self.stats['failed_holds'] += 1
            return None
    
    def auto_release_hold(self, hold_id: int, delay_seconds: float = None) -> bool:
        """
        Automatically release a holding constraint.
        
        Args:
            hold_id: Constraint ID to release
            delay_seconds: Optional delay before release
            
        Returns:
            True if successfully released
        """
        if hold_id not in self.active_constraints:
            self.logger.warning(f"Attempted to release non-existent hold: {hold_id}")
            return False
        
        try:
            # Optional delay for physics settling
            if delay_seconds is None:
                delay_seconds = self.config.get('auto_release_delay', 0.1)
            
            if delay_seconds > 0:
                time.sleep(delay_seconds)
            
            # Remove PyBullet constraint
            if PYBULLET_AVAILABLE:
                p.removeConstraint(hold_id, physicsClientId=self.physics_client)
            
            # Update tracking
            hold_info = self.active_constraints[hold_id]
            hold_duration = time.time() - hold_info['applied_time']
            
            del self.active_constraints[hold_id]
            
            self.logger.debug(f"Released hold {hold_id} on {hold_info['piece']} "
                            f"(held for {hold_duration:.2f}s)")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to release hold {hold_id}: {e}")
            # Force removal from tracking even if PyBullet removal failed
            if hold_id in self.active_constraints:
                del self.active_constraints[hold_id]
            return False
    
    def needs_holding(self, action: str, target_piece: str, puzzle_state: Dict[str, Any]) -> bool:
        """
        Determine if holding is needed for this action.
        
        Args:
            action: Type of action being performed
            target_piece: Piece being manipulated
            puzzle_state: Current puzzle state
            
        Returns:
            True if holding should be applied
        """
        # Check if action type requires holding
        if action not in self.hold_for_actions:
            return False
        
        # Check minimum piece count
        all_pieces = puzzle_state.get('all_pieces', [])
        if len(all_pieces) <= self.min_piece_count:
            return False
        
        # Use strategy-specific logic
        if self.strategy.should_skip_holding(target_piece, puzzle_state):
            return False
        
        # Check if puzzle is already stable (optional)
        if self.config.get('check_stability_first', False):
            stability_score = self._quick_stability_assessment(puzzle_state)
            if stability_score < self.auto_hold_threshold:
                return False
        
        return True
    
    def _apply_soft_constraint(self, piece_name: str, action_type: str, 
                             puzzle_state: Dict[str, Any]) -> Optional[int]:
        """
        Apply a soft constraint to hold a piece in place.
        
        Args:
            piece_name: Name of piece to constrain
            action_type: Type of action being performed
            puzzle_state: Current puzzle state
            
        Returns:
            Constraint ID or None if failed
        """
        if not PYBULLET_AVAILABLE:
            return 1  # Mock constraint ID
        
        try:
            # Get piece information
            piece_objects = puzzle_state.get('piece_objects', {})
            if piece_name not in piece_objects:
                self.logger.warning(f"Piece {piece_name} not found in puzzle state")
                return None
            
            piece_id = piece_objects[piece_name]
            
            # Get current position
            pos, orn = p.getBasePositionAndOrientation(piece_id, physicsClientId=self.physics_client)
            
            # Get constraint parameters from strategy
            constraint_params = self.strategy.get_constraint_params(piece_name, action_type)
            
            # Create soft constraint
            constraint_id = p.createConstraint(
                parentBodyUniqueId=piece_id,
                parentLinkIndex=-1,
                childBodyUniqueId=-1,  # World
                childLinkIndex=-1,
                jointType=p.JOINT_FIXED,
                jointAxis=[0, 0, 0],
                parentFramePosition=[0, 0, 0],
                childFramePosition=pos,
                parentFrameOrientation=[0, 0, 0, 1],
                childFrameOrientation=orn,
                physicsClientId=self.physics_client
            )
            
            # Apply constraint parameters
            p.changeConstraint(
                constraint_id,
                maxForce=constraint_params.get('maxForce', 300.0),
                erp=constraint_params.get('erp', 0.5),
                physicsClientId=self.physics_client
            )

            # Apply optional damping to the piece if provided
            try:
                joint_damping = constraint_params.get('jointDamping', None)
                if joint_damping is not None:
                    # Clamp damping into [0, 1] range
                    damp = max(0.0, min(1.0, float(joint_damping)))
                    p.changeDynamics(piece_id, -1,
                                     linearDamping=damp,
                                     angularDamping=damp,
                                     physicsClientId=self.physics_client)
            except Exception as e:
                self.logger.debug(f"Unable to apply dynamics damping for {piece_name}: {e}")
            
            return constraint_id
            
        except Exception as e:
            self.logger.error(f"Failed to apply constraint to {piece_name}: {e}")
            return None
    
    def _validate_hold_with_rollout(self, hold_piece: str, target_piece: str, 
                                  action: str, puzzle_state: Dict[str, Any]) -> bool:
        """
        Validate holding choice with quick physics rollout.
        
        Args:
            hold_piece: Piece to hold
            target_piece: Piece being manipulated 
            action: Action type
            puzzle_state: Current puzzle state
            
        Returns:
            True if hold appears beneficial
        """
        if not PYBULLET_AVAILABLE:
            return True  # Skip validation in mock mode
        
        try:
            # Save current state
            state_id = p.saveState(physicsClientId=self.physics_client)
            
            # Quick rollout with hold
            constraint_id = self._apply_soft_constraint(hold_piece, action, puzzle_state)
            stability_with_hold = self._run_quick_simulation(steps=24, puzzle_state=puzzle_state)  # ~0.1s at 240Hz
            
            if constraint_id:
                p.removeConstraint(constraint_id, physicsClientId=self.physics_client)
            
            # Restore state and test without hold
            p.restoreState(state_id, physicsClientId=self.physics_client)
            stability_without_hold = self._run_quick_simulation(steps=24, puzzle_state=puzzle_state)
            
            # Restore to original state
            p.restoreState(state_id, physicsClientId=self.physics_client)
            
            # Hold is beneficial if it improves stability
            improvement = stability_without_hold - stability_with_hold
            return improvement > 0.1  # Minimum improvement threshold
            
        except Exception as e:
            self.logger.warning(f"Rollout validation failed: {e}")
            return True  # Default to allowing hold
    
    def _run_quick_simulation(self, steps: int = 24, puzzle_state: Dict[str, Any] = None) -> float:
        """
        Run quick simulation and measure stability.
        
        Args:
            steps: Number of simulation steps
            
        Returns:
            Instability score (lower = more stable)
        """
        total_movement = 0.0
        if puzzle_state is None:
            for _ in range(steps):
                p.stepSimulation(physicsClientId=self.physics_client)
                total_movement += 0.0
            return total_movement

        piece_objects = puzzle_state.get('piece_objects', {})
        physics_client = puzzle_state.get('physics_client', self.physics_client)

        # Capture initial positions
        prev_positions: Dict[str, Tuple[float, float, float]] = {}
        for name, obj_id in piece_objects.items():
            try:
                pos, _ = p.getBasePositionAndOrientation(obj_id, physicsClientId=physics_client)
                prev_positions[name] = tuple(pos)
            except Exception:
                continue
        
        # Step simulation and accumulate movement deltas
        for _ in range(steps):
            p.stepSimulation(physicsClientId=physics_client)
            for name, obj_id in piece_objects.items():
                try:
                    pos, _ = p.getBasePositionAndOrientation(obj_id, physicsClientId=physics_client)
                    if name in prev_positions:
                        px, py, pz = prev_positions[name]
                        dx = pos[0] - px
                        dy = pos[1] - py
                        dz = pos[2] - pz
                        total_movement += (dx*dx + dy*dy + dz*dz) ** 0.5
                        prev_positions[name] = tuple(pos)
                except Exception:
                    continue
        
        return total_movement
    
    def _quick_stability_assessment(self, puzzle_state: Dict[str, Any]) -> float:
        """
        Quick stability assessment without simulation.
        
        Args:
            puzzle_state: Current puzzle state
            
        Returns:
            Stability score (lower = more stable)
        """
        if not PYBULLET_AVAILABLE:
            return 0.5  # Neutral stability
        
        total_velocity = 0.0
        piece_objects = puzzle_state.get('piece_objects', {})
        
        for piece_name, piece_id in piece_objects.items():
            try:
                lin_vel, ang_vel = p.getBaseVelocity(piece_id, physicsClientId=self.physics_client)
                total_velocity += sum(abs(v) for v in lin_vel) + sum(abs(v) for v in ang_vel)
            except:
                continue
        
        return total_velocity
    
    def _update_average_time(self, new_time_ms: float) -> None:
        """Update the rolling average selection time."""
        current_avg = self.stats['average_selection_time_ms']
        total_holds = self.stats['total_holds_applied'] + self.stats['total_holds_skipped']
        
        if total_holds == 0:
            self.stats['average_selection_time_ms'] = new_time_ms
        else:
            # Weighted average
            self.stats['average_selection_time_ms'] = (current_avg * (total_holds - 1) + new_time_ms) / total_holds
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return self.stats.copy()
    
    def release_all_holds(self) -> int:
        """
        Release all active holds (emergency cleanup).
        
        Returns:
            Number of holds released
        """
        released_count = 0
        hold_ids = list(self.active_constraints.keys())
        
        for hold_id in hold_ids:
            if self.auto_release_hold(hold_id, delay_seconds=0):
                released_count += 1
        
        return released_count
    
    def __del__(self):
        """Cleanup: release all holds when manager is destroyed."""
        if hasattr(self, 'active_constraints') and self.active_constraints:
            self.release_all_holds()
