"""
Holding strategy for Luban lock puzzles.
"""

import logging
from typing import Dict, List, Optional, Any

# Conditional pybullet import
try:
    import pybullet as p
    PYBULLET_AVAILABLE = True
except ImportError:
    PYBULLET_AVAILABLE = False
    class MockPyBullet:
        @staticmethod
        def getContactPoints(*args, **kwargs): return []
        @staticmethod
        def getBasePositionAndOrientation(*args, **kwargs): return ([0,0,0], [0,0,0,1])
    p = MockPyBullet()

from .base_strategy import HoldingStrategy


class LubanStrategy(HoldingStrategy):
    """
    Holding strategy specifically designed for Luban lock puzzles.
    
    Key principles:
    - Hold "cage" pieces that form the structural framework
    - Prefer pieces with high connectivity (many contacts)
    - In assembly mode: hold base/core pieces
    - In disassembly mode: hold remaining assembled structure
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize Luban strategy with configuration."""
        super().__init__(config)
        
        # Luban-specific configuration
        self.assembly_mode = self.config.get('assembly_mode', True)
        self.contact_weight = self.config.get('contact_weight', 2.0)
        self.distance_weight = self.config.get('distance_weight', 1.0)
        self.core_preference = self.config.get('core_preference', 1.5)
        # New weights/thresholds for improved selection
        self.adjacency_weight = self.config.get('adjacency_weight', 3.0)
        self.zero_adjacency_penalty = self.config.get('zero_adjacency_penalty', -0.5)
        self.velocity_penalty_weight = self.config.get('velocity_penalty_weight', 0.5)
        self.connected_fraction_threshold = self.config.get('connected_fraction_threshold', 0.3)
        
        self.logger = logging.getLogger(f"{__name__}.LubanStrategy")
        # Lightweight cache keyed by simulation step to avoid recomputation
        self._cache = {
            'step': None,
            'contact_counts': None,
            'positions': None,
            'center': None
        }
        
    def select_holding_piece(self, target_piece: str, all_pieces: List[str], 
                           action_type: str, puzzle_state: Dict[str, Any]) -> Optional[str]:
        """
        Select best piece to hold for Luban lock manipulation.
        
        Uses weighted scoring based on:
        1. Contact count (connectivity to other pieces)
        2. Distance from puzzle center of mass
        3. Role in structural framework (core vs peripheral)
        """
        candidates = [p for p in all_pieces if p != target_piece]
        
        if not candidates:
            return None
        
        # Get contact and position information (cached per step)
        contact_counts = self._get_contact_counts(puzzle_state)
        piece_positions = self._get_piece_positions(puzzle_state)
        puzzle_center = self._calculate_puzzle_center(piece_positions, all_pieces)
        # Adjacency (contacts) between target and others
        adjacency_to_target = self._get_neighbors_of_piece(target_piece, puzzle_state)
        # Velocity magnitudes per piece (penalize unstable candidates)
        velocity_mags = self._get_piece_velocity_magnitudes(puzzle_state)
        
        # Score each candidate
        best_piece = None
        best_score = -1
        
        for candidate in candidates:
            score = self._score_candidate(
                candidate=candidate,
                target_piece=target_piece,
                action_type=action_type,
                contact_counts=contact_counts,
                piece_positions=piece_positions,
                puzzle_center=puzzle_center,
                puzzle_state=puzzle_state,
                adjacency_to_target=adjacency_to_target,
                velocity_mags=velocity_mags
            )
            
            if score > best_score:
                best_score = score
                best_piece = candidate
        
        self.logger.debug(f"Selected {best_piece} for holding (score: {best_score:.2f}) "
                         f"during {action_type} of {target_piece}")
        
        return best_piece
    
    def _score_candidate(self, candidate: str, target_piece: str, action_type: str,
                        contact_counts: Dict[str, int], piece_positions: Dict[str, tuple],
                        puzzle_center: tuple, puzzle_state: Dict[str, Any],
                        adjacency_to_target: Dict[str, int],
                        velocity_mags: Dict[str, float]) -> float:
        """Score a candidate piece for holding suitability."""
        score = 0.0
        
        # 1. Contact count score (higher connectivity = better hold)
        contact_count = contact_counts.get(candidate, 0)
        score += contact_count * self.contact_weight
        
        # 2. Distance from center score (closer to center = better structural support)
        if candidate in piece_positions and puzzle_center:
            candidate_pos = piece_positions[candidate]
            distance_to_center = self._distance_3d(candidate_pos, puzzle_center)
            # Inverse distance score (closer = higher score)
            score += (1.0 / (distance_to_center + 0.1)) * self.distance_weight
        
        # 3. Structural role score (core pieces preferred)
        if self._is_core_piece(candidate, contact_counts, puzzle_state):
            score += self.core_preference
        
        # 4. Assembly mode specific adjustments
        if self.assembly_mode:
            # In assembly mode, prefer pieces that are already connected to others
            if contact_count > 0:
                score += 1.0  # Bonus for already-connected pieces
        else:
            # In disassembly mode, prefer pieces that support the most structure
            support_count = self._count_supported_pieces(candidate, contact_counts, puzzle_state)
            score += support_count * 0.5

        # 5. Adjacency to the target piece (direct stabilizing potential)
        adj_contacts = adjacency_to_target.get(candidate, 0)
        if adj_contacts > 0:
            score += adj_contacts * self.adjacency_weight
        else:
            score += self.zero_adjacency_penalty

        # 6. Penalize unstable candidates (moving/rotating fast)
        vel_mag = velocity_mags.get(candidate, 0.0)
        if vel_mag > 0.0:
            score -= vel_mag * self.velocity_penalty_weight

        return score
    
    def _get_contact_counts(self, puzzle_state: Dict[str, Any]) -> Dict[str, int]:
        """Get contact counts for each piece (counts only piece↔piece contacts)."""
        piece_objects = puzzle_state.get('piece_objects', {})
        physics_client = puzzle_state.get('physics_client')

        # Return empty early
        if not piece_objects:
            return {}

        # Cache by step
        step = puzzle_state.get('step_count')
        if self._cache['step'] == step and self._cache['contact_counts'] is not None:
            return dict(self._cache['contact_counts'])

        # Default/mock path
        if not PYBULLET_AVAILABLE:
            counts = {piece: 0 for piece in piece_objects.keys()}
            self._cache['step'] = step
            self._cache['contact_counts'] = dict(counts)
            return counts

        id_to_name = {obj_id: name for name, obj_id in piece_objects.items()}
        counts: Dict[str, int] = {name: 0 for name in piece_objects.keys()}

        for piece_name, piece_id in piece_objects.items():
            try:
                contacts = p.getContactPoints(bodyA=piece_id, physicsClientId=physics_client)
                contacting_objects = set()
                for contact in contacts:
                    other_id = contact[2] if contact[1] == piece_id else contact[1]
                    # Count only contacts with other puzzle pieces
                    if other_id in id_to_name and other_id != piece_id:
                        contacting_objects.add(other_id)
                counts[piece_name] = len(contacting_objects)
            except Exception as e:
                self.logger.warning(f"Failed to get contacts for {piece_name}: {e}")
                counts[piece_name] = counts.get(piece_name, 0)

        self._cache['step'] = step
        self._cache['contact_counts'] = dict(counts)
        return counts
    
    def _get_piece_positions(self, puzzle_state: Dict[str, Any]) -> Dict[str, tuple]:
        """Get current positions of all pieces."""
        piece_objects = puzzle_state.get('piece_objects', {})
        physics_client = puzzle_state.get('physics_client')

        # Cache by step
        step = puzzle_state.get('step_count')
        if self._cache['step'] == step and self._cache['positions'] is not None:
            return dict(self._cache['positions'])

        positions: Dict[str, tuple] = {}
        if not PYBULLET_AVAILABLE:
            positions = {piece: (0.0, 0.0, 0.5) for piece in piece_objects.keys()}
        else:
            for piece_name, piece_id in piece_objects.items():
                try:
                    pos, _ = p.getBasePositionAndOrientation(piece_id, physicsClientId=physics_client)
                    positions[piece_name] = pos
                except Exception as e:
                    self.logger.warning(f"Failed to get position for {piece_name}: {e}")
                    positions[piece_name] = (0.0, 0.0, 0.5)

        self._cache['step'] = step
        self._cache['positions'] = dict(positions)
        # Invalidate center cache; will be recomputed on demand
        self._cache['center'] = None
        return positions
    
    def _calculate_puzzle_center(self, piece_positions: Dict[str, tuple], 
                                all_pieces: List[str]) -> tuple:
        """Calculate center of mass of the puzzle."""
        if not piece_positions:
            return (0.0, 0.0, 0.5)
        
        valid_positions = [piece_positions[piece] for piece in all_pieces 
                          if piece in piece_positions]
        
        if not valid_positions:
            return (0.0, 0.0, 0.5)
        
        # Simple average (assuming equal masses)
        center_x = sum(pos[0] for pos in valid_positions) / len(valid_positions)
        center_y = sum(pos[1] for pos in valid_positions) / len(valid_positions)
        center_z = sum(pos[2] for pos in valid_positions) / len(valid_positions)
        
        return (center_x, center_y, center_z)

    def _get_neighbors_of_piece(self, piece: str, puzzle_state: Dict[str, Any]) -> Dict[str, int]:
        """Return a mapping of neighbor piece name -> number of contacts with the given piece."""
        piece_objects = puzzle_state.get('piece_objects', {})
        physics_client = puzzle_state.get('physics_client')
        neighbors: Dict[str, int] = {}

        if not PYBULLET_AVAILABLE:
            return neighbors

        if piece not in piece_objects:
            return neighbors

        target_id = piece_objects[piece]
        id_to_name = {obj_id: name for name, obj_id in piece_objects.items()}
        try:
            contacts = p.getContactPoints(bodyA=target_id, physicsClientId=physics_client)
            for contact in contacts:
                other_id = contact[2] if contact[1] == target_id else contact[1]
                if other_id in id_to_name and other_id != target_id:
                    other_name = id_to_name[other_id]
                    neighbors[other_name] = neighbors.get(other_name, 0) + 1
        except Exception as e:
            self.logger.debug(f"Failed to get neighbors for {piece}: {e}")

        return neighbors

    def _get_piece_velocity_magnitudes(self, puzzle_state: Dict[str, Any]) -> Dict[str, float]:
        """Get a simple velocity magnitude per piece for stability penalty."""
        piece_objects = puzzle_state.get('piece_objects', {})
        physics_client = puzzle_state.get('physics_client')
        if not PYBULLET_AVAILABLE:
            return {name: 0.0 for name in piece_objects.keys()}

        vmap: Dict[str, float] = {}
        for name, obj_id in piece_objects.items():
            try:
                lin_vel, ang_vel = p.getBaseVelocity(obj_id, physicsClientId=physics_client)
                lv = sum(abs(v) for v in lin_vel)
                av = sum(abs(v) for v in ang_vel)
                vmap[name] = lv + av
            except Exception:
                vmap[name] = 0.0
        return vmap
    
    def _distance_3d(self, pos1: tuple, pos2: tuple) -> float:
        """Calculate 3D distance between two positions."""
        return ((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2 + (pos1[2] - pos2[2])**2)**0.5
    
    def _is_core_piece(self, piece: str, contact_counts: Dict[str, int], 
                      puzzle_state: Dict[str, Any]) -> bool:
        """Determine if a piece is a core structural piece."""
        contact_count = contact_counts.get(piece, 0)
        
        # Core pieces typically have high connectivity
        total_pieces = len(puzzle_state.get('all_pieces', []))
        
        if total_pieces <= 3:
            # For small puzzles, any piece with contacts is core
            return contact_count > 0
        elif total_pieces <= 6:
            # For medium puzzles, pieces with 2+ contacts are core
            return contact_count >= 2
        else:
            # For large puzzles, pieces with 3+ contacts are core
            return contact_count >= 3
    
    def _count_supported_pieces(self, piece: str, contact_counts: Dict[str, int], 
                              puzzle_state: Dict[str, Any]) -> int:
        """Count how many pieces this piece supports (for disassembly mode)."""
        # Simplified implementation - in real scenario would analyze contact graph
        return contact_counts.get(piece, 0)
    
    def _would_interfere_with_motion(self, candidate: str, target_piece: str, 
                                   puzzle_state: Dict[str, Any]) -> bool:
        """Check if holding this candidate would interfere with target motion."""
        # Simplified implementation - would need geometric analysis
        # For now, assume no interference
        return False
    
    def get_hold_strength(self, piece_name: str, action_type: str) -> float:
        """Get appropriate constraint stiffness for Luban lock pieces."""
        # Base strength for wooden pieces
        base_strength = 800.0
        
        # Adjust based on action type
        if action_type == 'move':
            # Medium stiffness for moves
            return base_strength
        elif action_type == 'rotate':
            # Higher stiffness for rotations to prevent unwanted movement
            return base_strength * 1.2
        elif action_type == 'push':
            # Lower stiffness for pushes to allow some give
            return base_strength * 0.8
        
        return base_strength
    
    def should_skip_holding(self, target_piece: str, puzzle_state: Dict[str, Any]) -> bool:
        """Luban-specific logic for skipping holding."""
        # Override for testing: force holding regardless of other conditions
        if self.config.get('force_holding_for_testing', False):
            return False
        
        all_pieces = puzzle_state.get('all_pieces', [])
        
        # Skip for very simple puzzles
        if len(all_pieces) <= 2:
            return True
        
        # Skip if target piece appears to be already free/isolated
        contact_counts = self._get_contact_counts(puzzle_state)
        target_contacts = contact_counts.get(target_piece, 0)
        
        if target_contacts == 0:
            # Piece is already free-floating, probably doesn't need holding
            return True
        
        # In assembly mode, skip if only a few pieces remain unconnected
        if self.assembly_mode:
            # Compute fraction of pieces that have at least one piece↔piece contact
            connected = sum(1 for name in all_pieces if contact_counts.get(name, 0) > 0)
            total = max(1, len(all_pieces))
            connected_fraction = connected / total
            if connected_fraction < self.connected_fraction_threshold:
                return True
        
        return False
    
    def get_constraint_params(self, piece_name: str, action_type: str) -> Dict[str, Any]:
        """Get Luban-specific constraint parameters."""
        base_params = super().get_constraint_params(piece_name, action_type)
        
        # Luban locks benefit from moderate damping
        base_params['jointDamping'] = 0.8
        
        # Assembly mode uses softer constraints to allow natural settling
        if self.assembly_mode:
            base_params['erp'] *= 0.8  # Slightly softer
            base_params['maxForce'] *= 0.9
        
        return base_params
