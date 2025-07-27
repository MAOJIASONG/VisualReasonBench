"""
Success Metrics and Evaluation for PhyVPuzzle Environments

This module provides comprehensive success detection and evaluation metrics
for different puzzle types in the PhyVPuzzle benchmark.
"""

import numpy as np
import pybullet as p
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from enum import Enum
import math

from .phyvpuzzle_env import PuzzleType, PuzzleObjectConfig


class SuccessLevel(Enum):
    """Levels of success in puzzle solving."""
    FAILED = 0
    PARTIAL = 1
    COMPLETE = 2


@dataclass 
class SuccessMetrics:
    """Comprehensive success metrics for puzzle evaluation."""
    success_level: SuccessLevel
    overall_score: float  # 0.0 to 1.0
    component_scores: Dict[str, float]
    detailed_analysis: Dict[str, Any]
    reasoning: str
    
    def is_success(self) -> bool:
        """Check if the task is considered successful."""
        return self.success_level == SuccessLevel.COMPLETE
    
    def is_partial_success(self) -> bool:
        """Check if partial success was achieved."""
        return self.success_level == SuccessLevel.PARTIAL


class PuzzleSuccessEvaluator:
    """Evaluator for puzzle success detection and scoring."""
    
    def __init__(self, puzzle_type: PuzzleType):
        self.puzzle_type = puzzle_type
        
    def evaluate_success(self, puzzle_objects: Dict[str, PuzzleObjectConfig],
                        environment_state: Dict[str, Any]) -> SuccessMetrics:
        """
        Evaluate success based on puzzle type and current state.
        
        Args:
            puzzle_objects: Dictionary of puzzle objects
            environment_state: Current environment state
            
        Returns:
            SuccessMetrics with detailed evaluation
        """
        if self.puzzle_type == PuzzleType.LUBAN_LOCK:
            return self._evaluate_luban_lock_success(puzzle_objects, environment_state)
        elif self.puzzle_type == PuzzleType.PAGODA:
            return self._evaluate_pagoda_success(puzzle_objects, environment_state)
        else:
            raise ValueError(f"Unknown puzzle type: {self.puzzle_type}")
    
    def _evaluate_luban_lock_success(self, puzzle_objects: Dict[str, PuzzleObjectConfig],
                                   environment_state: Dict[str, Any]) -> SuccessMetrics:
        """Evaluate Luban lock puzzle success."""
        component_scores = {}
        detailed_analysis = {}
        
        # 1. Interlocking Analysis
        interlock_score, interlock_details = self._analyze_luban_interlocking(puzzle_objects)
        component_scores["interlocking"] = interlock_score
        detailed_analysis["interlocking"] = interlock_details
        
        # 2. Spatial Configuration Analysis  
        config_score, config_details = self._analyze_luban_configuration(puzzle_objects)
        component_scores["configuration"] = config_score
        detailed_analysis["configuration"] = config_details
        
        # 3. Stability Analysis
        stability_score, stability_details = self._analyze_luban_stability(puzzle_objects)
        component_scores["stability"] = stability_score
        detailed_analysis["stability"] = stability_details
        
        # 4. Assembly Completeness
        completeness_score, completeness_details = self._analyze_luban_completeness(puzzle_objects)
        component_scores["completeness"] = completeness_score
        detailed_analysis["completeness"] = completeness_details
        
        # Calculate overall score with weights
        weights = {
            "interlocking": 0.4,
            "configuration": 0.3,
            "stability": 0.2,
            "completeness": 0.1
        }
        
        overall_score = sum(weights[k] * v for k, v in component_scores.items())
        
        # Determine success level
        if overall_score >= 0.9:
            success_level = SuccessLevel.COMPLETE
            reasoning = "Luban lock fully assembled with proper interlocking"
        elif overall_score >= 0.6:
            success_level = SuccessLevel.PARTIAL
            reasoning = "Luban lock partially assembled, some pieces properly interlocked"
        else:
            success_level = SuccessLevel.FAILED
            reasoning = "Luban lock not properly assembled"
        
        return SuccessMetrics(
            success_level=success_level,
            overall_score=overall_score,
            component_scores=component_scores,
            detailed_analysis=detailed_analysis,
            reasoning=reasoning
        )
    
    def _analyze_luban_interlocking(self, puzzle_objects: Dict[str, PuzzleObjectConfig]) -> Tuple[float, Dict]:
        """Analyze interlocking quality of Luban pieces."""
        object_ids = [obj.object_id for obj in puzzle_objects.values() if obj.movable]
        
        if len(object_ids) < 2:
            return 0.0, {"error": "Insufficient objects for interlocking analysis"}
        
        # Count interlocking contacts
        interlocking_pairs = 0
        total_possible_pairs = len(object_ids) * (len(object_ids) - 1) // 2
        contact_details = []
        
        for i in range(len(object_ids)):
            for j in range(i + 1, len(object_ids)):
                obj1_id = object_ids[i]
                obj2_id = object_ids[j]
                
                # Get contact points
                contacts = p.getContactPoints(obj1_id, obj2_id)
                
                if contacts:
                    # Analyze contact quality
                    contact_quality = self._analyze_contact_quality(contacts)
                    if contact_quality > 0.5:  # Threshold for meaningful contact
                        interlocking_pairs += 1
                        contact_details.append({
                            "obj1": i,
                            "obj2": j,
                            "contact_points": len(contacts),
                            "quality": contact_quality
                        })
        
        # Calculate interlocking score
        expected_interlocks = max(1, len(object_ids) - 2)  # Expect most pieces to interlock
        interlock_ratio = min(1.0, interlocking_pairs / expected_interlocks)
        
        details = {
            "interlocking_pairs": interlocking_pairs,
            "total_objects": len(object_ids),
            "expected_interlocks": expected_interlocks,
            "interlock_ratio": interlock_ratio,
            "contact_details": contact_details
        }
        
        return interlock_ratio, details
    
    def _analyze_contact_quality(self, contacts: List) -> float:
        """Analyze the quality of contact between two objects."""
        if not contacts:
            return 0.0
        
        # Consider contact normal forces and penetration depth
        total_force = 0.0
        for contact in contacts:
            normal_force = contact[9]  # Normal force
            total_force += abs(normal_force)
        
        # Normalize force (this is a heuristic)
        quality_score = min(1.0, total_force / (len(contacts) * 10.0))
        return quality_score
    
    def _analyze_luban_configuration(self, puzzle_objects: Dict[str, PuzzleObjectConfig]) -> Tuple[float, Dict]:
        """Analyze spatial configuration of Luban pieces."""
        positions = []
        for obj in puzzle_objects.values():
            if obj.movable:
                pos, _ = p.getBasePositionAndOrientation(obj.object_id)
                positions.append(pos)
        
        if len(positions) < 2:
            return 0.0, {"error": "Insufficient objects for configuration analysis"}
        
        # Calculate center of mass
        center_of_mass = np.mean(positions, axis=0)
        
        # Calculate compactness (pieces should be close together)
        distances = [np.linalg.norm(np.array(pos) - center_of_mass) for pos in positions]
        avg_distance = np.mean(distances)
        max_distance = np.max(distances)
        
        # Good configuration has low spread
        compactness_score = max(0.0, 1.0 - (avg_distance / 2.0))  # Normalize by expected range
        
        # Check for proper alignment (pieces should be somewhat aligned)
        alignment_score = self._calculate_alignment_score(positions)
        
        # Combined configuration score
        config_score = 0.6 * compactness_score + 0.4 * alignment_score
        
        details = {
            "center_of_mass": center_of_mass.tolist(),
            "avg_distance_from_center": avg_distance,
            "max_distance_from_center": max_distance,
            "compactness_score": compactness_score,
            "alignment_score": alignment_score,
            "positions": positions
        }
        
        return config_score, details
    
    def _calculate_alignment_score(self, positions: List) -> float:
        """Calculate alignment score based on position distribution."""
        if len(positions) < 3:
            return 1.0
        
        positions = np.array(positions)
        
        # Principal component analysis to find main alignment axis
        centered = positions - np.mean(positions, axis=0)
        cov_matrix = np.cov(centered.T)
        eigenvalues, _ = np.linalg.eigh(cov_matrix)
        
        # Alignment score based on eigenvalue ratio
        # High ratio means points are well-aligned along principal axis
        eigenvalues = np.sort(eigenvalues)[::-1]  # Sort descending
        if eigenvalues[1] > 0:
            alignment_ratio = eigenvalues[0] / eigenvalues[1]
            alignment_score = min(1.0, alignment_ratio / 10.0)  # Normalize
        else:
            alignment_score = 1.0
        
        return alignment_score
    
    def _analyze_luban_stability(self, puzzle_objects: Dict[str, PuzzleObjectConfig]) -> Tuple[float, Dict]:
        """Analyze stability of Luban assembly."""
        velocities = []
        
        for obj in puzzle_objects.values():
            if obj.movable:
                lin_vel, ang_vel = p.getBaseVelocity(obj.object_id)
                total_vel = np.linalg.norm(lin_vel) + np.linalg.norm(ang_vel)
                velocities.append(total_vel)
        
        if not velocities:
            return 1.0, {"note": "No movable objects to analyze"}
        
        max_velocity = max(velocities)
        avg_velocity = np.mean(velocities)
        
        # Stability score - lower velocities = higher stability
        stability_threshold = 0.1  # m/s or rad/s
        stability_score = max(0.0, 1.0 - (avg_velocity / stability_threshold))
        
        details = {
            "max_velocity": max_velocity,
            "avg_velocity": avg_velocity,
            "stability_threshold": stability_threshold,
            "individual_velocities": velocities,
            "is_stable": max_velocity < stability_threshold
        }
        
        return stability_score, details
    
    def _analyze_luban_completeness(self, puzzle_objects: Dict[str, PuzzleObjectConfig]) -> Tuple[float, Dict]:
        """Analyze completeness of Luban assembly."""
        total_pieces = len([obj for obj in puzzle_objects.values() if obj.movable])
        
        # Check if pieces are in reasonable positions (not fallen off table, etc.)
        valid_pieces = 0
        piece_positions = []
        
        for obj in puzzle_objects.values():
            if obj.movable:
                pos, _ = p.getBasePositionAndOrientation(obj.object_id)
                piece_positions.append(pos)
                
                # Check if piece is in valid range (not fallen off table)
                if pos[2] > -0.5:  # Above ground level with some tolerance
                    valid_pieces += 1
        
        if total_pieces == 0:
            return 1.0, {"note": "No pieces to evaluate"}
        
        completeness_score = valid_pieces / total_pieces
        
        details = {
            "total_pieces": total_pieces,
            "valid_pieces": valid_pieces,
            "invalid_pieces": total_pieces - valid_pieces,
            "piece_positions": piece_positions,
            "completeness_ratio": completeness_score
        }
        
        return completeness_score, details
    
    def _evaluate_pagoda_success(self, puzzle_objects: Dict[str, PuzzleObjectConfig],
                                environment_state: Dict[str, Any]) -> SuccessMetrics:
        """Evaluate Pagoda puzzle success."""
        component_scores = {}
        detailed_analysis = {}
        
        # 1. Structural Stability
        stability_score, stability_details = self._analyze_pagoda_stability(puzzle_objects)
        component_scores["stability"] = stability_score
        detailed_analysis["stability"] = stability_details
        
        # 2. Height Achievement  
        height_score, height_details = self._analyze_pagoda_height(puzzle_objects)
        component_scores["height"] = height_score
        detailed_analysis["height"] = height_details
        
        # 3. Balance Quality
        balance_score, balance_details = self._analyze_pagoda_balance(puzzle_objects)
        component_scores["balance"] = balance_score
        detailed_analysis["balance"] = balance_details
        
        # 4. Symmetry
        symmetry_score, symmetry_details = self._analyze_pagoda_symmetry(puzzle_objects)
        component_scores["symmetry"] = symmetry_score
        detailed_analysis["symmetry"] = symmetry_details
        
        # Calculate overall score
        weights = {
            "stability": 0.4,
            "height": 0.3,
            "balance": 0.2,
            "symmetry": 0.1
        }
        
        overall_score = sum(weights[k] * v for k, v in component_scores.items())
        
        # Determine success level
        if overall_score >= 0.85:
            success_level = SuccessLevel.COMPLETE
            reasoning = "Pagoda tower successfully built with good stability and height"
        elif overall_score >= 0.6:
            success_level = SuccessLevel.PARTIAL
            reasoning = "Pagoda tower partially built but lacks full stability or height"
        else:
            success_level = SuccessLevel.FAILED
            reasoning = "Pagoda tower construction failed"
        
        return SuccessMetrics(
            success_level=success_level,
            overall_score=overall_score,
            component_scores=component_scores,
            detailed_analysis=detailed_analysis,
            reasoning=reasoning
        )
    
    def _analyze_pagoda_stability(self, puzzle_objects: Dict[str, PuzzleObjectConfig]) -> Tuple[float, Dict]:
        """Analyze structural stability of pagoda."""
        pagoda_obj = None
        for obj in puzzle_objects.values():
            if obj.object_type == "pagoda":
                pagoda_obj = obj
                break
        
        if not pagoda_obj:
            return 0.0, {"error": "No pagoda object found"}
        
        # Check velocity - stable structure should have low velocity
        lin_vel, ang_vel = p.getBaseVelocity(pagoda_obj.object_id)
        total_velocity = np.linalg.norm(lin_vel) + np.linalg.norm(ang_vel)
        
        # Check if structure is falling
        pos, orn = p.getBasePositionAndOrientation(pagoda_obj.object_id)
        euler = p.getEulerFromQuaternion(orn)
        tilt_angle = max(abs(euler[0]), abs(euler[1]))  # Roll and pitch
        
        # Stability metrics
        velocity_stability = max(0.0, 1.0 - (total_velocity / 0.5))  # Threshold 0.5
        tilt_stability = max(0.0, 1.0 - (tilt_angle / (math.pi / 6)))  # 30 degree threshold
        
        stability_score = 0.6 * velocity_stability + 0.4 * tilt_stability
        
        details = {
            "total_velocity": total_velocity,
            "tilt_angle_rad": tilt_angle,
            "tilt_angle_deg": math.degrees(tilt_angle),
            "velocity_stability": velocity_stability,
            "tilt_stability": tilt_stability,
            "position": pos,
            "orientation_euler": euler
        }
        
        return stability_score, details
    
    def _analyze_pagoda_height(self, puzzle_objects: Dict[str, PuzzleObjectConfig]) -> Tuple[float, Dict]:
        """Analyze height achievement of pagoda."""
        pagoda_obj = None
        for obj in puzzle_objects.values():
            if obj.object_type == "pagoda":
                pagoda_obj = obj
                break
        
        if not pagoda_obj:
            return 0.0, {"error": "No pagoda object found"}
        
        pos, _ = p.getBasePositionAndOrientation(pagoda_obj.object_id)
        current_height = pos[2]
        
        # Expected height range (this would be calibrated based on puzzle design)
        min_height = 0.5
        target_height = 2.0
        
        if current_height >= target_height:
            height_score = 1.0
        elif current_height >= min_height:
            height_score = (current_height - min_height) / (target_height - min_height)
        else:
            height_score = 0.0
        
        details = {
            "current_height": current_height,
            "min_height": min_height,
            "target_height": target_height,
            "height_ratio": current_height / target_height if target_height > 0 else 0
        }
        
        return height_score, details
    
    def _analyze_pagoda_balance(self, puzzle_objects: Dict[str, PuzzleObjectConfig]) -> Tuple[float, Dict]:
        """Analyze balance quality of pagoda."""
        # This would involve analyzing center of mass, support points, etc.
        # For now, simplified based on position and orientation
        
        pagoda_obj = None
        for obj in puzzle_objects.values():
            if obj.object_type == "pagoda":
                pagoda_obj = obj
                break
        
        if not pagoda_obj:
            return 0.0, {"error": "No pagoda object found"}
        
        pos, orn = p.getBasePositionAndOrientation(pagoda_obj.object_id)
        
        # Check if center is over base (simplified)
        center_offset = math.sqrt(pos[0]**2 + pos[1]**2)
        balance_threshold = 0.5  # Acceptable offset from center
        
        balance_score = max(0.0, 1.0 - (center_offset / balance_threshold))
        
        details = {
            "center_offset": center_offset,
            "balance_threshold": balance_threshold,
            "position": pos,
            "is_balanced": center_offset < balance_threshold
        }
        
        return balance_score, details
    
    def _analyze_pagoda_symmetry(self, puzzle_objects: Dict[str, PuzzleObjectConfig]) -> Tuple[float, Dict]:
        """Analyze symmetry of pagoda structure."""
        # Simplified symmetry analysis
        # In a full implementation, this would analyze the geometric arrangement
        
        pagoda_obj = None
        for obj in puzzle_objects.values():
            if obj.object_type == "pagoda":
                pagoda_obj = obj
                break
        
        if not pagoda_obj:
            return 0.0, {"error": "No pagoda object found"}
        
        # For now, assume good symmetry if structure is well-balanced and stable
        # This is a placeholder for more sophisticated geometric analysis
        symmetry_score = 0.8  # Default reasonable symmetry
        
        details = {
            "note": "Symmetry analysis not fully implemented",
            "assumed_symmetry": symmetry_score
        }
        
        return symmetry_score, details


def evaluate_puzzle_success(puzzle_type: PuzzleType,
                          puzzle_objects: Dict[str, PuzzleObjectConfig],
                          environment_state: Dict[str, Any]) -> SuccessMetrics:
    """
    Convenience function to evaluate puzzle success.
    
    Args:
        puzzle_type: Type of puzzle to evaluate
        puzzle_objects: Dictionary of puzzle objects
        environment_state: Current environment state
        
    Returns:
        SuccessMetrics with detailed evaluation
    """
    evaluator = PuzzleSuccessEvaluator(puzzle_type)
    return evaluator.evaluate_success(puzzle_objects, environment_state)