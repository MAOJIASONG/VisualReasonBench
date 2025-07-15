"""
Action Descriptor Module

This module handles the parsing and description of actions from VLLM output.
"""

from typing import Dict, List, Optional, Tuple, Any
import re
from dataclasses import dataclass
from enum import Enum


class ActionType(Enum):
    """Types of actions that can be performed."""
    PICK = "pick"
    PLACE = "place"
    MOVE = "move"
    ROTATE = "rotate"
    PUSH = "push"
    PULL = "pull"
    STACK = "stack"
    UNSTACK = "unstack"
    ASSEMBLE = "assemble"
    DISASSEMBLE = "disassemble"
    FINISH = "finish"
    WAIT = "wait"


@dataclass
class ActionParameters:
    """Parameters for an action."""
    object_id: Optional[str] = None
    target_position: Optional[Tuple[float, float, float]] = None
    target_orientation: Optional[Tuple[float, float, float, float]] = None
    target_object_id: Optional[str] = None
    force: Optional[float] = None
    duration: Optional[float] = None
    additional_params: Optional[Dict[str, Any]] = None


@dataclass
class ParsedAction:
    """Parsed action with type and parameters."""
    action_type: ActionType
    parameters: ActionParameters
    description: str
    confidence: float = 1.0


class ActionDescriptor:
    """
    Parses and describes actions from VLLM output.
    """
    
    def __init__(self):
        self.action_patterns = self._initialize_patterns()
        self.object_extractor = ObjectExtractor()
        
    def _initialize_patterns(self) -> Dict[str, List[str]]:
        """Initialize regex patterns for action recognition."""
        return {
            ActionType.PICK.value: [
                r"pick\s+up\s+(.+)",
                r"grab\s+(.+)",
                r"grasp\s+(.+)",
                r"take\s+(.+)",
                r"lift\s+(.+)",
            ],
            ActionType.PLACE.value: [
                r"place\s+(.+?)\s+(?:on|at|to)\s+(.+)",
                r"put\s+(.+?)\s+(?:on|at|to)\s+(.+)",
                r"set\s+(.+?)\s+(?:on|at|to)\s+(.+)",
                r"position\s+(.+?)\s+(?:on|at|to)\s+(.+)",
            ],
            ActionType.MOVE.value: [
                r"move\s+(.+?)\s+(?:to|towards)\s+(.+)",
                r"shift\s+(.+?)\s+(?:to|towards)\s+(.+)",
                r"slide\s+(.+?)\s+(?:to|towards)\s+(.+)",
            ],
            ActionType.ROTATE.value: [
                r"rotate\s+(.+?)\s+(?:by|to)\s+(.+)",
                r"turn\s+(.+?)\s+(?:by|to)\s+(.+)",
                r"spin\s+(.+?)\s+(?:by|to)\s+(.+)",
            ],
            ActionType.PUSH.value: [
                r"push\s+(.+)",
                r"shove\s+(.+)",
                r"press\s+(.+)",
            ],
            ActionType.PULL.value: [
                r"pull\s+(.+)",
                r"drag\s+(.+)",
                r"draw\s+(.+)",
            ],
            ActionType.STACK.value: [
                r"stack\s+(.+?)\s+(?:on|onto)\s+(.+)",
                r"pile\s+(.+?)\s+(?:on|onto)\s+(.+)",
            ],
            ActionType.UNSTACK.value: [
                r"unstack\s+(.+)",
                r"remove\s+(.+?)\s+from\s+(.+)",
                r"take\s+(.+?)\s+off\s+(.+)",
            ],
            ActionType.ASSEMBLE.value: [
                r"assemble\s+(.+)",
                r"connect\s+(.+?)\s+(?:to|with)\s+(.+)",
                r"attach\s+(.+?)\s+(?:to|with)\s+(.+)",
            ],
            ActionType.DISASSEMBLE.value: [
                r"disassemble\s+(.+)",
                r"disconnect\s+(.+?)\s+from\s+(.+)",
                r"detach\s+(.+?)\s+from\s+(.+)",
            ],
            ActionType.FINISH.value: [
                r"finish",
                r"complete",
                r"done",
                r"finished",
            ],
            ActionType.WAIT.value: [
                r"wait",
                r"pause",
                r"hold",
                r"stop",
            ],
        }
    
    def parse_action(self, action_description: str) -> ParsedAction:
        """
        Parse action description into structured action.
        
        Args:
            action_description: Natural language action description
            
        Returns:
            ParsedAction object
        """
        action_description = action_description.strip().lower()
        
        # Try to match against patterns
        for action_type, patterns in self.action_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, action_description)
                if match:
                    return self._create_parsed_action(
                        ActionType(action_type), 
                        match.groups(), 
                        action_description
                    )
        
        # If no pattern matches, create a generic action
        return ParsedAction(
            action_type=ActionType.MOVE,
            parameters=ActionParameters(),
            description=action_description,
            confidence=0.5
        )
    
    def _create_parsed_action(self, action_type: ActionType, 
                            groups: Tuple[str, ...], 
                            description: str) -> ParsedAction:
        """Create ParsedAction from matched pattern."""
        params = ActionParameters()
        
        if action_type == ActionType.PICK:
            params.object_id = self.object_extractor.extract_object_id(groups[0])
            
        elif action_type == ActionType.PLACE:
            params.object_id = self.object_extractor.extract_object_id(groups[0])
            params.target_object_id = self.object_extractor.extract_object_id(groups[1])
            
        elif action_type == ActionType.MOVE:
            params.object_id = self.object_extractor.extract_object_id(groups[0])
            params.target_position = self.object_extractor.extract_position(groups[1])
            
        elif action_type == ActionType.ROTATE:
            params.object_id = self.object_extractor.extract_object_id(groups[0])
            params.target_orientation = self.object_extractor.extract_orientation(groups[1])
            
        elif action_type in [ActionType.PUSH, ActionType.PULL]:
            params.object_id = self.object_extractor.extract_object_id(groups[0])
            
        elif action_type == ActionType.STACK:
            params.object_id = self.object_extractor.extract_object_id(groups[0])
            params.target_object_id = self.object_extractor.extract_object_id(groups[1])
            
        elif action_type == ActionType.UNSTACK:
            params.object_id = self.object_extractor.extract_object_id(groups[0])
            if len(groups) > 1:
                params.target_object_id = self.object_extractor.extract_object_id(groups[1])
                
        elif action_type == ActionType.ASSEMBLE:
            params.object_id = self.object_extractor.extract_object_id(groups[0])
            if len(groups) > 1:
                params.target_object_id = self.object_extractor.extract_object_id(groups[1])
                
        return ParsedAction(
            action_type=action_type,
            parameters=params,
            description=description,
            confidence=0.9
        )
    
    def describe_action(self, action: ParsedAction) -> str:
        """
        Generate natural language description of an action.
        
        Args:
            action: ParsedAction object
            
        Returns:
            Natural language description
        """
        if action.action_type == ActionType.PICK:
            return f"Pick up {action.parameters.object_id}"
            
        elif action.action_type == ActionType.PLACE:
            return f"Place {action.parameters.object_id} on {action.parameters.target_object_id}"
            
        elif action.action_type == ActionType.MOVE:
            return f"Move {action.parameters.object_id} to {action.parameters.target_position}"
            
        elif action.action_type == ActionType.ROTATE:
            return f"Rotate {action.parameters.object_id} by {action.parameters.target_orientation}"
            
        elif action.action_type == ActionType.PUSH:
            return f"Push {action.parameters.object_id}"
            
        elif action.action_type == ActionType.PULL:
            return f"Pull {action.parameters.object_id}"
            
        elif action.action_type == ActionType.STACK:
            return f"Stack {action.parameters.object_id} on {action.parameters.target_object_id}"
            
        elif action.action_type == ActionType.UNSTACK:
            return f"Unstack {action.parameters.object_id}"
            
        elif action.action_type == ActionType.ASSEMBLE:
            return f"Assemble {action.parameters.object_id} with {action.parameters.target_object_id}"
            
        elif action.action_type == ActionType.DISASSEMBLE:
            return f"Disassemble {action.parameters.object_id}"
            
        elif action.action_type == ActionType.FINISH:
            return "Finish the task"
            
        elif action.action_type == ActionType.WAIT:
            return "Wait"
            
        else:
            return action.description


class ObjectExtractor:
    """Extract object IDs and positions from text."""
    
    def __init__(self):
        self.object_patterns = [
            r"(?:the\s+)?(\w+(?:\s+\w+)*?)(?:\s+(?:block|piece|object|item))?",
            r"object\s+(\d+)",
            r"(\w+)\s+(?:block|piece|object|item)",
        ]
    
    def extract_object_id(self, text: str) -> str:
        """Extract object ID from text."""
        text = text.strip()
        
        # Try patterns
        for pattern in self.object_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip()
        
        # Fallback to the whole text
        return text
    
    def extract_position(self, text: str) -> Optional[Tuple[float, float, float]]:
        """Extract 3D position from text."""
        # Look for coordinate patterns
        coord_pattern = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
        coords = re.findall(coord_pattern, text)
        
        if len(coords) >= 3:
            try:
                return (float(coords[0]), float(coords[1]), float(coords[2]))
            except ValueError:
                pass
        
        # Named positions
        position_map = {
            "left": (-1.0, 0.0, 0.0),
            "right": (1.0, 0.0, 0.0),
            "front": (0.0, 1.0, 0.0),
            "back": (0.0, -1.0, 0.0),
            "up": (0.0, 0.0, 1.0),
            "down": (0.0, 0.0, -1.0),
            "center": (0.0, 0.0, 0.0),
        }
        
        for name, pos in position_map.items():
            if name in text.lower():
                return pos
        
        return None
    
    def extract_orientation(self, text: str) -> Optional[Tuple[float, float, float, float]]:
        """Extract quaternion orientation from text."""
        # Look for degree patterns
        degree_pattern = r"(\d+(?:\.\d+)?)\s*(?:degrees?|°)"
        match = re.search(degree_pattern, text)
        
        if match:
            degrees = float(match.group(1))
            # Convert to radians and create quaternion (simplified)
            import math
            radians = math.radians(degrees)
            # Simple rotation around Z-axis
            return (0.0, 0.0, math.sin(radians/2), math.cos(radians/2))
        
        return None 