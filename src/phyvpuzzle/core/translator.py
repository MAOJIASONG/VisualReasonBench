"""
Translator Module

This module handles the translation of action descriptions into 
environment-executable commands using small models.
"""

from typing import Dict, List, Optional, Tuple, Any, Union
import json
import re
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
from .action_descriptor import ParsedAction, ActionType, ActionParameters


@dataclass
class EnvironmentCommand:
    """Command that can be executed in the environment."""
    command_type: str
    parameters: Dict[str, Any]
    execution_order: int = 0
    prerequisites: List[str] = None
    estimated_duration: float = 1.0
    
    def __post_init__(self):
        if self.prerequisites is None:
            self.prerequisites = []


@dataclass
class TranslationResult:
    """Result of action translation."""
    commands: List[EnvironmentCommand]
    success: bool
    error_message: Optional[str] = None
    confidence: float = 1.0


class Translator(ABC):
    """Abstract base class for action translators."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.command_templates = self._initialize_command_templates()
        
    @abstractmethod
    def translate_action(self, action: ParsedAction, 
                        environment_state: Dict[str, Any]) -> TranslationResult:
        """Translate a parsed action into environment commands."""
        pass
    
    def _initialize_command_templates(self) -> Dict[str, Dict[str, Any]]:
        """Initialize command templates for different action types."""
        return {
            ActionType.PICK.value: {
                "command_type": "pick_object",
                "required_params": ["object_id"],
                "optional_params": ["approach_direction", "grip_force"]
            },
            ActionType.PLACE.value: {
                "command_type": "place_object",
                "required_params": ["object_id", "target_position"],
                "optional_params": ["target_orientation", "placement_force"]
            },
            ActionType.MOVE.value: {
                "command_type": "move_object",
                "required_params": ["object_id", "target_position"],
                "optional_params": ["target_orientation", "movement_speed"]
            },
            ActionType.ROTATE.value: {
                "command_type": "rotate_object",
                "required_params": ["object_id", "target_orientation"],
                "optional_params": ["rotation_speed", "rotation_axis"]
            },
            ActionType.PUSH.value: {
                "command_type": "push_object",
                "required_params": ["object_id"],
                "optional_params": ["push_direction", "push_force"]
            },
            ActionType.PULL.value: {
                "command_type": "pull_object",
                "required_params": ["object_id"],
                "optional_params": ["pull_direction", "pull_force"]
            },
            ActionType.STACK.value: {
                "command_type": "stack_object",
                "required_params": ["object_id", "target_object_id"],
                "optional_params": ["alignment", "stacking_precision"]
            },
            ActionType.UNSTACK.value: {
                "command_type": "unstack_object",
                "required_params": ["object_id"],
                "optional_params": ["removal_direction", "separation_force"]
            },
            ActionType.ASSEMBLE.value: {
                "command_type": "assemble_objects",
                "required_params": ["object_id", "target_object_id"],
                "optional_params": ["assembly_type", "connection_points"]
            },
            ActionType.DISASSEMBLE.value: {
                "command_type": "disassemble_objects",
                "required_params": ["object_id"],
                "optional_params": ["disassembly_sequence", "force_threshold"]
            },
            ActionType.FINISH.value: {
                "command_type": "task_complete",
                "required_params": [],
                "optional_params": ["final_state_validation"]
            },
            ActionType.WAIT.value: {
                "command_type": "wait",
                "required_params": [],
                "optional_params": ["duration", "wait_condition"]
            }
        }


class RuleBasedTranslator(Translator):
    """Rule-based translator that uses predefined rules."""
    
    def __init__(self):
        super().__init__("rule_based")
        self.position_resolver = PositionResolver()
        self.object_registry = {}
        
    def translate_action(self, action: ParsedAction, 
                        environment_state: Dict[str, Any]) -> TranslationResult:
        """Translate action using rule-based approach."""
        try:
            template = self.command_templates.get(action.action_type.value)
            if not template:
                return TranslationResult(
                    commands=[],
                    success=False,
                    error_message=f"No template for action type: {action.action_type}"
                )
            
            # Resolve parameters
            resolved_params = self._resolve_parameters(
                action.parameters, 
                environment_state
            )
            
            # Create command
            command = EnvironmentCommand(
                command_type=template["command_type"],
                parameters=resolved_params,
                execution_order=0
            )
            
            # Post-process command based on action type
            commands = self._post_process_command(command, action, environment_state)
            
            return TranslationResult(
                commands=commands,
                success=True,
                confidence=action.confidence
            )
            
        except Exception as e:
            return TranslationResult(
                commands=[],
                success=False,
                error_message=str(e),
                confidence=0.0
            )
    
    def _resolve_parameters(self, params: ActionParameters, 
                          environment_state: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve action parameters to environment-specific values."""
        resolved = {}
        
        if params.object_id:
            resolved["object_id"] = self._resolve_object_id(
                params.object_id, environment_state
            )
        
        if params.target_object_id:
            resolved["target_object_id"] = self._resolve_object_id(
                params.target_object_id, environment_state
            )
        
        if params.target_position:
            resolved["target_position"] = self._resolve_position(
                params.target_position, environment_state
            )
        
        if params.target_orientation:
            resolved["target_orientation"] = params.target_orientation
        
        if params.force:
            resolved["force"] = params.force
        
        if params.duration:
            resolved["duration"] = params.duration
        
        if params.additional_params:
            resolved.update(params.additional_params)
        
        return resolved
    
    def _resolve_object_id(self, object_id: str, 
                          environment_state: Dict[str, Any]) -> str:
        """Resolve object ID to environment-specific identifier."""
        objects = environment_state.get("objects", {})
        
        # Direct match
        if object_id in objects:
            return object_id
        
        # Fuzzy match
        for env_id, obj_info in objects.items():
            if object_id.lower() in obj_info.get("name", "").lower():
                return env_id
            if object_id.lower() in obj_info.get("type", "").lower():
                return env_id
        
        # Return as-is if not found
        return object_id
    
    def _resolve_position(self, position: Tuple[float, float, float], 
                         environment_state: Dict[str, Any]) -> Tuple[float, float, float]:
        """Resolve position to environment coordinates."""
        return self.position_resolver.resolve_position(position, environment_state)
    
    def _post_process_command(self, command: EnvironmentCommand, 
                             action: ParsedAction, 
                             environment_state: Dict[str, Any]) -> List[EnvironmentCommand]:
        """Post-process command based on action type."""
        commands = []
        
        if action.action_type == ActionType.PLACE:
            # For place actions, might need to pick first if not already holding
            if not self._is_object_held(command.parameters.get("object_id"), environment_state):
                pick_command = EnvironmentCommand(
                    command_type="pick_object",
                    parameters={"object_id": command.parameters.get("object_id")},
                    execution_order=0
                )
                commands.append(pick_command)
                command.execution_order = 1
                command.prerequisites = ["pick_object"]
        
        elif action.action_type == ActionType.STACK:
            # For stack actions, ensure both objects are accessible
            target_pos = self._get_object_position(
                command.parameters.get("target_object_id"), 
                environment_state
            )
            if target_pos:
                # Calculate stacking position
                stack_pos = (target_pos[0], target_pos[1], target_pos[2] + 0.1)
                command.parameters["target_position"] = stack_pos
        
        commands.append(command)
        return commands
    
    def _is_object_held(self, object_id: str, environment_state: Dict[str, Any]) -> bool:
        """Check if object is currently held by robot."""
        robot_state = environment_state.get("robot", {})
        held_objects = robot_state.get("held_objects", [])
        return object_id in held_objects
    
    def _get_object_position(self, object_id: str, 
                           environment_state: Dict[str, Any]) -> Optional[Tuple[float, float, float]]:
        """Get object position from environment state."""
        objects = environment_state.get("objects", {})
        obj_info = objects.get(object_id, {})
        return obj_info.get("position")


class LLMTranslator(Translator):
    """LLM-based translator using small language models."""
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-small"):
        super().__init__(model_name)
        self.model = None
        self.tokenizer = None
        
    def load_model(self) -> None:
        """Load the small language model."""
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def translate_action(self, action: ParsedAction, 
                        environment_state: Dict[str, Any]) -> TranslationResult:
        """Translate action using small LLM."""
        if not self.model or not self.tokenizer:
            self.load_model()
        
        try:
            # Create prompt
            prompt = self._create_translation_prompt(action, environment_state)
            
            # Generate translation
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=150,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Parse response into commands
            commands = self._parse_llm_response(response)
            
            return TranslationResult(
                commands=commands,
                success=True,
                confidence=0.8
            )
            
        except Exception as e:
            return TranslationResult(
                commands=[],
                success=False,
                error_message=str(e),
                confidence=0.0
            )
    
    def _create_translation_prompt(self, action: ParsedAction, 
                                 environment_state: Dict[str, Any]) -> str:
        """Create prompt for LLM translation."""
        state_summary = self._summarize_environment_state(environment_state)
        
        prompt = f"""
        Translate the following action into environment commands:
        
        Action: {action.description}
        Action Type: {action.action_type.value}
        Parameters: {asdict(action.parameters)}
        
        Environment State: {state_summary}
        
        Output the commands in JSON format with the following structure:
        {{
            "commands": [
                {{
                    "command_type": "...",
                    "parameters": {{...}},
                    "execution_order": 0
                }}
            ]
        }}
        
        Commands:
        """
        
        return prompt
    
    def _summarize_environment_state(self, environment_state: Dict[str, Any]) -> str:
        """Create a concise summary of environment state."""
        objects = environment_state.get("objects", {})
        robot = environment_state.get("robot", {})
        
        summary = f"Objects: {len(objects)} items, "
        summary += f"Robot position: {robot.get('position', 'unknown')}, "
        summary += f"Held objects: {robot.get('held_objects', [])}"
        
        return summary
    
    def _parse_llm_response(self, response: str) -> List[EnvironmentCommand]:
        """Parse LLM response into environment commands."""
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if not json_match:
                return []
            
            json_str = json_match.group(0)
            data = json.loads(json_str)
            
            commands = []
            for cmd_data in data.get("commands", []):
                command = EnvironmentCommand(
                    command_type=cmd_data.get("command_type", ""),
                    parameters=cmd_data.get("parameters", {}),
                    execution_order=cmd_data.get("execution_order", 0)
                )
                commands.append(command)
            
            return commands
            
        except Exception:
            return []


class PositionResolver:
    """Resolves relative positions to absolute coordinates."""
    
    def __init__(self):
        self.reference_positions = {
            "origin": (0.0, 0.0, 0.0),
            "table_center": (0.0, 0.0, 0.0),
            "workspace_center": (0.0, 0.0, 0.0),
        }
    
    def resolve_position(self, position: Tuple[float, float, float], 
                        environment_state: Dict[str, Any]) -> Tuple[float, float, float]:
        """Resolve position to absolute coordinates."""
        # For now, return position as-is
        # In a real implementation, this would handle coordinate transformations
        return position
    
    def add_reference_position(self, name: str, position: Tuple[float, float, float]):
        """Add a reference position."""
        self.reference_positions[name] = position


def create_translator(translator_type: str = "rule_based", **kwargs) -> Translator:
    """
    Factory function to create translator.
    
    Args:
        translator_type: Type of translator ("rule_based" or "llm")
        **kwargs: Additional arguments for translator
        
    Returns:
        Translator instance
    """
    if translator_type == "rule_based":
        return RuleBasedTranslator(**kwargs)
    elif translator_type == "llm":
        return LLMTranslator(**kwargs)
    else:
        raise ValueError(f"Unknown translator type: {translator_type}") 