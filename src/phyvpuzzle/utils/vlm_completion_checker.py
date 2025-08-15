"""
VLM-based Task Completion Checker

This module provides functionality for using Vision-Language Models to determine
if a task has been completed by comparing target and current states.
"""

from typing import Dict, Any, Optional, Tuple
from PIL import Image
import base64
import io
import json


class VLMCompletionChecker:
    """Uses VLM to check if a task is completed."""
    
    def __init__(self, vllm_processor=None):
        """Initialize the completion checker.
        
        Args:
            vllm_processor: The VLM processor to use for checking completion
        """
        self.vllm_processor = vllm_processor
        
    def check_completion(self,
                        target_image: Optional[Image.Image],
                        current_image: Image.Image,
                        task_description: str,
                        context: Optional[Dict[str, Any]] = None) -> Tuple[bool, str]:
        """Check if task is completed using VLM.
        
        Args:
            target_image: The target/goal state image (optional)
            current_image: The current environment state
            task_description: Description of what needs to be achieved
            context: Additional context about the task
            
        Returns:
            Tuple of (is_completed, explanation)
        """
        if not self.vllm_processor:
            # Fallback to simple completion check
            return False, "No VLM processor available"
        
        # Prepare the prompt
        if target_image:
            prompt = self._create_comparison_prompt(task_description, context)
        else:
            prompt = self._create_single_image_prompt(task_description, context)
        
        # Prepare images for VLM
        images = []
        if target_image:
            images.append(("Target State", target_image))
        images.append(("Current State", current_image))
        
        # Call VLM with tool for completion check
        tools = [{
            "type": "function",
            "function": {
                "name": "check_task_completion",
                "description": "Determine if the task has been completed",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "is_completed": {
                            "type": "boolean",
                            "description": "Whether the task is completed"
                        },
                        "explanation": {
                            "type": "string",
                            "description": "Brief explanation of why the task is/isn't completed"
                        },
                        "confidence": {
                            "type": "number",
                            "description": "Confidence score (0-1)",
                            "minimum": 0,
                            "maximum": 1
                        }
                    },
                    "required": ["is_completed", "explanation", "confidence"]
                }
            }
        }]
        
        try:
            # Process with VLM
            result = self.vllm_processor.process_completion_check(
                images=images,
                prompt=prompt,
                tools=tools
            )
            
            # Parse the response
            if "tool_calls" in result and result["tool_calls"]:
                tool_call = result["tool_calls"][0]
                args = tool_call.get("function", {}).get("arguments", {})
                if isinstance(args, str):
                    args = json.loads(args)
                
                is_completed = args.get("is_completed", False)
                explanation = args.get("explanation", "No explanation provided")
                confidence = args.get("confidence", 0.5)
                
                # Add confidence to explanation if low
                if confidence < 0.7:
                    explanation += f" (Confidence: {confidence:.0%})"
                
                return is_completed, explanation
            else:
                # Parse text response as fallback
                content = result.get("content", "").lower()
                is_completed = any(word in content for word in ["completed", "finished", "done", "success"])
                return is_completed, content[:200]  # Truncate explanation
                
        except Exception as e:
            return False, f"Error checking completion: {str(e)}"
    
    def _create_comparison_prompt(self, task_description: str, context: Optional[Dict[str, Any]]) -> str:
        """Create prompt for comparing target and current states."""
        prompt = f"""You are a task completion checker for a physics simulation environment.

Task: {task_description}

You are shown two images:
1. Target State: The desired end state
2. Current State: The current state of the environment

Please determine if the current state matches or achieves the target state.
Consider the task completed if the essential goal is achieved, even if minor details differ.

"""
        if context:
            if "num_dominoes" in context:
                prompt += f"Context: There are {context['num_dominoes']} dominoes in the scene.\n"
            if "hint" in context:
                prompt += f"Hint: {context['hint']}\n"
        
        prompt += "\nUse the check_task_completion tool to provide your assessment."
        return prompt
    
    def _create_single_image_prompt(self, task_description: str, context: Optional[Dict[str, Any]]) -> str:
        """Create prompt for checking completion with only current state."""
        prompt = f"""You are a task completion checker for a physics simulation environment.

Task: {task_description}

You are shown the current state of the environment.
Please determine if the task has been completed successfully.

"""
        if context:
            if "num_dominoes" in context:
                prompt += f"Context: There are {context['num_dominoes']} dominoes in the scene.\n"
                if context['num_dominoes'] == 1:
                    prompt += "The task is to topple the single domino. Check if it has fallen over.\n"
                else:
                    prompt += f"The task is to topple all {context['num_dominoes']} dominoes. Check if they have all fallen.\n"
        
        prompt += "\nUse the check_task_completion tool to provide your assessment."
        return prompt


def create_target_image_for_dominoes(num_dominoes: int, environment) -> Image.Image:
    """Create a target image showing toppled dominoes.
    
    This function simulates the target state by actually toppling the dominoes
    and capturing an image.
    """
    import pybullet as p
    
    # Apply force to topple dominoes
    for i in range(num_dominoes):
        domino_name = f"domino_{i+1}"
        if domino_name in environment.objects:
            obj = environment.objects[domino_name]
            # Apply a strong lateral force
            p.applyExternalForce(
                objectUniqueId=obj.object_id,
                linkIndex=-1,
                forceObj=[2.0, 0, 0],
                posObj=[0, 0, 0.1],
                flags=p.LINK_FRAME
            )
    
    # Simulate for a bit to let dominoes fall
    for _ in range(100):
        p.stepSimulation()
    
    # Capture the target state
    target_image = environment.render()
    
    # Reset the environment to initial state
    # (This would need to be implemented based on your reset mechanism)
    
    return target_image