"""
Domino Task Tools

Provides tool definitions and handlers for domino toppling tasks.
"""

from typing import Dict, Any, List, Optional
import pybullet as p
import numpy as np


class DominoTools:
    """Tools for domino manipulation."""
    
    def __init__(self, environment):
        self.environment = environment
        self.domino_names = []
        
    def set_domino_names(self, names: List[str]):
        """Set the list of domino names."""
        self.domino_names = names
        
    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Get tool schemas for the model."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "push_domino",
                    "description": "Push a specific domino to topple it",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "domino_index": {
                                "type": "integer",
                                "description": "Index of the domino to push (0 for first, 1 for second, etc.)",
                                "minimum": 0,
                                "maximum": len(self.domino_names) - 1 if self.domino_names else 10
                            },
                            "force": {
                                "type": "number",
                                "description": "Force magnitude (0.5 to 2.0)",
                                "default": 1.0,
                                "minimum": 0.5,
                                "maximum": 2.0
                            },
                            "direction": {
                                "type": "string",
                                "description": "Direction to push",
                                "enum": ["forward", "backward", "left", "right"],
                                "default": "forward"
                            }
                        },
                        "required": ["domino_index"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "check_dominoes",
                    "description": "Check the status of all dominoes",
                    "parameters": {
                        "type": "object",
                        "properties": {}
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "finish_task",
                    "description": "Indicate that the task is complete",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "reason": {
                                "type": "string",
                                "description": "Reason for finishing",
                                "default": "All dominoes have been toppled"
                            }
                        }
                    }
                }
            }
        ]
    
    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool and return the result."""
        
        if tool_name == "push_domino":
            return self.push_domino(
                domino_index=arguments.get("domino_index", 0),
                force=arguments.get("force", 1.0),
                direction=arguments.get("direction", "forward")
            )
        elif tool_name == "check_dominoes":
            return self.check_dominoes()
        elif tool_name == "finish_task":
            return self.finish_task(arguments.get("reason", "Task completed"))
        else:
            return {"status": "error", "message": f"Unknown tool: {tool_name}"}
    
    def push_domino(self, domino_index: int, force: float = 1.0, direction: str = "forward") -> Dict[str, Any]:
        """Push a specific domino."""
        try:
            if not self.domino_names:
                return {"status": "error", "message": "No dominoes available"}
            
            if domino_index < 0 or domino_index >= len(self.domino_names):
                return {"status": "error", "message": f"Invalid domino index: {domino_index}"}
            
            domino_name = self.domino_names[domino_index]
            
            # Get domino object
            if domino_name not in self.environment.objects:
                return {"status": "error", "message": f"Domino {domino_name} not found"}
            
            domino_info = self.environment.objects[domino_name]
            domino_id = domino_info.object_id
            
            # Calculate force vector based on direction
            force_vectors = {
                "forward": [force, 0, 0],
                "backward": [-force, 0, 0],
                "left": [0, force, 0],
                "right": [0, -force, 0]
            }
            force_vec = force_vectors.get(direction, [force, 0, 0])
            
            # Apply force to the domino
            p.applyExternalForce(
                objectUniqueId=domino_id,
                linkIndex=-1,  # Base link
                forceObj=force_vec,
                posObj=[0, 0, 0.2],  # Apply at middle height
                flags=p.LINK_FRAME
            )
            
            # Step simulation a few times to see effect
            for _ in range(10):
                p.stepSimulation()
            
            return {
                "status": "success",
                "message": f"Pushed domino {domino_index} ({domino_name}) {direction} with force {force}",
                "domino_pushed": domino_index
            }
            
        except Exception as e:
            return {"status": "error", "message": f"Failed to push domino: {str(e)}"}
    
    def check_dominoes(self) -> Dict[str, Any]:
        """Check the status of all dominoes."""
        try:
            statuses = []
            num_toppled = 0
            
            for i, name in enumerate(self.domino_names):
                if name in self.environment.objects:
                    domino_info = self.environment.objects[name]
                    domino_id = domino_info.object_id
                    
                    # Get domino orientation
                    pos, orn = p.getBasePositionAndOrientation(domino_id)
                    euler = p.getEulerFromQuaternion(orn)
                    
                    # Check if domino is toppled (significant tilt)
                    is_toppled = abs(euler[0]) > 0.3 or abs(euler[1]) > 0.3
                    
                    statuses.append({
                        "index": i,
                        "name": name,
                        "toppled": is_toppled,
                        "position": pos,
                        "tilt": max(abs(euler[0]), abs(euler[1]))
                    })
                    
                    if is_toppled:
                        num_toppled += 1
            
            return {
                "status": "success",
                "total_dominoes": len(self.domino_names),
                "toppled_count": num_toppled,
                "all_toppled": num_toppled == len(self.domino_names),
                "dominoes": statuses
            }
            
        except Exception as e:
            return {"status": "error", "message": f"Failed to check dominoes: {str(e)}"}
    
    def finish_task(self, reason: str = "Task completed") -> Dict[str, Any]:
        """Indicate task completion."""
        check_result = self.check_dominoes()
        
        if check_result["status"] == "success":
            all_toppled = check_result.get("all_toppled", False)
            toppled_count = check_result.get("toppled_count", 0)
            total = check_result.get("total_dominoes", 0)
            
            return {
                "status": "finished",
                "reason": reason,
                "task_success": all_toppled,
                "score": toppled_count / total if total > 0 else 0,
                "message": f"Task finished. {toppled_count}/{total} dominoes toppled."
            }
        else:
            return {
                "status": "finished",
                "reason": reason,
                "task_success": False,
                "message": "Task finished but status unknown."
            }