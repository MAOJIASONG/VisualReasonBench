"""
Domino task implementation for falling domino puzzles.
"""

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Set, Tuple

from phyvpuzzle.core import (TaskConfig, register_task,
                             register_task_config)
from phyvpuzzle.core.base import (Action, ObjectInfo, State, TaskDifficulty,
                                  TaskResult, Observation)
from phyvpuzzle.environment.domino_env import DominoEnvironment, p
from phyvpuzzle.tasks.base_task import PhysicsTask


@register_task_config("domino_dont_fall")
@dataclass
class DominoDontFallTaskConfig(TaskConfig):
    """Configuration for domino task."""
    num_dominoes: int = 5
    domino_spacing: float = 0.08  # Distance between dominoes
    domino_height: float = 0.05
    arrangement_pattern: str = "line"
    ruled_evaluation: bool = False

@register_task("domino_dont_fall")
class DominoDontFallTask(PhysicsTask):
    """Task for domino falling chain reaction puzzles."""
    
    def __init__(self, config: TaskConfig):
        super().__init__(config)

    def _calculate_optimal_steps(self) -> int:
        """Calculate optimal steps for domino task."""
        # Basic domino tasks typically require:
        # 1. Observe initial setup
        # 2. Push first domino  
        # 3. Wait for chain reaction
        # 4. Check solution
        
        if self.config.difficulty == TaskDifficulty.EASY:
            return 4
        elif self.config.difficulty == TaskDifficulty.MEDIUM:
            return 6
        elif self.config.difficulty == TaskDifficulty.HARD:
            return 8
        else:
            return 4  # Default to 4 if difficulty is not specified
            
    def _configure_environment(self) -> None:
        """Configure environment for domino task."""
        # Ensure we have a domino environment
        if not isinstance(self.environment, DominoEnvironment):
            raise ValueError("DominoTask requires DominoEnvironment")
        
        self._load_domino_models()
        self.environment.dominoes = [obj for obj in self.environment.objects if obj.object_type == "domino"]
        self._arrange_dominoes()

    def _load_domino_models(self) -> None:
        """Load domino URDF models."""
        domino_base_path = os.path.join(self.environment.config.urdf_local_path, "domino")
        
        if not os.path.exists(domino_base_path):
            print(f"Warning: Domino models not found at {domino_base_path}")
            print("Creating simple domino shapes instead")
            self._create_simple_dominoes()
            return
            
        # Load available domino URDF files
        available_dominoes = []
        if os.path.exists(domino_base_path):
            for item in os.listdir(domino_base_path):
                domino_dir = os.path.join(domino_base_path, item)
                urdf_path = os.path.join(domino_dir, "urdf", f"{item}.urdf")
                if os.path.isdir(domino_dir) and os.path.exists(urdf_path):
                    available_dominoes.append(urdf_path)
                
        if not available_dominoes:
            print("No domino URDF files found, creating simple dominoes")
            self._create_simple_dominoes()
            return
            
        # Select dominoes to use
        num_to_load = min(self.config.num_dominoes, len(available_dominoes))
        selected_dominoes = available_dominoes[:num_to_load]
        
        print(f"Loading {len(selected_dominoes)} dominoes from URDF files")
        
        # Avoid initial overlap
        for i, domino_path in enumerate(selected_dominoes):
            domino_id = f"domino_{i+1}"
            # Stagger temporary spawn positions to prevent immediate collisions
            temp_x = (i - (num_to_load - 1) / 2) * (self.config.domino_spacing * 2.0)
            temp_z = 0.5 + 0.01 * i
            self.environment.add_object(
                object_name=domino_id,
                urdf_path=domino_path,
                position=(temp_x, 0, temp_z),
                orientation=(0, 0, 0, 1),
                object_type='domino',
                properties={'index': i}
            )
                
        # If we didn't load enough dominoes, create simple ones for the rest
        if len(self.environment.objects) < self.config.num_dominoes:
            remaining = self.config.num_dominoes - len(self.environment.objects)
            print(f"Creating {remaining} simple dominoes to reach target count")
            self._create_simple_dominoes(start_index=len(self.environment.objects))
            
    def _create_simple_dominoes(self, start_index: int = 0) -> None:
        """Create simple domino shapes using primitive objects."""
        domino_width = 0.02
        domino_length = 0.04
        domino_height = self.config.domino_height
        
        num_to_create = self.config.num_dominoes - start_index
        
        for i in range(num_to_create):
            domino_id = f"domino_{start_index + i + 1}"
            
            # Create collision shape
            collision_shape = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=[domino_width/2, domino_length/2, domino_height/2]
            )
            
            # Create visual shape
            visual_shape = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[domino_width/2, domino_length/2, domino_height/2],
                rgbaColor=[0.8, 0.4, 0.2, 1.0]  # Brown color
            )
            
            # Create domino body
            # Stagger temporary spawn positions to prevent immediate collisions
            temp_x = (start_index + i) * (self.config.domino_spacing * 2.0)
            temp_z = 0.5 + 0.01 * (start_index + i)
            obj_id = p.createMultiBody(
                baseMass=0.1,
                baseCollisionShapeIndex=collision_shape,
                baseVisualShapeIndex=visual_shape,
                basePosition=[temp_x, 0, temp_z]  # Temporary non-overlapping position
            )

            p.changeDynamics(obj_id, -1, lateralFriction=1.0, spinningFriction=0.002, rollingFriction=0.002, linearDamping=0.04, angularDamping=0.04)
            
            self.environment.objects.append(ObjectInfo(
                object_id=obj_id,
                name=domino_id,
                position=[temp_x, 0, temp_z],
                orientation=[0, 0, 0, 1],
                object_type='domino',
                properties={'index': start_index + i}
            ))

    def _arrange_dominoes(self) -> None:
        """Arrange dominoes according to the specified pattern."""
        positions = self._calculate_domino_positions()
        
        for i, obj in enumerate(self.environment.dominoes):
            if i < len(positions):
                pos, orient = positions[i]

                # Set domino position and orientation
                p.resetBasePositionAndOrientation(obj.object_id, pos, orient)
                p.resetBaseVelocity(obj.object_id, [0, 0, 0], [0, 0, 0])
                
                # Update object info
                obj.position = pos
                obj.orientation = orient

        # Wait for dominoes to settle
        for _ in range(5):
            p.stepSimulation()
                
        print(f"Arranged {len(positions)} dominoes in {self.config.arrangement_pattern} pattern")
        
    def _calculate_domino_positions(self) -> List[Tuple[List[float], List[float]]]:
        """Calculate positions and orientations for dominoes based on arrangement pattern."""
        positions = []
        spacing = self.config.domino_spacing
        table_height = self.environment.config.table_position[2] + 0.02  # Slightly above table
        
        if self.config.arrangement_pattern == "line":
            # Simple line arrangement
            for i in range(len(self.environment.objects)):
                x = i * spacing - (len(self.environment.objects) - 1) * spacing / 2
                pos = [x, 0, table_height + self.config.domino_height / 2]
                orient = [0, 0, 0, 1]  # No rotation
                positions.append((pos, orient))
                
        elif self.config.arrangement_pattern == "curve":
            # Curved arrangement (arc)
            import math
            radius = len(self.environment.objects) * spacing / (2 * math.pi) * 2
            
            for i in range(len(self.environment.objects)):
                angle = i * math.pi / len(self.environment.objects)
                x = radius * math.cos(angle)
                y = radius * math.sin(angle)
                pos = [x, y, table_height + self.config.domino_height/2]
                
                # Orient domino to face the center
                yaw = angle + math.pi/2
                orient = p.getQuaternionFromEuler([0, 0, yaw])
                positions.append((pos, orient))
                
        elif self.config.arrangement_pattern == "zigzag":
            # Zigzag pattern
            import math
            
            for i in range(len(self.environment.objects)):
                x = i * spacing - (len(self.environment.objects) - 1) * spacing / 2
                y = 0.1 * math.sin(i * math.pi / 3)  # Zigzag with amplitude 0.1
                pos = [x, y, table_height + self.config.domino_height/2]
                
                # Slight rotation based on zigzag direction
                yaw = math.sin(i * math.pi / 3) * 0.2
                orient = p.getQuaternionFromEuler([0, 0, yaw])
                positions.append((pos, orient))
                
        elif self.config.arrangement_pattern == "circle":
            # Circular arrangement
            import math
            radius = len(self.environment.objects) * spacing / (2 * math.pi)
            
            for i in range(len(self.environment.objects)):
                angle = i * 2 * math.pi / len(self.environment.objects)
                x = radius * math.cos(angle)
                y = radius * math.sin(angle)
                pos = [x, y, table_height + self.config.domino_height/2]
                
                # Orient domino tangent to circle
                yaw = angle + math.pi/2
                orient = p.getQuaternionFromEuler([0, 0, yaw])
                positions.append((pos, orient))
                
        return positions
      
    def _evaluate_success(self, task_results: List[TaskResult]) -> List[TaskResult]:
        """Return success criteria description for LLM judge evaluation."""
        if self.config.ruled_evaluation:
            return self._evaluate_success_ruled(task_results)
        else:
            return self._evaluate_success_agent(task_results)

    def _evaluate_success_ruled(self, task_results: List[TaskResult]) -> List[TaskResult]:
        for task_result in task_results:
            fallen_dominoes, fallen_count = self._find_fallen_dominoes(task_result.trajectory)
            print(f"Fallen dominoes: {fallen_dominoes}")
            print(f"Fallen count: {fallen_count}")
            if fallen_count >= self.config.num_dominoes * 0.8:
                task_result.success = True
            else:
                task_result.success = False
        return task_results

    def _find_fallen_dominoes(self, trajectory: List[Tuple[Action, Observation]]) -> Tuple[Set[str], int]:
        """Find fallen dominoes."""
        fallen_count = 0
        fallen_dominoes = set()
        for obj in trajectory[-1][-1].objects:  # (action, tool_response, resulting_state)
            orient = obj.orientation
            
            # Convert quaternion to euler angles
            euler = p.getEulerFromQuaternion(orient)
            
            # Check if domino is tilted significantly (fallen)
            tilt_threshold = 0.5  # radians (about 28 degrees)
            
            if abs(euler[0]) > tilt_threshold or abs(euler[1]) > tilt_threshold:
                fallen_count += 1
                fallen_dominoes.add(obj.name)
                
        return fallen_dominoes, fallen_count

    def _evaluate_success_agent(self, task_results: List[TaskResult]) -> List[TaskResult]:
        num_dominoes = self.config.num_dominoes
        arrangement = self.config.arrangement_pattern
        difficulty = self.config.difficulty.value

        task_success_criteria = f"""
DOMINO FALLING TASK SUCCESS EVALUATION

Task Setup:
- Total dominoes: {num_dominoes}
- Arrangement: {arrangement}
- Difficulty: {difficulty}

Success Criteria:
1. PRIMARY GOAL: At least 80% of dominoes must have fallen over
2. The dominoes should form a clear chain reaction pattern
3. Most dominoes should be lying horizontally or at a steep angle (not upright)

What to look for in the final image:
- Count how many dominoes are clearly fallen (lying down)
- Check if there's a clear progression of the chain reaction
- Upright dominoes indicate failure at that point in the chain

Evaluation Guidelines:
- If 80% or more dominoes are fallen: SUCCESS
- If 60-79% dominoes are fallen: PARTIAL SUCCESS (judge as success if chain reaction clearly occurred)
- If less than 60% dominoes are fallen: FAILURE

Please analyze the final image and determine if the chain reaction was successful based on the domino positions.
"""  
        for task_result in task_results:
            try:
                judge_metrics = self.evaluator._evaluate_with_judge(task_result, task_success_criteria)
                task_result.success = judge_metrics.get("judge_success", False)
                # record judge metrics
                task_result.metadata["judge_metrics"] = judge_metrics
            except Exception as e:
                print(f"Judge evaluation failed: {e}")
                task_result.success = False
                task_result.metadata["judge_metrics"] = {"judge_success": False, "judge_confidence": 0.0, "judge_reasoning": f"Judge evaluation failed: {e}"}
        
        return task_results
    
    def _get_initial_system_prompt(self) -> str:
        """Get base system prompt for domino tasks."""
        base_prompt = """You are an AI agent controlling a physics simulation to solve domino falling puzzles. 
Your goal is to create a chain reaction by pushing the first domino, causing all dominoes to fall in sequence.
You can observe the environment through images showing the current state of the domino setup from multiple viewpoints.
You may use the available tools to interact with the dominoes as needed.

IMPORTANT GUIDELINES:
1. Always observe the initial setup carefully before taking action
2. Usually, pushing the first domino with appropriate force is sufficient
3. The chain reaction should propagate through all dominoes automatically
4. Success is measured by the percentage of dominoes that fall (typically >80%)
5. Get different viewpoints if needed to assess the situation
6. Be patient - allow time for the physics simulation to propagate the chain reaction"""


        arrangement = self.config.arrangement_pattern
        num_dominoes = self.config.num_dominoes
        
        prompt = base_prompt + "\n\n" + f"""
        TASK SPECIFICS:
        - Number of dominoes: {num_dominoes}
        - Arrangement pattern: {arrangement}
        - Difficulty: {self.config.difficulty.value}
        """
        
        if arrangement == "line":
            prompt += "- The dominoes are arranged in a straight line. Push the first domino to start the chain."
        elif arrangement == "curve":
            prompt += "- The dominoes are arranged in a curved pattern. Consider the curve when pushing."
        elif arrangement == "zigzag":
            prompt += "- The dominoes are arranged in a zigzag pattern. The chain reaction follows the zigzag path."
        elif arrangement == "circle":
            prompt += "- The dominoes are arranged in a circle. The chain reaction should propagate around the circle."
            
        # Add difficulty-specific guidance
        if self.config.difficulty == TaskDifficulty.EASY:
            prompt += "\n- This is an easy setup - standard force should work well."
        elif self.config.difficulty == TaskDifficulty.MEDIUM:
            prompt += "\n- This is a medium difficulty setup - you may need to adjust force or direction."
        elif self.config.difficulty == TaskDifficulty.HARD:
            prompt += "\n- This is a hard setup - careful observation and precise pushing may be needed."
        
        return prompt
        
    def _get_initial_instruction(self) -> str:
        """Get description of the specific domino task instance."""
        return f"""DOMINO FALLING PUZZLE

You are presented with {self.config.num_dominoes} dominoes arranged in a {self.config.arrangement_pattern} pattern.
Your objective is to create a chain reaction by pushing the first domino, causing all dominoes to fall in sequence.

Task Details:
- Difficulty Level: {self.config.difficulty.value}
- Number of Dominoes: {self.config.num_dominoes}
- Arrangement: {self.config.arrangement_pattern}
- Success Criteria: At least 80% of dominoes must fall

The physics simulation will handle the chain reaction once you provide the initial push.
Observe the setup carefully and determine the best approach to ensure maximum domino fall rate.

Please start by observing the domino setup. Look at the arrangement pattern, spacing, and orientation of the dominoes. 
Once you understand the setup, push the first domino with appropriate force to start the chain reaction.

Remember to:
1. First observe the scene to understand the domino layout
2. Push the first domino (usually domino_1) to start the chain
3. Wait for the chain reaction to complete
4. Check the solution to see how many dominoes fell
5. If needed, you can observe from different angles or make additional pushes"""
            
        
