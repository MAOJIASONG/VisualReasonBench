# PhyVPuzzle: Physical Visual Reasoning Benchmark

A comprehensive benchmark for evaluating Vision-Language Models (VLMs) on physical reasoning tasks with interactive 3D environments.

## 📋 Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Adding New Tasks](#adding-new-tasks)
- [Environment Configuration](#environment-configuration)
- [Prompt Engineering](#prompt-engineering)
- [Multi-Round Interaction Configuration](#multi-round-interaction-configuration)
- [VLM Completion Judgment](#vlm-completion-judgment)
- [Evaluation Metrics](#evaluation-metrics)
- [Configuration Reference](#configuration-reference)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)

## 🎯 Overview

PhyVPuzzle evaluates VLMs on physical reasoning through interactive 3D environments using PyBullet physics simulation. The system supports:

- **Multiple Task Types**: Dominoes, Lego building, mechanical puzzles
- **Interactive Environments**: Real-time physics simulation with tool calling
- **Comprehensive Metrics**: Accuracy, Pass@K, distance-to-optimal, token efficiency
- **Flexible Configuration**: JSON-based configuration with CLI overrides

## 🚀 Quick Start

```bash
# Run a domino task with GPT-4o
phyvpuzzle run --task-type dominoes --difficulty very-easy --vllm-model gpt-4o

# Run evaluation with multiple attempts
phyvpuzzle evaluate --task-type dominoes --difficulty easy --num-runs 4

# Custom physics settling time (faster/slower physics stabilization)
phyvpuzzle run --task-type dominoes --difficulty easy --physics-settle-time 1.5
```

## 📦 Adding New Tasks

### 1. Task Structure Overview

All tasks inherit from `BaseTask` and follow this structure:

```
src/phyvpuzzle/tasks/
├── base_task.py          # Base class for all tasks
├── domino_task.py        # Example: Domino toppling task
├── your_new_task.py      # Your new task implementation
└── task_tools/           # Task-specific tool definitions
    ├── domino_tools.py
    └── your_task_tools.py
```

### 2. Creating a New Task Class

Create a new file `src/phyvpuzzle/tasks/puzzle_task.py`:

```python
"""
Puzzle Task Implementation

Example: Block stacking/tower building puzzle
"""
from typing import Dict, Any, List, Optional, Tuple
from .base_task import BaseTask, TaskConfiguration, TaskType, TaskDifficulty
from .puzzle_tools import PuzzleTools

class PuzzleTask(BaseTask):
    """Block stacking puzzle task."""
    
    def __init__(self, config: Optional[TaskConfiguration] = None):
        # Define difficulty-specific parameters
        if config and config.difficulty == TaskDifficulty.EASY:
            config.parameters = {
                "num_blocks": 3,
                "target_height": 3,
                "block_size": [0.1, 0.1, 0.1]
            }
        elif config and config.difficulty == TaskDifficulty.HARD:
            config.parameters = {
                "num_blocks": 8,
                "target_height": 5,
                "block_size": [0.08, 0.08, 0.08]
            }
        
        cfg = config or TaskConfiguration(
            task_type=TaskType.PUZZLE,  # Add to TaskType enum
            difficulty=TaskDifficulty.EASY,
            max_steps=20,
            time_limit=180.0,
        )
        super().__init__(cfg)
        self.block_names: List[str] = []
        self.target_height = 0
        self.tools = None
    
    def setup_task(self, environment) -> bool:
        """Initialize the puzzle environment."""
        self.environment = environment
        
        # Get task parameters
        params = self.config.parameters or {}
        num_blocks = params.get("num_blocks", 3)
        block_size = params.get("block_size", [0.1, 0.1, 0.1])
        self.target_height = params.get("target_height", 3)
        
        # Create blocks on the table
        for i in range(num_blocks):
            block_name = f"block_{i+1}"
            self.environment.create_primitive_object(
                object_name=block_name,
                shape_type="box",
                size=block_size,
                position=[i * 0.15, 0, 0.5],  # Spread blocks on table
                color=[0.7, 0.3, 0.3, 1.0],   # Red blocks
                mass=1.0
            )
            self.block_names.append(block_name)
            self.current_objects[block_name] = {"type": "block"}
        
        # Initialize task-specific tools
        self.tools = PuzzleTools(self.environment)
        self.tools.set_block_names(self.block_names)
        environment.puzzle_tools = self.tools  # Make available to environment
        
        return True
    
    def get_task_description(self) -> str:
        """Return human-readable task description."""
        return f"Stack blocks to build a tower {self.target_height} blocks high."
    
    def check_completion(self) -> bool:
        """Check if the puzzle is solved."""
        # Implementation: Check if blocks are stacked to target height
        import pybullet as p
        
        # Find the highest block position
        max_height = 0
        stacked_blocks = 0
        
        for block_name in self.block_names:
            if block_name in self.environment.objects:
                obj = self.environment.objects[block_name]
                pos, _ = p.getBasePositionAndOrientation(obj.object_id)
                if pos[2] > max_height:
                    max_height = pos[2]
                
                # Count blocks that are properly stacked (height > ground level)
                if pos[2] > 0.6:  # Above table level + block height
                    stacked_blocks += 1
        
        # Success if we have target_height blocks stacked
        return stacked_blocks >= self.target_height
    
    def evaluate_state(self) -> float:
        """Return completion percentage (0.0 to 1.0)."""
        # Count properly stacked blocks
        stacked_count = 0
        for block_name in self.block_names:
            if block_name in self.environment.objects:
                obj = self.environment.objects[block_name]
                pos, _ = p.getBasePositionAndOrientation(obj.object_id)
                if pos[2] > 0.6:  # Above ground level
                    stacked_count += 1
        
        return min(1.0, stacked_count / self.target_height)
    
    def get_optimal_solution(self) -> List[str]:
        """Return optimal sequence of actions."""
        return [f"stack block_{i+1} on block_{i}" for i in range(self.target_height-1)]
    
    def get_task_specific_context(self) -> Dict[str, Any]:
        """Provide task context to VLM."""
        return {
            "num_blocks": len(self.block_names),
            "target_height": self.target_height,
            "available_blocks": self.block_names,
            "hint": "Use pick, place, and stack_block tools to build the tower."
        }
```

### 3. Creating Task-Specific Tools

Create `src/phyvpuzzle/tasks/puzzle_tools.py`:

```python
"""
Puzzle Task Tools

Tool definitions for block stacking puzzles.
"""
from typing import Dict, Any, List
import pybullet as p

class PuzzleTools:
    """Tools for block stacking puzzles."""
    
    def __init__(self, environment):
        self.environment = environment
        self.block_names = []
    
    def set_block_names(self, names: List[str]):
        """Set available block names."""
        self.block_names = names
    
    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Define available tools for the VLM."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "pick_block",
                    "description": "Pick up a block",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "block_name": {
                                "type": "string",
                                "description": "Name of the block to pick",
                                "enum": self.block_names
                            }
                        },
                        "required": ["block_name"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "stack_block",
                    "description": "Stack one block on top of another",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "bottom_block": {
                                "type": "string",
                                "description": "Block to stack on top of",
                                "enum": self.block_names
                            },
                            "top_block": {
                                "type": "string", 
                                "description": "Block to place on top",
                                "enum": self.block_names
                            }
                        },
                        "required": ["bottom_block", "top_block"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "check_tower",
                    "description": "Check the current tower height and stability",
                    "parameters": {
                        "type": "object",
                        "properties": {}
                    }
                }
            }
        ]
    
    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool and return results."""
        if tool_name == "pick_block":
            return self.pick_block(arguments.get("block_name"))
        elif tool_name == "stack_block":
            return self.stack_block(
                arguments.get("bottom_block"),
                arguments.get("top_block")
            )
        elif tool_name == "check_tower":
            return self.check_tower()
        else:
            return {"status": "error", "message": f"Unknown tool: {tool_name}"}
    
    def pick_block(self, block_name: str) -> Dict[str, Any]:
        """Pick up a block."""
        if block_name not in self.block_names:
            return {"status": "error", "message": f"Block {block_name} not found"}
        
        # Use environment's pick functionality
        result = self.environment.pick(block_name)
        return {
            "status": "success" if result["success"] else "error",
            "message": f"Picked up {block_name}" if result["success"] else f"Failed to pick {block_name}",
            "block_picked": block_name
        }
    
    def stack_block(self, bottom_block: str, top_block: str) -> Dict[str, Any]:
        """Stack one block on another."""
        try:
            # Get position of bottom block
            if bottom_block not in self.environment.objects:
                return {"status": "error", "message": f"Bottom block {bottom_block} not found"}
            
            bottom_obj = self.environment.objects[bottom_block]
            bottom_pos, _ = p.getBasePositionAndOrientation(bottom_obj.object_id)
            
            # Calculate stacking position (on top + small offset)
            stack_position = (bottom_pos[0], bottom_pos[1], bottom_pos[2] + 0.2)
            
            # Place the top block
            result = self.environment.place(top_block, stack_position)
            
            return {
                "status": "success" if result["success"] else "error",
                "message": f"Stacked {top_block} on {bottom_block}" if result["success"] else "Stacking failed",
                "bottom_block": bottom_block,
                "top_block": top_block,
                "position": stack_position
            }
            
        except Exception as e:
            return {"status": "error", "message": f"Stacking failed: {str(e)}"}
    
    def check_tower(self) -> Dict[str, Any]:
        """Check tower status."""
        try:
            tower_info = []
            for block_name in self.block_names:
                if block_name in self.environment.objects:
                    obj = self.environment.objects[block_name]
                    pos, orn = p.getBasePositionAndOrientation(obj.object_id)
                    tower_info.append({
                        "block": block_name,
                        "height": pos[2],
                        "position": pos
                    })
            
            # Sort by height
            tower_info.sort(key=lambda x: x["height"], reverse=True)
            max_height = max(info["height"] for info in tower_info)
            tower_height = len([info for info in tower_info if info["height"] > 0.6])
            
            return {
                "status": "success",
                "tower_height": tower_height,
                "max_height": max_height,
                "blocks": tower_info
            }
            
        except Exception as e:
            return {"status": "error", "message": f"Check failed: {str(e)}"}
```

### 4. Register New Task Type

Add to `src/phyvpuzzle/tasks/base_task.py`:

```python
class TaskType(Enum):
    """Available task types."""
    DOMINOES = "dominoes"
    LEGO = "lego"
    PUZZLE = "puzzle"        # Add your new task type
    MECHANICAL = "mechanical" # Another example
```

### 5. Update CLI Integration

In `src/phyvpuzzle/cli.py`, add your task to the choices:

```python
# Update task-type choices
eval_parser.add_argument('--task-type', choices=['dominoes', 'puzzle', 'mechanical'], 
                        required=True, help='Type of task to evaluate')

run_parser.add_argument('--task-type', choices=['dominoes', 'puzzle', 'mechanical'],
                       required=True, help='Type of task to run')

# Update create_sample_tasks function
def create_sample_tasks(task_type: str, difficulty: str, count: int) -> List[BaseTask]:
    """Create sample tasks for testing."""
    tasks = []
    
    for i in range(count):
        config = create_task_config(
            task_type=task_type,
            difficulty=difficulty,
            max_steps=100,
            parameters={"task_id": f"{task_type}_{difficulty}_{i}"}
        )
        
        # Add your task creation logic
        if task_type == "dominoes":
            from .tasks.domino_task import DominoTask
            task = DominoTask(config)
        elif task_type == "puzzle":
            from .tasks.puzzle_task import PuzzleTask
            task = PuzzleTask(config)
        elif task_type == "mechanical":
            from .tasks.mechanical_task import MechanicalTask
            task = MechanicalTask(config)
        else:
            raise ValueError(f"Unknown task type: {task_type}")
        
        tasks.append(task)
    
    return tasks
```

## 🌍 Environment Configuration

### Physics Environment Settings

The physics environment can be configured in `configs/default_config.json`:

```json
{
  "environment_type": "pybullet",
  "physics_settle_time": 2.0,           // Time to wait after actions (seconds)
  "physics_timestep": 0.004166667,      // 1/240 seconds
  "gravity": -9.81,                     // Gravity strength
  "camera_config": {
    "position": [0.0, -1.0, 1.0],
    "target": [0.0, 0.0, 0.0],
    "fov": 60.0,
    "image_width": 512,
    "image_height": 512
  }
}
```

### Adding Custom Objects and Materials

In your task's `setup_task()` method:

```python
def setup_task(self, environment) -> bool:
    # Create custom objects with specific materials
    environment.create_primitive_object(
        object_name="heavy_block",
        shape_type="box",
        size=[0.2, 0.2, 0.2],
        position=[0, 0, 0.5],
        color=[0.8, 0.2, 0.2, 1.0],  # RGBA
        mass=5.0,                    # Heavy object
        friction=0.8,                # High friction
        restitution=0.1              # Low bounce
    )
    
    # Load complex URDF objects
    environment.add_object(
        object_name="robot_arm",
        urdf_path="path/to/robot.urdf",
        position=[0, 0, 0],
        object_type="robot"
    )
    
    return True
```

### Multi-View Rendering

Enable multi-view rendering for better visual understanding:

```python
# In environment setup
environment.camera_config.multi_view = True

# Renders 2x2 grid: front, side, top, angled views
image = environment.render(multi_view=True)
```

## 💬 Prompt Engineering

### 1. Task Description Prompts

The primary task description is defined in your task's `get_task_description()` method:

```python
def get_task_description(self) -> str:
    """This prompt is shown to the VLM every round."""
    difficulty_hints = {
        TaskDifficulty.EASY: "Start with the largest blocks at the bottom.",
        TaskDifficulty.HARD: "Plan carefully - stability is crucial for tall towers."
    }
    
    hint = difficulty_hints.get(self.config.difficulty, "")
    
    return f"""Build a stable tower {self.target_height} blocks high using the available blocks.
    
Rules:
- Use pick_block to pick up blocks
- Use stack_block to place one block on another  
- Use check_tower to verify your progress
- The tower must be stable and not fall over

{hint}

Success condition: Stack {self.target_height} blocks vertically."""
```

### 2. VLM System Prompt Configuration

The main VLM prompt is in `src/phyvpuzzle/core/vllm_processor.py`:

```python
def process_input(self, image: Image.Image, task_description: str, 
                 context: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    
    # Build comprehensive task prompt
    task_prompt = f"""You are an AI assistant controlling a robot to solve physics puzzles.

TASK: {task_description}

CURRENT CONTEXT:
{context}

HISTORY OF PREVIOUS ACTIONS:
{history_context}

IMPORTANT: You must use the provided tools to interact with the environment.

AVAILABLE TOOLS AND WHEN TO USE THEM:
- pick_block(block_name): Pick up a specific block
- stack_block(bottom_block, top_block): Stack one block on another
- check_tower(): Check current tower status
- finish_task(reason): Mark the task as complete

RESPONSE FORMAT:
1. Analyze the current image carefully
2. Identify what needs to be done next
3. Use the appropriate tool to take action
4. Only call finish_task when the objective is fully achieved

REASONING APPROACH:
- Always check the current state before acting
- Plan your stacking sequence from bottom to top
- Ensure each block is properly positioned before placing the next
- Verify stability before continuing

Current image shows the physics simulation. Analyze what you see and take the next logical action."""
```

### 3. Difficulty-Specific Prompts

You can customize prompts based on difficulty in the VLM processor:

```python
def _get_difficulty_prompt(self, difficulty: TaskDifficulty) -> str:
    """Get difficulty-specific guidance."""
    prompts = {
        TaskDifficulty.VERY_EASY: "This is a simple task. Take your time and be methodical.",
        TaskDifficulty.EASY: "Plan your approach step by step.",
        TaskDifficulty.MEDIUM: "This requires careful planning. Consider stability and balance.",
        TaskDifficulty.HARD: "This is challenging. Think multiple steps ahead and prioritize stability.",
        TaskDifficulty.EXPERT: "This requires expert-level planning. Consider all physics constraints."
    }
    return prompts.get(difficulty, "")
```

### 4. Context-Aware Prompts

Provide dynamic context based on current state:

```python
def get_task_specific_context(self) -> Dict[str, Any]:
    """Provide current task context to VLM."""
    # Get current tower height
    current_height = self._get_current_tower_height()
    remaining_blocks = self._get_available_blocks()
    
    context = {
        "objective": f"Build tower {self.target_height} blocks high",
        "current_height": current_height,
        "progress": f"{current_height}/{self.target_height}",
        "remaining_blocks": remaining_blocks,
        "next_steps": self._suggest_next_steps(current_height),
    }
    
    # Add state-specific guidance
    if current_height == 0:
        context["guidance"] = "Start by picking up the largest block for the foundation."
    elif current_height < self.target_height:
        context["guidance"] = f"Continue stacking. You need {self.target_height - current_height} more blocks."
    else:
        context["guidance"] = "Check if the tower is stable and complete."
    
    return context
```

## 🔄 Multi-Round Interaction Configuration

### 1. Pipeline Configuration

Configure interaction limits in `src/phyvpuzzle/core/pipeline.py`:

```python
@dataclass
class PipelineConfig:
    max_iterations: int = 5              # Maximum VLM interactions
    timeout: float = 300.0               # Total timeout in seconds
    physics_settle_time: float = 2.0     # Wait time after each action
    feedback_history_size: int = 5       # How many previous steps to remember
    retry_attempts: int = 3              # Retries on tool execution failure
```

### 2. Dynamic Prompting Across Rounds

The VLM receives different information each round:

```python
def _format_history_context(self) -> str:
    """Format interaction history for VLM context."""
    if not self.history:
        return "This is your first action in this task."
    
    # Show last N interactions
    recent_history = self.history[-self.config.feedback_history_size:]
    
    formatted = ["PREVIOUS ACTIONS:"]
    for i, entry in enumerate(recent_history, 1):
        action_result = entry.get('execution_result', 'unknown')
        formatted.append(f"Round {i}: {entry['response']} -> {'SUCCESS' if action_result else 'FAILED'}")
    
    # Add performance feedback
    success_rate = sum(1 for entry in recent_history if entry.get('execution_result', False))
    success_rate /= len(recent_history)
    
    if success_rate < 0.5:
        formatted.append("\n⚠️  PERFORMANCE WARNING: Many recent actions failed. Be more careful.")
    elif success_rate == 1.0:
        formatted.append("\n✅ PERFORMANCE GOOD: Recent actions were successful.")
    
    return "\n".join(formatted)
```

### 3. Round-Specific Guidance

Provide guidance that evolves as the task progresses:

```python
def _get_round_specific_guidance(self, round_number: int, task_progress: float) -> str:
    """Provide guidance specific to the current round."""
    guidance = []
    
    # Early rounds: Focus on planning
    if round_number <= 2:
        guidance.append("🎯 EARLY STAGE: Take time to analyze the scene and plan your approach.")
        
    # Mid rounds: Focus on execution
    elif round_number <= 4:
        guidance.append("⚡ EXECUTION STAGE: Focus on steady progress toward your goal.")
        
    # Late rounds: Focus on completion
    else:
        guidance.append("🏁 FINAL STAGE: You're running out of actions. Focus on completing the task.")
    
    # Progress-based guidance
    if task_progress < 0.3:
        guidance.append("📊 PROGRESS: You're just getting started. Build a solid foundation.")
    elif task_progress < 0.7:
        guidance.append("📊 PROGRESS: Good progress! Continue with your current approach.")
    else:
        guidance.append("📊 PROGRESS: Almost there! Focus on the final steps.")
    
    return " ".join(guidance)
```

### 4. Adaptive Tool Availability

You can restrict tools based on task progression:

```python
def get_tool_schemas(self) -> List[Dict[str, Any]]:
    """Get available tools based on current task state."""
    base_tools = self._get_base_tools()
    
    # Add tools based on task progress
    current_height = self._get_current_tower_height()
    
    if current_height == 0:
        # Early stage: Only allow picking and basic stacking
        return [tool for tool in base_tools if tool['function']['name'] in 
                ['pick_block', 'check_tower']]
    elif current_height >= self.target_height:
        # Late stage: Only allow checking and finishing
        return [tool for tool in base_tools if tool['function']['name'] in 
                ['check_tower', 'finish_task']]
    else:
        # All tools available
        return base_tools
```

## ✅ VLM Completion Judgment

### 1. Task-Specific Completion Checks

Define completion logic in your task's `check_completion()` method:

```python
def check_completion(self) -> bool:
    """Determine if the task is complete."""
    try:
        # Method 1: Physics-based check
        current_height = self._get_current_tower_height()
        is_stable = self._check_tower_stability()
        
        # Success conditions
        height_achieved = current_height >= self.target_height
        tower_stable = is_stable
        
        return height_achieved and tower_stable
        
    except Exception as e:
        # Fallback: Conservative completion check
        return False

def _check_tower_stability(self) -> bool:
    """Check if the tower is physically stable."""
    import pybullet as p
    
    # Check velocities of all blocks
    for block_name in self.block_names:
        if block_name in self.environment.objects:
            obj = self.environment.objects[block_name]
            linear_vel, angular_vel = p.getBaseVelocity(obj.object_id)
            
            # If any block is moving too fast, tower is unstable
            total_velocity = (sum(v**2 for v in linear_vel) + 
                            sum(v**2 for v in angular_vel)) ** 0.5
            
            if total_velocity > 0.1:  # Threshold for "moving"
                return False
    
    return True
```

### 2. VLM-Based Completion Judgment

Enable VLM to judge completion by setting `use_vlm_completion_check = True`:

```python
def setup_task(self, environment) -> bool:
    # Enable VLM-based completion checking
    self.use_vlm_completion_check = True
    
    # Capture reference images for comparison
    self.initial_image = environment.render()  # Before state
    
    return True
```

The VLM completion check prompt is in `src/phyvpuzzle/core/pipeline.py`:

```python
def _check_task_completion_vlm(self, task: BaseTask) -> bool:
    """Use VLM to judge if task is complete."""
    if not hasattr(task, 'use_vlm_completion_check') or not task.use_vlm_completion_check:
        return False
    
    try:
        # Prepare images for comparison
        current_image = self.environment.render()
        images = [
            ("Initial State", task.initial_image),
            ("Current State", current_image)
        ]
        
        # Completion check prompt
        completion_prompt = f"""You are evaluating whether a task has been completed successfully.

TASK OBJECTIVE: {task.get_task_description()}

COMPLETION CRITERIA:
{task.get_completion_criteria()}  # Define this method in your task

INSTRUCTIONS:
1. Compare the initial state vs current state images
2. Determine if the objective has been achieved
3. Consider both completion AND stability of the result
4. Use the provided tool to report your decision

Look at both images carefully and decide if the task is truly complete."""

        # Define completion check tools
        tools = [{
            "type": "function",
            "function": {
                "name": "judge_completion",
                "description": "Judge whether the task has been completed",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "completed": {
                            "type": "boolean",
                            "description": "Whether the task is completed"
                        },
                        "confidence": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "description": "Confidence level in the judgment"
                        },
                        "reasoning": {
                            "type": "string",
                            "description": "Explanation for the decision"
                        }
                    },
                    "required": ["completed", "confidence", "reasoning"]
                }
            }
        }]
        
        # Get VLM judgment
        result = self.vllm_processor.process_completion_check(
            images=images,
            prompt=completion_prompt,
            tools=tools
        )
        
        # Parse VLM response
        tool_calls = result.get("tool_calls", [])
        if tool_calls:
            for call in tool_calls:
                if call.get("function", {}).get("name") == "judge_completion":
                    args = json.loads(call["function"]["arguments"])
                    completed = args.get("completed", False)
                    confidence = args.get("confidence", 0.0)
                    reasoning = args.get("reasoning", "")
                    
                    self.logger.info(f"VLM Completion Judgment: {completed} (confidence: {confidence:.2f})")
                    self.logger.info(f"VLM Reasoning: {reasoning}")
                    
                    # Only accept high-confidence positive judgments
                    return completed and confidence >= 0.8
        
        return False
        
    except Exception as e:
        self.logger.error(f"VLM completion check failed: {e}")
        return False
```

### 3. Custom Completion Criteria

Define specific completion criteria for your task:

```python
def get_completion_criteria(self) -> str:
    """Return detailed completion criteria for VLM evaluation."""
    return f"""COMPLETION CRITERIA FOR TOWER BUILDING:

✅ SUCCESS CONDITIONS:
1. Tower height: Exactly {self.target_height} blocks stacked vertically
2. Stability: Tower must not be falling or tilting significantly
3. Alignment: Blocks should be reasonably centered on each other
4. All blocks used: All available blocks should be part of the tower

❌ FAILURE CONDITIONS:
- Tower is shorter than {self.target_height} blocks
- Tower has fallen over or is clearly unstable
- Blocks are scattered or not properly stacked
- Tower is leaning at a dangerous angle (>15 degrees)

🔍 EVALUATION FOCUS:
- Count the number of blocks in the vertical stack
- Assess the physical stability of the structure
- Check if blocks are properly positioned on top of each other

The task is complete ONLY when all success conditions are met and none of the failure conditions apply."""
```

## 📊 Evaluation Metrics

### 1. Built-in Metrics

The system supports comprehensive evaluation metrics:

```json
{
  "evaluation": {
    "num_runs": 4,
    "metrics": [
      "accuracy",           // Success rate (0.0-1.0)
      "pass_at_4",         // Pass@K metric (at least one success in K attempts)
      "distance_to_optimal", // Distance from optimal solution
      "step_efficiency",    // Steps taken vs optimal steps
      "time_efficiency",    // Time taken vs time limit
      "token_efficiency"    // Token usage efficiency
    ]
  }
}
```

### 2. Custom Evaluation Metrics

Add custom metrics by extending the evaluation system:

```python
# In src/phyvpuzzle/evaluation/metrics.py

class CustomTaskMetric:
    """Custom metric for specific task evaluation."""
    
    def __init__(self, task_type: str):
        self.task_type = task_type
        self.name = f"custom_{task_type}_metric"
    
    def evaluate(self, results: List[TaskResult]) -> float:
        """Calculate custom metric."""
        if self.task_type == "puzzle":
            return self._evaluate_puzzle_specific(results)
        else:
            return 0.0
    
    def _evaluate_puzzle_specific(self, results: List[TaskResult]) -> float:
        """Puzzle-specific evaluation logic."""
        total_score = 0.0
        
        for result in results:
            # Custom scoring based on task specifics
            if result.success:
                # Bonus for efficiency
                step_bonus = max(0, 1.0 - (result.steps_taken - 3) / 10)  # Optimal is 3 steps
                time_bonus = max(0, 1.0 - result.time_taken / 60.0)       # Under 1 minute
                
                score = 1.0 + 0.2 * step_bonus + 0.1 * time_bonus
                total_score += min(1.2, score)  # Cap at 120%
        
        return total_score / len(results) if results else 0.0
```

### 3. Real-time Evaluation Feedback

Provide feedback during task execution:

```python
def update_state(self, action_description: str, success: bool) -> None:
    """Update task state and provide feedback."""
    super().update_state(action_description, success)
    
    # Calculate current performance metrics
    current_progress = self.evaluate_state()
    efficiency = self.state.steps_taken / self.get_optimal_steps()
    
    # Provide feedback for next round
    if current_progress > 0.8:
        self.feedback_history.append("🎯 Excellent progress! You're almost there.")
    elif current_progress > 0.5:
        self.feedback_history.append("👍 Good progress. Keep going steadily.")
    elif efficiency > 1.5:
        self.feedback_history.append("⚠️ You're using many steps. Try to be more efficient.")
    else:
        self.feedback_history.append("📈 Making progress. Stay focused on the goal.")
```

## ⚙️ Configuration Reference

### Complete Configuration File Structure

```json
{
  // VLM Configuration
  "vllm_type": "openai",                    // "openai" or "huggingface"
  "vllm_model": "gpt-4o",                   // Model name
  
  // Task Configuration  
  "translator_type": "rule_based",          // "rule_based" or "llm"
  "environment_type": "pybullet",           // Physics engine
  "max_iterations": 5,                      // Max VLM interactions per task
  "timeout": 300.0,                         // Task timeout (seconds)
  "physics_settle_time": 2.0,               // Physics stabilization time
  
  // Logging Configuration
  "enable_logging": true,
  "log_level": "INFO",                      // DEBUG, INFO, WARNING, ERROR
  "feedback_history_size": 5,               // History context size
  "retry_attempts": 3,                      // Tool execution retries
  
  // Evaluation Configuration
  "evaluation": {
    "num_runs": 4,                          // Attempts per task for Pass@K
    "metrics": [
      "accuracy",
      "pass_at_4", 
      "distance_to_optimal",
      "step_efficiency",
      "time_efficiency",
      "token_efficiency"
    ]
  },
  
  // Task-Specific Configuration
  "tasks": {
    "puzzle": {
      "difficulty_levels": ["easy", "medium", "hard"],
      "max_steps": 20,
      "time_limit": 180.0,
      "success_threshold": 0.9,
      "custom_parameters": {
        "stability_check": true,
        "precision_tolerance": 0.05
      }
    },
    "dominoes": {
      "difficulty_levels": ["very-easy", "easy", "medium", "hard"],
      "max_steps": 10,
      "time_limit": 120.0,
      "success_threshold": 1.0
    }
  },
  
  // Environment Configuration
  "environment": {
    "gravity": -9.81,
    "timestep": 0.004166667,                // 1/240 seconds
    "camera": {
      "position": [0.0, -1.0, 1.0],
      "target": [0.0, 0.0, 0.0],
      "fov": 60.0,
      "image_size": [512, 512],
      "multi_view": true
    },
    "materials": {
      "default_friction": 0.7,
      "default_restitution": 0.3
    }
  }
}
```

## 🎮 Examples

### Example 1: Running a Custom Puzzle Task

```bash
# Basic run
phyvpuzzle run --task-type puzzle --difficulty easy --vllm-model gpt-4o

# With custom physics timing
phyvpuzzle run --task-type puzzle --difficulty hard --physics-settle-time 3.0 --max-steps 15

# With GUI for debugging
phyvpuzzle run --task-type puzzle --difficulty medium --gui --verbose
```

### Example 2: Evaluation with Custom Configuration

```bash
# Comprehensive evaluation
phyvpuzzle evaluate --task-type puzzle --difficulty easy --num-runs 8 --output results.json

# Quick evaluation for development
phyvpuzzle evaluate --task-type puzzle --difficulty easy --num-runs 2 --physics-settle-time 1.0
```

### Example 3: Multi-Task Evaluation Script

```python
#!/usr/bin/env python3
"""
Custom evaluation script for multiple tasks and difficulties.
"""

import json
from src.phyvpuzzle.cli import run_evaluation
import argparse

def run_comprehensive_evaluation():
    """Run evaluation across multiple tasks and difficulties."""
    
    tasks = ["dominoes", "puzzle"]
    difficulties = ["easy", "medium", "hard"]
    
    all_results = {}
    
    for task in tasks:
        all_results[task] = {}
        
        for difficulty in difficulties:
            print(f"\n🔄 Evaluating {task} - {difficulty}")
            
            # Create args object
            args = argparse.Namespace(
                task_type=task,
                difficulty=difficulty,
                num_runs=4,
                vllm_type="openai",
                vllm_model="gpt-4o",
                output=f"results_{task}_{difficulty}.json",
                config=None,
                verbose=True
            )
            
            try:
                run_evaluation(args)
                
                # Load results
                with open(args.output, 'r') as f:
                    results = json.load(f)
                
                all_results[task][difficulty] = results
                print(f"✅ {task} - {difficulty}: Success rate = {results['results']['metrics'].get('accuracy', 0):.2f}")
                
            except Exception as e:
                print(f"❌ {task} - {difficulty} failed: {e}")
                all_results[task][difficulty] = {"error": str(e)}
    
    # Save comprehensive results
    with open("comprehensive_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\n📊 Comprehensive evaluation complete! Results saved to comprehensive_results.json")

if __name__ == "__main__":
    run_comprehensive_evaluation()
```

## 🔧 Troubleshooting

### Common Issues

1. **Physics Settling Too Fast**
   ```bash
   # Increase settling time for complex objects
   phyvpuzzle run --task-type puzzle --physics-settle-time 5.0
   ```

2. **VLM Not Using Tools**
   - Check tool schema definitions in your task's `get_tool_schemas()`
   - Verify tool names match in `execute_tool()` method
   - Review VLM prompt clarity in `vllm_processor.py`

3. **Task Completion Not Detected**
   ```python
   # Add debugging to your check_completion method
   def check_completion(self) -> bool:
       result = self._check_completion_logic()
       print(f"DEBUG: Completion check result: {result}")
       return result
   ```

4. **Poor Evaluation Scores**
   - Adjust optimal solution definitions in `get_optimal_solution()`
   - Review completion criteria in `get_completion_criteria()`
   - Check if task difficulty matches VLM capabilities

### Performance Optimization

1. **Faster Physics Simulation**
   ```json
   {
     "physics_settle_time": 1.0,    // Reduce for simple scenes
     "max_iterations": 3,           // Limit VLM rounds
     "environment": {
       "timestep": 0.008333333      // Larger timestep (120 Hz)
     }
   }
   ```

2. **Memory Optimization**
   ```python
   # In your task setup
   def setup_task(self, environment) -> bool:
       # Limit object count for large evaluations
       max_objects = 10
       # Use simpler collision shapes when possible
       shape_type = "sphere"  # Faster than "box" or "mesh"
       return True
   ```

3. **Token Usage Optimization**
   ```python
   # Shorter, more focused prompts
   def get_task_description(self) -> str:
       return f"Stack {self.target_height} blocks. Use: pick_block, stack_block, check_tower."
   ```

## 📝 Contributing

When adding new tasks:

1. **Follow the established patterns** in existing tasks
2. **Include comprehensive tool definitions** with clear descriptions
3. **Provide multiple difficulty levels** with appropriate parameters
4. **Write clear completion criteria** for both physics and VLM evaluation
5. **Add configuration examples** in this README
6. **Test thoroughly** with different VLM models and settings

### Code Style Guidelines

- Use type hints for all function parameters and returns
- Follow the existing docstring format
- Add error handling for all physics operations
- Include logging statements for debugging
- Write unit tests for new functionality

This comprehensive guide should help you create sophisticated physical reasoning tasks with rich VLM interaction capabilities. The system is designed to be extensible while maintaining consistency across different task types.