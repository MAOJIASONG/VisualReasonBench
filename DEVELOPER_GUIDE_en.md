# PhyVPuzzle Developer Technical Documentation

## Table of Contents

1. [System Architecture Overview](#system-architecture-overview)
2. [Migrating Existing Environments to PhyVPuzzle](#migrating-existing-environments-to-phyvpuzzle)
3. [Prompt Configuration Locations](#prompt-configuration-locations)
4. [Complete Workflow for Adding New Tasks](#complete-workflow-for-adding-new-tasks)
5. [Tool System Integration](#tool-system-integration)
6. [Multi-Round Interaction Mechanism](#multi-round-interaction-mechanism)
7. [VLM Judgment Mechanism](#vlm-judgment-mechanism)
8. [Debugging and Testing](#debugging-and-testing)

## System Architecture Overview

```
PhyVPuzzle/
├── src/phyvpuzzle/
│   ├── core/                      # Core system
│   │   ├── pipeline.py           # Main pipeline control
│   │   ├── vllm_processor.py     # VLM processor (main prompt location)
│   │   ├── action_descriptor.py  # Action description
│   │   └── translator.py         # Action translator
│   │
│   ├── environment/               # Physics environment
│   │   ├── physics_env.py        # PyBullet environment base class
│   │   └── phobos_models/        # URDF model files
│   │       ├── domino/           # Domino models
│   │       ├── luban-*/          # Luban lock models
│   │       └── YOUR_PUZZLE/      # Place your puzzle models here
│   │
│   ├── tasks/                     # Task definitions
│   │   ├── base_task.py          # Task base class
│   │   ├── domino_task.py        # Domino task example
│   │   ├── domino_tools.py       # Domino tool definitions
│   │   └── YOUR_TASK.py          # Place your new task here
│   │
│   └── utils/                     # Utility classes
│       ├── logger.py             
│       └── token_calculator.py   
│
└── configs/
    └── default_config.json        # Global configuration file
```

## Migrating Existing Environments to PhyVPuzzle

### 1. Prepare URDF Model Files

If you have existing environments (puzzles, mechanical devices, etc.), first prepare the model files:

```bash
# 1. Place your URDF/STL files in the appropriate directory
src/phyvpuzzle/environment/phobos_models/YOUR_PUZZLE/
├── urdf/
│   └── puzzle.urdf          # URDF description file
└── meshes/
    └── stl/                  # STL mesh files
        ├── piece1.stl
        ├── piece2.stl
        └── ...
```

### 2. Create New Task Class

Create `src/phyvpuzzle/tasks/puzzle_task.py`:

```python
from typing import Dict, Any, List, Optional
from .base_task import BaseTask, TaskConfiguration, TaskType, TaskDifficulty
import os

class PuzzleTask(BaseTask):
    """Your Puzzle task implementation"""
    
    def __init__(self, config: Optional[TaskConfiguration] = None):
        # Set parameters based on difficulty
        if config:
            if config.difficulty == TaskDifficulty.EASY:
                config.parameters = {
                    "puzzle_type": "3x3",
                    "num_pieces": 9,
                    "time_limit": 180
                }
            elif config.difficulty == TaskDifficulty.HARD:
                config.parameters = {
                    "puzzle_type": "5x5", 
                    "num_pieces": 25,
                    "time_limit": 300
                }
        
        super().__init__(config or TaskConfiguration(
            task_type=TaskType.PUZZLE,  # Need to add PUZZLE type to base_task.py
            difficulty=TaskDifficulty.EASY,
            max_steps=30,
            time_limit=300.0,
        ))
        
        self.puzzle_pieces = []
        self.target_positions = []
    
    def setup_task(self, environment) -> bool:
        """Set up the physics environment"""
        self.environment = environment
        
        # Load URDF models
        base_dir = os.path.dirname(os.path.dirname(__file__))
        urdf_path = os.path.join(
            base_dir, 
            "environment/phobos_models/YOUR_PUZZLE/urdf/puzzle.urdf"
        )
        
        # Create puzzle pieces based on parameters
        params = self.config.parameters or {}
        num_pieces = params.get("num_pieces", 9)
        
        for i in range(num_pieces):
            piece_name = f"puzzle_piece_{i}"
            # Use your URDF or create primitive objects
            if os.path.exists(urdf_path):
                self.environment.add_object(
                    object_name=piece_name,
                    urdf_path=urdf_path,
                    position=[i * 0.1, 0, 0.5],
                    object_type="puzzle_piece"
                )
            else:
                # Use primitive geometry as fallback
                self.environment.create_primitive_object(
                    object_name=piece_name,
                    shape_type="box",
                    size=[0.05, 0.05, 0.02],
                    position=[i * 0.1, 0, 0.5],
                    color=[0.5, 0.5, 0.8, 1.0],
                    mass=0.1
                )
            
            self.puzzle_pieces.append(piece_name)
            self.current_objects[piece_name] = {"type": "puzzle_piece"}
        
        # Initialize tool system
        self._setup_tools()
        
        return True
    
    def _setup_tools(self):
        """Set up task-specific tools"""
        from .puzzle_tools import PuzzleTools
        self.tools = PuzzleTools(self.environment)
        self.tools.set_pieces(self.puzzle_pieces)
        self.environment.puzzle_tools = self.tools
```

### 3. Create Tool Definitions

Create `src/phyvpuzzle/tasks/puzzle_tools.py`:

```python
from typing import Dict, Any, List
import pybullet as p

class PuzzleTools:
    """Puzzle manipulation tools"""
    
    def __init__(self, environment):
        self.environment = environment
        self.pieces = []
        
    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Define tools available to VLM"""
        return [
            {
                "type": "function",
                "function": {
                    "name": "pick_piece",
                    "description": "Pick up a puzzle piece",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "piece_id": {
                                "type": "string",
                                "description": "ID of the piece to pick",
                                "enum": self.pieces
                            }
                        },
                        "required": ["piece_id"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "place_piece",
                    "description": "Place a piece at a specific position",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "piece_id": {"type": "string"},
                            "x": {"type": "number"},
                            "y": {"type": "number"},
                            "rotation": {"type": "number", "default": 0}
                        },
                        "required": ["piece_id", "x", "y"]
                    }
                }
            },
            {
                "type": "function", 
                "function": {
                    "name": "rotate_piece",
                    "description": "Rotate a piece",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "piece_id": {"type": "string"},
                            "angle": {"type": "number", "description": "Rotation angle in degrees"}
                        },
                        "required": ["piece_id", "angle"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "check_progress",
                    "description": "Check puzzle completion progress",
                    "parameters": {"type": "object", "properties": {}}
                }
            }
        ]
    
    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute tool calls"""
        if tool_name == "pick_piece":
            return self.pick_piece(arguments.get("piece_id"))
        elif tool_name == "place_piece":
            return self.place_piece(
                arguments.get("piece_id"),
                arguments.get("x"),
                arguments.get("y"),
                arguments.get("rotation", 0)
            )
        elif tool_name == "rotate_piece":
            return self.rotate_piece(
                arguments.get("piece_id"),
                arguments.get("angle")
            )
        elif tool_name == "check_progress":
            return self.check_progress()
        else:
            return {"status": "error", "message": f"Unknown tool: {tool_name}"}
```

## Prompt Configuration Locations

### 1. Main VLM System Prompt

**Location**: `src/phyvpuzzle/core/vllm_processor.py:133-147`

```python
# This is the main prompt VLM sees each round
def process_input(self, image: Image.Image, task_description: str, 
                 context: Dict[str, Any], **kwargs):
    
    task_prompt = f"""You are controlling a robot to solve a physics puzzle.
Task: {task_description}

Current context:
{context}

History of actions:
{history_context}

IMPORTANT: Use the provided tools to interact with the environment. Available tools include:
- pick_piece: Pick up a puzzle piece
- place_piece: Place piece at specified position
- rotate_piece: Rotate a piece
- check_progress: Check completion progress

Analyze the image and use the appropriate tool."""
```

**How to modify**:
```python
# Modify the main prompt template at lines 132-147 in vllm_processor.py
# Can adjust dynamically based on task_type
if "puzzle" in task_description.lower():
    task_prompt = "Your puzzle-specific prompt..."
elif "domino" in task_description.lower():
    task_prompt = "Your domino-specific prompt..."
```

### 2. Task Description Prompt

**Location**: Your task class's `get_task_description()` method

```python
# src/phyvpuzzle/tasks/puzzle_task.py
def get_task_description(self) -> str:
    """This description is passed to VLM as task_description"""
    return f"""Complete the {self.config.parameters.get('puzzle_type')} puzzle.
    
Task objectives:
- Place all pieces in their correct positions
- Ensure pieces are oriented correctly
- All pieces must fit together tightly

Hint: Start with edge pieces and work your way inward."""
```

### 3. Task Context Prompt

**Location**: Your task class's `get_task_specific_context()` method

```python
def get_task_specific_context(self) -> Dict[str, Any]:
    """Provide dynamic context information"""
    placed_pieces = self._count_placed_pieces()
    total_pieces = len(self.puzzle_pieces)
    
    return {
        "puzzle_type": self.config.parameters.get('puzzle_type'),
        "progress": f"{placed_pieces}/{total_pieces} pieces placed",
        "remaining_pieces": [p for p in self.puzzle_pieces if not self._is_placed(p)],
        "hint": self._get_current_hint(),  # Dynamic hints
        "last_action_feedback": self._get_last_action_feedback()
    }

def _get_current_hint(self) -> str:
    """Provide hints based on current state"""
    placed = self._count_placed_pieces()
    if placed == 0:
        return "Start with corner pieces - they have two straight edges."
    elif placed < 4:
        return "Complete the frame first by placing all edge pieces."
    else:
        return "Now fill in the center pieces, matching patterns and colors."
```

### 4. VLM Completion Judgment Prompt

**Location**: `src/phyvpuzzle/core/pipeline.py:_check_task_completion_vlm()`

```python
# Around lines 173-176 in pipeline.py
completion_prompt = f"""You are evaluating whether a task has been completed successfully.

TASK OBJECTIVE: {task.get_task_description()}

COMPLETION CRITERIA:
{task.get_completion_criteria()}  # You need to define this method in your task class

Look at both images carefully and decide if the task is truly complete."""
```

Add to your task class:

```python
def get_completion_criteria(self) -> str:
    """Define completion criteria for VLM judgment"""
    return f"""
    Puzzle Completion Criteria:
    ✅ All {len(self.puzzle_pieces)} pieces are placed
    ✅ Pieces fit together tightly with no gaps
    ✅ Pattern is complete and continuous, colors match correctly
    ✅ No pieces overlap or are misaligned
    
    ❌ Task is NOT complete if:
    - Any pieces are not placed
    - There are visible gaps between pieces
    - Pattern is discontinuous or colors don't match
    - Pieces are incorrectly oriented
    """
```

## Complete Workflow for Adding New Tasks

### Step 1: Add New Type to TaskType Enum

```python
# src/phyvpuzzle/tasks/base_task.py
class TaskType(Enum):
    DOMINOES = "dominoes"
    LEGO = "lego"
    PUZZLE = "puzzle"        # Add this line
    LUBAN = "luban"         # Luban lock
    MECHANICAL = "mechanical" # Mechanical device
```

### Step 2: Update CLI Support

```python
# src/phyvpuzzle/cli.py lines 41-42
eval_parser.add_argument('--task-type', 
    choices=['dominoes', 'puzzle', 'luban', 'mechanical'],  # Add new types
    required=True, help='Type of task to evaluate')

# Lines 130-140, update create_sample_tasks function
if task_type == "puzzle":
    from .tasks.puzzle_task import PuzzleTask
    task = PuzzleTask(config)
elif task_type == "luban":
    from .tasks.luban_task import LubanTask
    task = LubanTask(config)
```

### Step 3: Update Configuration File

```json
// configs/default_config.json
{
  "tasks": {
    "puzzle": {
      "difficulty_levels": ["easy", "medium", "hard"],
      "max_steps": 30,
      "time_limit": 300.0,
      "success_threshold": 0.95,
      "parameters": {
        "easy": {"puzzle_type": "3x3", "num_pieces": 9},
        "medium": {"puzzle_type": "4x4", "num_pieces": 16},
        "hard": {"puzzle_type": "5x5", "num_pieces": 25}
      }
    }
  }
}
```

## Tool System Integration

### 1. Register Tools in Environment

In `src/phyvpuzzle/environment/physics_env.py`:

```python
class PyBulletEnvironment(PhysicsEnvironment):
    def __init__(self, gui: bool = False, gravity: float = -9.81):
        super().__init__(gui)
        self.domino_tools = None  # Domino tools
        self.puzzle_tools = None  # Add: Puzzle tools
        self.luban_tools = None   # Add: Luban lock tools
    
    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Get tool definitions for current task"""
        if self.domino_tools:
            return self.domino_tools.get_tool_schemas()
        elif self.puzzle_tools:  # Add
            return self.puzzle_tools.get_tool_schemas()
        elif self.luban_tools:   # Add
            return self.luban_tools.get_tool_schemas()
        return []
    
    def execute_tool_call(self, tool_name: str, arguments: Dict[str, Any]):
        """Execute tool calls"""
        if self.domino_tools:
            return self.domino_tools.execute_tool(tool_name, arguments)
        elif self.puzzle_tools:  # Add
            return self.puzzle_tools.execute_tool(tool_name, arguments)
        elif self.luban_tools:   # Add
            return self.luban_tools.execute_tool(tool_name, arguments)
        return {"status": "error", "message": "No tools available"}
```

### 2. Tool Call Flow

```
VLM sees image → Decides to call tool → Pipeline processes tool call → Environment executes → Wait for physics → Render new image
```

Key code locations:
- Tool call handling: `src/phyvpuzzle/core/pipeline.py:316-334`
- Physics wait time: `src/phyvpuzzle/core/pipeline.py:360-362`

## Multi-Round Interaction Mechanism

### 1. Interaction Round Control

```python
# configs/default_config.json
"max_iterations": 5,  # Maximum 5 rounds of VLM interaction

# Or via command line
phyvpuzzle run --task-type puzzle --max-steps 10
```

### 2. History Information Passing

Each interaction round, VLM receives:
- Current image
- Task description
- History of actions (last 5)
- Task-specific context

History formatting code: `src/phyvpuzzle/core/vllm_processor.py:294-303`

```python
def _format_history_context(self) -> str:
    """Format history for VLM reference"""
    if not self.history:
        return "No previous actions."
    
    formatted_history = []
    for i, entry in enumerate(self.history[-5:]):  # Last 5 actions
        formatted_history.append(f"Step {i+1}: {entry['response']}")
    
    return " | ".join(formatted_history)
```

### 3. Dynamic Prompt Adjustment

Adjust prompts dynamically based on rounds:

```python
# In your task class
def get_task_specific_context(self) -> Dict[str, Any]:
    context = super().get_task_specific_context()
    
    # Adjust strategy based on step count
    if self.state.steps_taken < 3:
        context["strategy"] = "Exploration phase: Carefully observe all pieces"
    elif self.state.steps_taken < 10:
        context["strategy"] = "Execution phase: Systematically place pieces"
    else:
        context["strategy"] = "Final phase: Check and correct errors"
    
    return context
```

## VLM Judgment Mechanism

### 1. Physics-Based Judgment (Primary)

```python
# In your task class
def check_completion(self) -> bool:
    """Judge completion based on physical state"""
    # Check if all pieces are in correct positions
    for i, piece in enumerate(self.puzzle_pieces):
        if not self._is_piece_in_position(piece, self.target_positions[i]):
            return False
    return True

def _is_piece_in_position(self, piece: str, target_pos: Tuple) -> bool:
    """Check if piece is at target position"""
    import pybullet as p
    obj = self.environment.objects.get(piece)
    if not obj:
        return False
    pos, _ = p.getBasePositionAndOrientation(obj.object_id)
    distance = ((pos[0] - target_pos[0])**2 + 
                (pos[1] - target_pos[1])**2)**0.5
    return distance < 0.05  # 5cm tolerance
```

### 2. VLM Visual Judgment (Auxiliary)

Enable VLM judgment:

```python
def __init__(self, config: Optional[TaskConfiguration] = None):
    super().__init__(config)
    self.use_vlm_completion_check = True  # Enable VLM judgment
```

VLM judgment is called at `src/phyvpuzzle/core/pipeline.py:173-176`.

### 3. Hybrid Judgment Strategy

```python
def check_completion(self) -> bool:
    """Hybrid physics and visual judgment"""
    # First physics judgment
    physics_complete = self._check_physics_completion()
    
    if not physics_complete:
        return False
    
    # If physics check passes, optionally require VLM confirmation
    if self.use_vlm_completion_check:
        # VLM will be called automatically by Pipeline
        return True  # Let Pipeline handle VLM judgment
    
    return physics_complete
```

## Debugging and Testing

### 1. Test New Task Independently

```python
# test_puzzle.py
from src.phyvpuzzle.tasks.puzzle_task import PuzzleTask
from src.phyvpuzzle.environment.physics_env import PyBulletEnvironment

# Create environment and task
env = PyBulletEnvironment(gui=True)  # Enable GUI for debugging
env.setup_environment()

task = PuzzleTask()
task.setup_task(env)

# Test tools
tools = env.puzzle_tools
result = tools.execute_tool("pick_piece", {"piece_id": "puzzle_piece_0"})
print(result)

# Test completion check
print(f"Task complete: {task.check_completion()}")
```

### 2. Test Prompt Generation

```python
# Test task description
print(task.get_task_description())

# Test context
print(task.get_task_specific_context())

# Test completion criteria
print(task.get_completion_criteria())
```

### 3. Full Pipeline Test

```bash
# Test run
phyvpuzzle run --task-type puzzle --difficulty easy --gui --verbose

# With debugging output
PYTHONPATH=. python -m pdb src/phyvpuzzle/cli.py run --task-type puzzle --difficulty easy
```

### 4. Performance Analysis

```bash
# Test physics settling time
phyvpuzzle run --task-type puzzle --physics-settle-time 1.0  # Fast
phyvpuzzle run --task-type puzzle --physics-settle-time 3.0  # Precise

# Token usage analysis
phyvpuzzle run --task-type puzzle --verbose  # Will show token statistics
```

## Common Issues

### 1. URDF Loading Failure
```python
# Use fallback primitive geometry
if not os.path.exists(urdf_path):
    self.environment.create_primitive_object(...)  # Fallback solution
```

### 2. Tools Not Being Called by VLM
- Check if tool names are explicitly mentioned in prompts
- Ensure tool descriptions are clear
- Verify tool schema format is correct

### 3. Inaccurate Completion Judgment
- Adjust physics judgment tolerance values
- Improve completion_criteria descriptions
- Consider using multi-frame averaging for stability

### 4. Performance Optimization
- Reduce unnecessary rendering
- Optimize physics simulation timestep
- Simplify collision geometry

## Quick Start Template

```python
# src/phyvpuzzle/tasks/new_task_template.py
"""
New Task Template - Copy this file and modify
"""
from typing import Dict, Any, List, Optional
from .base_task import BaseTask, TaskConfiguration, TaskType, TaskDifficulty

class NewTask(BaseTask):
    def __init__(self, config: Optional[TaskConfiguration] = None):
        # TODO: Set difficulty parameters
        super().__init__(config)
        
    def setup_task(self, environment) -> bool:
        # TODO: Create physics environment
        return True
    
    def get_task_description(self) -> str:
        # TODO: Task description (for VLM)
        return "Your task description"
    
    def check_completion(self) -> bool:
        # TODO: Completion judgment logic
        return False
    
    def evaluate_state(self) -> float:
        # TODO: Return completion percentage 0.0-1.0
        return 0.0
    
    def get_task_specific_context(self) -> Dict[str, Any]:
        # TODO: Dynamic context
        return {}
```

This documentation should help developers quickly understand how to migrate existing environments and add new tasks to the PhyVPuzzle system.