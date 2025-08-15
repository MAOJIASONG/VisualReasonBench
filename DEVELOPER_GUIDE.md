# PhyVPuzzle 开发者技术文档

## 目录

1. [系统架构概述](#系统架构概述)
2. [迁移现有环境到PhyVPuzzle](#迁移现有环境到phyvpuzzle)
3. [Prompt配置位置详解](#prompt配置位置详解)
4. [添加新任务的完整流程](#添加新任务的完整流程)
5. [工具系统集成](#工具系统集成)
6. [多轮交互机制](#多轮交互机制)
7. [VLM判断机制](#vlm判断机制)
8. [调试与测试](#调试与测试)

## 系统架构概述

```
PhyVPuzzle/
├── src/phyvpuzzle/
│   ├── core/                      # 核心系统
│   │   ├── pipeline.py           # 主流程控制
│   │   ├── vllm_processor.py     # VLM处理器（主要Prompt位置）
│   │   ├── action_descriptor.py  # 动作描述
│   │   └── translator.py         # 动作翻译器
│   │
│   ├── environment/               # 物理环境
│   │   ├── physics_env.py        # PyBullet环境基类
│   │   └── phobos_models/        # URDF模型文件
│   │       ├── domino/           # 多米诺模型
│   │       ├── luban-*/          # 鲁班锁模型
│   │       └── YOUR_PUZZLE/      # 你的puzzle模型放这里
│   │
│   ├── tasks/                     # 任务定义
│   │   ├── base_task.py          # 任务基类
│   │   ├── domino_task.py        # 多米诺任务示例
│   │   ├── domino_tools.py       # 多米诺工具定义
│   │   └── YOUR_TASK.py          # 你的新任务放这里
│   │
│   └── utils/                     # 工具类
│       ├── logger.py             
│       └── token_calculator.py   
│
└── configs/
    └── default_config.json        # 全局配置文件
```

## 迁移现有环境到PhyVPuzzle

### 1. 准备URDF模型文件

如果你已有其他环境（如puzzle、机械装置等），首先需要准备模型文件：

```bash
# 1. 将你的URDF/STL文件放到对应目录
src/phyvpuzzle/environment/phobos_models/YOUR_PUZZLE/
├── urdf/
│   └── puzzle.urdf          # URDF描述文件
└── meshes/
    └── stl/                  # STL网格文件
        ├── piece1.stl
        ├── piece2.stl
        └── ...
```

### 2. 创建新的Task类

创建 `src/phyvpuzzle/tasks/puzzle_task.py`：

```python
from typing import Dict, Any, List, Optional
from .base_task import BaseTask, TaskConfiguration, TaskType, TaskDifficulty
import os

class PuzzleTask(BaseTask):
    """你的Puzzle任务实现"""
    
    def __init__(self, config: Optional[TaskConfiguration] = None):
        # 根据难度设置参数
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
            task_type=TaskType.PUZZLE,  # 需要在base_task.py添加PUZZLE类型
            difficulty=TaskDifficulty.EASY,
            max_steps=30,
            time_limit=300.0,
        ))
        
        self.puzzle_pieces = []
        self.target_positions = []
    
    def setup_task(self, environment) -> bool:
        """设置物理环境"""
        self.environment = environment
        
        # 加载URDF模型
        base_dir = os.path.dirname(os.path.dirname(__file__))
        urdf_path = os.path.join(
            base_dir, 
            "environment/phobos_models/YOUR_PUZZLE/urdf/puzzle.urdf"
        )
        
        # 根据参数创建puzzle pieces
        params = self.config.parameters or {}
        num_pieces = params.get("num_pieces", 9)
        
        for i in range(num_pieces):
            piece_name = f"puzzle_piece_{i}"
            # 使用你的URDF或创建原始物体
            if os.path.exists(urdf_path):
                self.environment.add_object(
                    object_name=piece_name,
                    urdf_path=urdf_path,
                    position=[i * 0.1, 0, 0.5],
                    object_type="puzzle_piece"
                )
            else:
                # 使用原始几何体作为备选
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
        
        # 初始化工具系统
        self._setup_tools()
        
        return True
    
    def _setup_tools(self):
        """设置任务专用工具"""
        from .puzzle_tools import PuzzleTools
        self.tools = PuzzleTools(self.environment)
        self.tools.set_pieces(self.puzzle_pieces)
        self.environment.puzzle_tools = self.tools
```

### 3. 创建工具定义

创建 `src/phyvpuzzle/tasks/puzzle_tools.py`：

```python
from typing import Dict, Any, List
import pybullet as p

class PuzzleTools:
    """Puzzle操作工具"""
    
    def __init__(self, environment):
        self.environment = environment
        self.pieces = []
        
    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """定义VLM可用的工具"""
        return [
            {
                "type": "function",
                "function": {
                    "name": "pick_piece",
                    "description": "拾取一个puzzle碎片",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "piece_id": {
                                "type": "string",
                                "description": "要拾取的碎片ID",
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
                    "description": "放置碎片到指定位置",
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
                    "description": "旋转碎片",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "piece_id": {"type": "string"},
                            "angle": {"type": "number", "description": "旋转角度（度）"}
                        },
                        "required": ["piece_id", "angle"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "check_progress",
                    "description": "检查puzzle完成进度",
                    "parameters": {"type": "object", "properties": {}}
                }
            }
        ]
    
    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """执行工具调用"""
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

## Prompt配置位置详解

### 1. 主VLM系统Prompt

**位置**: `src/phyvpuzzle/core/vllm_processor.py:133-147`

```python
# 这是VLM每轮看到的主要prompt
def process_input(self, image: Image.Image, task_description: str, 
                 context: Dict[str, Any], **kwargs):
    
    task_prompt = f"""You are controlling a robot to solve a physics puzzle.
Task: {task_description}

Current context:
{context}

History of actions:
{history_context}

IMPORTANT: Use the provided tools to interact with the environment. Available tools include:
- pick_piece: 拾取puzzle碎片
- place_piece: 放置碎片到指定位置
- rotate_piece: 旋转碎片
- check_progress: 检查完成进度

Analyze the image and use the appropriate tool."""
```

**修改方法**：
```python
# 在vllm_processor.py的132-147行修改主prompt模板
# 可以根据task_type动态调整
if "puzzle" in task_description.lower():
    task_prompt = "你的puzzle专用prompt..."
elif "domino" in task_description.lower():
    task_prompt = "你的domino专用prompt..."
```

### 2. 任务描述Prompt

**位置**: 你的任务类的 `get_task_description()` 方法

```python
# src/phyvpuzzle/tasks/puzzle_task.py
def get_task_description(self) -> str:
    """这个描述会作为task_description传给VLM"""
    return f"""完成{self.config.parameters.get('puzzle_type')}拼图。
    
任务目标：
- 将所有碎片正确放置到目标位置
- 确保碎片方向正确
- 所有碎片必须紧密连接

提示：从边缘碎片开始，逐步向内完成。"""
```

### 3. 任务上下文Prompt

**位置**: 你的任务类的 `get_task_specific_context()` 方法

```python
def get_task_specific_context(self) -> Dict[str, Any]:
    """提供动态上下文信息"""
    placed_pieces = self._count_placed_pieces()
    total_pieces = len(self.puzzle_pieces)
    
    return {
        "puzzle_type": self.config.parameters.get('puzzle_type'),
        "progress": f"{placed_pieces}/{total_pieces} pieces placed",
        "remaining_pieces": [p for p in self.puzzle_pieces if not self._is_placed(p)],
        "hint": self._get_current_hint(),  # 动态提示
        "last_action_feedback": self._get_last_action_feedback()
    }

def _get_current_hint(self) -> str:
    """根据当前状态提供提示"""
    placed = self._count_placed_pieces()
    if placed == 0:
        return "Start with corner pieces - they have two straight edges."
    elif placed < 4:
        return "Complete the frame first by placing all edge pieces."
    else:
        return "Now fill in the center pieces, matching patterns and colors."
```

### 4. VLM完成判断Prompt

**位置**: `src/phyvpuzzle/core/pipeline.py:_check_task_completion_vlm()`

```python
# 大约在pipeline.py的第173-176行附近
completion_prompt = f"""You are evaluating whether a task has been completed successfully.

TASK OBJECTIVE: {task.get_task_description()}

COMPLETION CRITERIA:
{task.get_completion_criteria()}  # 你需要在任务类中定义这个方法

Look at both images carefully and decide if the task is truly complete."""
```

在你的任务类中添加：

```python
def get_completion_criteria(self) -> str:
    """定义完成标准供VLM判断"""
    return f"""
    拼图完成标准：
    ✅ 所有{len(self.puzzle_pieces)}个碎片都已放置
    ✅ 碎片之间紧密连接，没有明显缝隙
    ✅ 图案完整连续，颜色匹配正确
    ✅ 没有碎片重叠或错位
    
    ❌ 如果有以下情况则未完成：
    - 还有碎片未放置
    - 碎片之间有明显缝隙
    - 图案不连续或颜色不匹配
    - 碎片方向错误
    """
```

## 添加新任务的完整流程

### Step 1: 在TaskType枚举中添加新类型

```python
# src/phyvpuzzle/tasks/base_task.py
class TaskType(Enum):
    DOMINOES = "dominoes"
    LEGO = "lego"
    PUZZLE = "puzzle"        # 添加这行
    LUBAN = "luban"         # 鲁班锁
    MECHANICAL = "mechanical" # 机械装置
```

### Step 2: 更新CLI支持

```python
# src/phyvpuzzle/cli.py 第41-42行
eval_parser.add_argument('--task-type', 
    choices=['dominoes', 'puzzle', 'luban', 'mechanical'],  # 添加新类型
    required=True, help='Type of task to evaluate')

# 第130-140行，更新create_sample_tasks函数
if task_type == "puzzle":
    from .tasks.puzzle_task import PuzzleTask
    task = PuzzleTask(config)
elif task_type == "luban":
    from .tasks.luban_task import LubanTask
    task = LubanTask(config)
```

### Step 3: 配置文件更新

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

## 工具系统集成

### 1. 环境注册工具

在 `src/phyvpuzzle/environment/physics_env.py` 中：

```python
class PyBulletEnvironment(PhysicsEnvironment):
    def __init__(self, gui: bool = False, gravity: float = -9.81):
        super().__init__(gui)
        self.domino_tools = None  # 多米诺工具
        self.puzzle_tools = None  # 添加：拼图工具
        self.luban_tools = None   # 添加：鲁班锁工具
    
    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """获取当前任务的工具定义"""
        if self.domino_tools:
            return self.domino_tools.get_tool_schemas()
        elif self.puzzle_tools:  # 添加
            return self.puzzle_tools.get_tool_schemas()
        elif self.luban_tools:   # 添加
            return self.luban_tools.get_tool_schemas()
        return []
    
    def execute_tool_call(self, tool_name: str, arguments: Dict[str, Any]):
        """执行工具调用"""
        if self.domino_tools:
            return self.domino_tools.execute_tool(tool_name, arguments)
        elif self.puzzle_tools:  # 添加
            return self.puzzle_tools.execute_tool(tool_name, arguments)
        elif self.luban_tools:   # 添加
            return self.luban_tools.execute_tool(tool_name, arguments)
        return {"status": "error", "message": "No tools available"}
```

### 2. 工具调用流程

```
VLM看到图像 → 决定调用工具 → Pipeline处理工具调用 → 环境执行 → 等待物理稳定 → 渲染新图像
```

关键代码位置：
- 工具调用处理：`src/phyvpuzzle/core/pipeline.py:316-334`
- 物理等待时间：`src/phyvpuzzle/core/pipeline.py:360-362`

## 多轮交互机制

### 1. 交互轮数控制

```python
# configs/default_config.json
"max_iterations": 5,  # 最多5轮VLM交互

# 或通过命令行
phyvpuzzle run --task-type puzzle --max-steps 10
```

### 2. 历史信息传递

每轮交互时，VLM会收到：
- 当前图像
- 任务描述
- 历史动作记录（最近5个）
- 任务特定上下文

历史格式化代码：`src/phyvpuzzle/core/vllm_processor.py:294-303`

```python
def _format_history_context(self) -> str:
    """格式化历史供VLM参考"""
    if not self.history:
        return "No previous actions."
    
    formatted_history = []
    for i, entry in enumerate(self.history[-5:]):  # 最近5个动作
        formatted_history.append(f"Step {i+1}: {entry['response']}")
    
    return " | ".join(formatted_history)
```

### 3. 动态提示调整

可以根据轮次动态调整prompt：

```python
# 在你的任务类中
def get_task_specific_context(self) -> Dict[str, Any]:
    context = super().get_task_specific_context()
    
    # 根据步数调整策略
    if self.state.steps_taken < 3:
        context["strategy"] = "探索阶段：仔细观察所有碎片"
    elif self.state.steps_taken < 10:
        context["strategy"] = "执行阶段：系统地放置碎片"
    else:
        context["strategy"] = "收尾阶段：检查并修正错误"
    
    return context
```

## VLM判断机制

### 1. 物理判断（主要）

```python
# 在你的任务类中
def check_completion(self) -> bool:
    """基于物理状态判断是否完成"""
    # 检查所有碎片是否在正确位置
    for i, piece in enumerate(self.puzzle_pieces):
        if not self._is_piece_in_position(piece, self.target_positions[i]):
            return False
    return True

def _is_piece_in_position(self, piece: str, target_pos: Tuple) -> bool:
    """检查碎片是否在目标位置"""
    import pybullet as p
    obj = self.environment.objects.get(piece)
    if not obj:
        return False
    pos, _ = p.getBasePositionAndOrientation(obj.object_id)
    distance = ((pos[0] - target_pos[0])**2 + 
                (pos[1] - target_pos[1])**2)**0.5
    return distance < 0.05  # 容差5cm
```

### 2. VLM视觉判断（辅助）

启用VLM判断：

```python
def __init__(self, config: Optional[TaskConfiguration] = None):
    super().__init__(config)
    self.use_vlm_completion_check = True  # 启用VLM判断
```

VLM判断会在 `src/phyvpuzzle/core/pipeline.py:173-176` 被调用。

### 3. 混合判断策略

```python
def check_completion(self) -> bool:
    """混合物理和视觉判断"""
    # 首先物理判断
    physics_complete = self._check_physics_completion()
    
    if not physics_complete:
        return False
    
    # 如果物理判断通过，可选择性地要求VLM确认
    if self.use_vlm_completion_check:
        # VLM会被Pipeline自动调用
        return True  # 让Pipeline处理VLM判断
    
    return physics_complete
```

## 调试与测试

### 1. 单独测试新任务

```python
# test_puzzle.py
from src.phyvpuzzle.tasks.puzzle_task import PuzzleTask
from src.phyvpuzzle.environment.physics_env import PyBulletEnvironment

# 创建环境和任务
env = PyBulletEnvironment(gui=True)  # 开启GUI调试
env.setup_environment()

task = PuzzleTask()
task.setup_task(env)

# 测试工具
tools = env.puzzle_tools
result = tools.execute_tool("pick_piece", {"piece_id": "puzzle_piece_0"})
print(result)

# 测试完成判断
print(f"Task complete: {task.check_completion()}")
```

### 2. 测试Prompt生成

```python
# 测试任务描述
print(task.get_task_description())

# 测试上下文
print(task.get_task_specific_context())

# 测试完成标准
print(task.get_completion_criteria())
```

### 3. 完整流程测试

```bash
# 测试运行
phyvpuzzle run --task-type puzzle --difficulty easy --gui --verbose

# 带调试输出
PYTHONPATH=. python -m pdb src/phyvpuzzle/cli.py run --task-type puzzle --difficulty easy
```

### 4. 性能分析

```bash
# 测试物理等待时间
phyvpuzzle run --task-type puzzle --physics-settle-time 1.0  # 快速
phyvpuzzle run --task-type puzzle --physics-settle-time 3.0  # 精确

# Token使用分析
phyvpuzzle run --task-type puzzle --verbose  # 会显示token统计
```

## 常见问题

### 1. URDF加载失败
```python
# 使用备用的原始几何体
if not os.path.exists(urdf_path):
    self.environment.create_primitive_object(...)  # 降级方案
```

### 2. 工具未被VLM调用
- 检查工具名称是否在prompt中明确提及
- 确保工具描述清晰
- 验证工具schema格式正确

### 3. 完成判断不准确
- 调整物理判断的容差值
- 完善completion_criteria描述
- 考虑使用多帧平均来判断稳定性

### 4. 性能优化
- 减少不必要的渲染
- 优化物理模拟步长
- 简化碰撞几何体

## 快速开始模板

```python
# src/phyvpuzzle/tasks/new_task_template.py
"""
新任务模板 - 复制此文件并修改
"""
from typing import Dict, Any, List, Optional
from .base_task import BaseTask, TaskConfiguration, TaskType, TaskDifficulty

class NewTask(BaseTask):
    def __init__(self, config: Optional[TaskConfiguration] = None):
        # TODO: 设置难度参数
        super().__init__(config)
        
    def setup_task(self, environment) -> bool:
        # TODO: 创建物理环境
        return True
    
    def get_task_description(self) -> str:
        # TODO: 任务描述（给VLM看的）
        return "Your task description"
    
    def check_completion(self) -> bool:
        # TODO: 完成判断逻辑
        return False
    
    def evaluate_state(self) -> float:
        # TODO: 返回完成度 0.0-1.0
        return 0.0
    
    def get_task_specific_context(self) -> Dict[str, Any]:
        # TODO: 动态上下文
        return {}
```

这个文档应该能帮助开发者快速理解如何迁移现有环境和添加新任务到PhyVPuzzle系统中。