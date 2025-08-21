# PhyVPuzzle 开发者指南

> 本指南详细说明如何扩展 PhyVPuzzle 框架，包括添加新环境、工具、评估指标等。

## 📋 目录

- [架构概览](#-架构概览)
- [添加新环境](#-添加新环境)
- [添加新工具](#-添加新工具)
- [添加新任务](#-添加新任务)
- [添加新评估指标](#-添加新评估指标)
- [添加新 VLM 代理](#-添加新vlm代理)
- [测试和调试](#-测试和调试)
- [最佳实践](#-最佳实践)

## 🏗️ 架构概览

PhyVPuzzle 采用模块化设计，核心组件包括：

```
src/phyvpuzzle/
├── core/           # 基础框架和抽象类
├── environment/    # 物理环境实现
├── tasks/          # 任务定义和逻辑
├── agents/         # VLM 代理实现
├── evaluation/     # 评估系统和指标
├── utils/          # 辅助工具
├── runner.py       # 主运行器
└── cli.py          # 命令行界面
```

### 核心设计原则

1. **统一基类**: 所有组件都继承自对应的抽象基类
2. **配置驱动**: 使用 YAML 配置文件管理参数
3. **工具系统**: VLM 通过预定义工具与环境交互
4. **可扩展性**: 通过继承和重写方法轻松添加新功能

## 🌍 添加新环境

### 步骤 1: 创建环境类

继承 `PhysicsEnvironment` 基类：

```python
# src/phyvpuzzle/environment/my_puzzle_env.py
from typing import Dict, List, Any, Tuple
from .base_env import PhysicsEnvironment, ObjectInfo
from ..core.base import State

class MyPuzzleEnvironment(PhysicsEnvironment):
    """我的拼图环境实现"""
    
    def __init__(self, config: Dict[str, Any]):
        # 环境特定的配置
        self.puzzle_pieces = config.get("puzzle_pieces", 6)
        self.difficulty_level = config.get("difficulty", "medium")
        
        # 调用父类初始化
        super().__init__(config)
        
        # 环境特定的状态
        self.pieces = {}
        self.target_positions = {}
        self.is_solved = False
    
    def _setup_task_environment(self) -> None:
        """设置拼图特定的环境"""
        self._load_puzzle_pieces()
        self._setup_target_configuration()
        
    def _load_puzzle_pieces(self) -> None:
        """加载拼图块"""
        for i in range(self.puzzle_pieces):
            piece_name = f"piece_{i+1}"
            
            # 方法1: 加载 URDF 模型
            if hasattr(self, 'urdf_paths') and piece_name in self.urdf_paths:
                obj_id = self.add_object(
                    piece_name,
                    self.urdf_paths[piece_name],
                    position=(i * 0.1, 0, 0.5),
                    object_type="puzzle_piece"
                )
            # 方法2: 创建基础几何体
            else:
                obj_id = self.create_primitive_object(
                    piece_name,
                    shape_type="box",
                    size=(0.05, 0.05, 0.02),
                    position=(i * 0.1, 0, 0.5),
                    color=(0.8, 0.4, 0.2, 1.0),
                    mass=0.1
                )
            
            self.pieces[piece_name] = obj_id
            
    def _setup_target_configuration(self) -> None:
        """设置目标配置"""
        # 定义每个拼图块的目标位置
        self.target_positions = {
            f"piece_{i+1}": (i * 0.06, 0, 0.41)
            for i in range(self.puzzle_pieces)
        }
```

### 步骤 2: 添加任务特定工具

```python
    def _get_task_specific_tool_schemas(self) -> List[Dict[str, Any]]:
        """定义拼图特定工具"""
        def build_schema(name: str, desc: str, properties: Dict[str, Any], required: List[str]):
            return {
                "type": "function",
                "function": {
                    "name": name,
                    "description": desc,
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                    },
                },
            }
        
        return [
            build_schema(
                "connect_pieces",
                "连接两个拼图块",
                {
                    "piece1_id": {"type": "string", "description": "第一个拼图块名称"},
                    "piece2_id": {"type": "string", "description": "第二个拼图块名称"},
                    "connection_type": {
                        "type": "string", 
                        "enum": ["edge", "corner", "center"],
                        "description": "连接类型"
                    }
                },
                ["piece1_id", "piece2_id"]
            ),
            build_schema(
                "check_fit",
                "检查两个拼图块是否匹配",
                {
                    "piece1_id": {"type": "string", "description": "第一个拼图块"},
                    "piece2_id": {"type": "string", "description": "第二个拼图块"}
                },
                ["piece1_id", "piece2_id"]
            ),
            build_schema(
                "get_piece_info",
                "获取拼图块的详细信息",
                {
                    "piece_id": {"type": "string", "description": "拼图块名称"}
                },
                ["piece_id"]
            )
        ]

    def _execute_task_specific_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """执行拼图特定工具"""
        if tool_name == "connect_pieces":
            return self._connect_pieces(
                arguments.get("piece1_id"),
                arguments.get("piece2_id"),
                arguments.get("connection_type", "edge")
            )
        elif tool_name == "check_fit":
            return self._check_piece_fit(
                arguments.get("piece1_id"),
                arguments.get("piece2_id")
            )
        elif tool_name == "get_piece_info":
            return self._get_piece_info(arguments.get("piece_id"))
        else:
            return super()._execute_task_specific_tool(tool_name, arguments)
```

### 步骤 3: 实现工具功能

```python
    def _connect_pieces(self, piece1_id: str, piece2_id: str, connection_type: str) -> Dict[str, Any]:
        """连接拼图块实现"""
        if piece1_id not in self.pieces or piece2_id not in self.pieces:
            return {"status": "error", "message": "拼图块不存在"}
            
        # 获取拼图块位置
        piece1_state = self.get_object_state(piece1_id)
        piece2_state = self.get_object_state(piece2_id)
        
        if not piece1_state or not piece2_state:
            return {"status": "error", "message": "无法获取拼图块状态"}
        
        # 计算连接位置
        if connection_type == "edge":
            target_pos = self._calculate_edge_connection(piece1_state, piece2_state)
        elif connection_type == "corner":
            target_pos = self._calculate_corner_connection(piece1_state, piece2_state)
        else:
            target_pos = piece1_state["position"]
            
        # 移动拼图块到连接位置
        self._tool_move(piece2_id, target_pos)
        
        return {
            "status": "success", 
            "message": f"已连接 {piece1_id} 和 {piece2_id}",
            "connection_type": connection_type
        }
    
    def _check_piece_fit(self, piece1_id: str, piece2_id: str) -> Dict[str, Any]:
        """检查拼图块匹配"""
        # 实现匹配检查逻辑
        # 这里可以基于几何形状、颜色、纹理等进行判断
        
        contacts = self.get_contact_points(piece1_id, piece2_id)
        fit_score = len(contacts) / 10.0  # 简化的匹配评分
        
        return {
            "status": "success",
            "fit": fit_score > 0.5,
            "fit_score": fit_score,
            "contact_points": len(contacts)
        }
```

### 步骤 4: 实现成功判断

```python
    def _evaluate_success(self) -> bool:
        """评估拼图是否完成"""
        if not self.pieces or not self.target_positions:
            return False
            
        tolerance = 0.05  # 位置容差
        correct_pieces = 0
        
        for piece_name, target_pos in self.target_positions.items():
            if piece_name in self.pieces:
                current_state = self.get_object_state(piece_name)
                if current_state:
                    current_pos = current_state["position"]
                    distance = sum((a - b) ** 2 for a, b in zip(current_pos, target_pos)) ** 0.5
                    
                    if distance < tolerance:
                        correct_pieces += 1
        
        success_ratio = correct_pieces / len(self.target_positions)
        self.is_solved = success_ratio >= 0.8  # 80% 的拼图块在正确位置
        
        return self.is_solved
    
    def _get_current_state(self) -> State:
        """获取当前环境状态"""
        # 收集所有拼图块状态
        objects = {}
        for piece_name in self.pieces:
            piece_state = self.get_object_state(piece_name)
            if piece_state:
                objects[piece_name] = piece_state
        
        return State(
            step=self.step_count,
            objects=objects,
            completed=self.is_solved,
            success=self.is_solved,
            metadata={
                "puzzle_pieces": len(self.pieces),
                "correct_positions": self._count_correct_positions(),
                "completion_ratio": self._get_completion_ratio()
            }
        )
    
    def _get_state_description(self) -> str:
        """获取状态描述"""
        correct_count = self._count_correct_positions()
        total_count = len(self.pieces)
        completion = (correct_count / total_count) * 100 if total_count > 0 else 0
        
        return f"拼图进度: {correct_count}/{total_count} 块正确放置 ({completion:.1f}%)"
```

### 步骤 5: 注册新环境

```python
# src/phyvpuzzle/environment/__init__.py
from .my_puzzle_env import MyPuzzleEnvironment

__all__ = [
    # ... 其他环境
    "MyPuzzleEnvironment",
]
```

## 🔧 添加新工具

### 全局工具 (所有环境可用)

在 `base_env.py` 中添加：

```python
# 在 get_tool_schemas 方法中添加新工具schema
build_schema(
    "my_new_tool",
    "新工具的描述",
    {
        "param1": {"type": "string", "description": "参数1"},
        "param2": {"type": "number", "default": 1.0}
    },
    ["param1"]
),

# 在 execute_tool_call 方法中添加处理逻辑
elif tool_name == "my_new_tool":
    return self._tool_my_new_tool(
        arguments.get("param1"),
        arguments.get("param2", 1.0)
    )

# 实现工具功能
def _tool_my_new_tool(self, param1: str, param2: float) -> Dict[str, Any]:
    """新工具实现"""
    try:
        # 工具逻辑实现
        result = f"处理参数 {param1} 和 {param2}"
        
        return {
            "status": "success",
            "message": result,
            "data": {"param1": param1, "param2": param2}
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}
```

### 环境特定工具

在具体环境类中重写 `_get_task_specific_tool_schemas` 和 `_execute_task_specific_tool` 方法（如上面示例）。

## 📝 添加新任务

### 步骤 1: 创建任务类

```python
# src/phyvpuzzle/tasks/my_puzzle_task.py
from typing import Dict, Any
from .base_task import PuzzleTask
from ..core.base import TaskType, TaskDifficulty

class MyPuzzleTask(PuzzleTask):
    """我的拼图任务"""
    
    def __init__(self, difficulty: TaskDifficulty, config: Dict[str, Any]):
        super().__init__(TaskType.CUSTOM, difficulty, config)
        
        # 任务特定配置
        self.puzzle_pieces = config.get("puzzle_pieces", 6)
        self.time_limit = config.get("time_limit", 300)
        
    def _get_base_system_prompt(self) -> str:
        """基础系统提示"""
        return """你是一个拼图解决专家。你的任务是将散落的拼图块组装成完整的图案。

可用工具:
- pick(object_id): 拾取拼图块
- place(object_id, position): 放置拼图块到指定位置
- move(object_id, position): 移动拼图块
- rotate(object_id, axis, angle): 旋转拼图块
- connect_pieces(piece1_id, piece2_id, connection_type): 连接两个拼图块
- check_fit(piece1_id, piece2_id): 检查两个拼图块是否匹配
- get_piece_info(piece_id): 获取拼图块信息
- observe(angle): 从不同角度观察
- check_solution(): 检查拼图是否完成

解题策略:
1. 首先观察所有拼图块，了解它们的形状和颜色
2. 寻找边缘和角落块，这些通常更容易识别
3. 根据颜色和图案分组拼图块
4. 从边缘开始，逐步向内组装
5. 使用 check_fit 验证拼图块是否匹配
6. 定期检查解答进度

请仔细观察环境，制定解题计划并逐步执行。"""

    def _get_difficulty_specific_prompt(self) -> str:
        """难度特定提示"""
        if self.difficulty == TaskDifficulty.VERY_EASY:
            return f"这是一个包含{self.puzzle_pieces}块的简单拼图，块数较少，形状明显。"
        elif self.difficulty == TaskDifficulty.EASY:
            return f"这是一个包含{self.puzzle_pieces}块的拼图，有清晰的边缘和图案。"
        elif self.difficulty == TaskDifficulty.MEDIUM:
            return f"这是一个包含{self.puzzle_pieces}块的中等难度拼图，需要仔细观察细节。"
        elif self.difficulty == TaskDifficulty.HARD:
            return f"这是一个包含{self.puzzle_pieces}块的困难拼图，颜色和图案相似，需要精细操作。"
        else:
            return f"这是一个包含{self.puzzle_pieces}块的超高难度拼图，需要极其仔细的观察和操作。"
    
    def validate_completion(self, state: Dict[str, Any]) -> bool:
        """验证任务完成"""
        return state.get("success", False)
    
    def get_success_criteria(self) -> Dict[str, Any]:
        """获取成功标准"""
        return {
            "completion_threshold": 0.8,  # 80% 拼图块正确放置
            "time_limit": self.time_limit,
            "required_tools": ["connect_pieces", "check_fit"],
            "success_conditions": [
                "所有拼图块连接成完整图案",
                "图案稳定且正确对齐",
                "在时间限制内完成"
            ]
        }
```

### 步骤 2: 注册任务

```python
# src/phyvpuzzle/tasks/__init__.py
from .my_puzzle_task import MyPuzzleTask

__all__ = [
    # ... 其他任务
    "MyPuzzleTask",
]
```

## 📊 添加新评估指标

### 步骤 1: 扩展指标计算器

```python
# src/phyvpuzzle/evaluation/metrics.py
class MetricsCalculator:
    # ... 现有方法 ...
    
    def calculate_spatial_efficiency(self, task_results: List[Dict[str, Any]]) -> float:
        """计算空间效率指标 - 拼图块移动的总距离"""
        total_distance = 0
        total_tasks = len(task_results)
        
        for result in task_results:
            if result.get("success", False):
                steps_history = result.get("steps_history", [])
                distance = 0
                
                for step in steps_history:
                    if step.get("action_type") == "move":
                        # 计算移动距离
                        start_pos = step.get("start_position", [0, 0, 0])
                        end_pos = step.get("end_position", [0, 0, 0])
                        step_distance = sum((a - b) ** 2 for a, b in zip(start_pos, end_pos)) ** 0.5
                        distance += step_distance
                
                total_distance += distance
        
        return total_distance / total_tasks if total_tasks > 0 else 0
    
    def calculate_assembly_accuracy(self, task_results: List[Dict[str, Any]]) -> float:
        """计算组装精度 - 拼图块位置的准确性"""
        total_accuracy = 0
        successful_tasks = [r for r in task_results if r.get("success", False)]
        
        for result in successful_tasks:
            final_state = result.get("final_state", {})
            metadata = final_state.get("metadata", {})
            
            correct_positions = metadata.get("correct_positions", 0)
            total_pieces = metadata.get("puzzle_pieces", 1)
            
            accuracy = correct_positions / total_pieces
            total_accuracy += accuracy
        
        return total_accuracy / len(successful_tasks) if successful_tasks else 0
    
    def calculate_tool_usage_efficiency(self, task_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """计算工具使用效率"""
        tool_usage = {}
        tool_success = {}
        
        for result in task_results:
            steps_history = result.get("steps_history", [])
            
            for step in steps_history:
                tool_name = step.get("tool_name")
                if tool_name:
                    tool_usage[tool_name] = tool_usage.get(tool_name, 0) + 1
                    
                    if step.get("execution_result", False):
                        tool_success[tool_name] = tool_success.get(tool_name, 0) + 1
        
        # 计算每个工具的成功率
        efficiency = {}
        for tool, total_uses in tool_usage.items():
            successes = tool_success.get(tool, 0)
            efficiency[tool] = successes / total_uses if total_uses > 0 else 0
        
        return efficiency
    
    def calculate_comprehensive_metrics(self, task_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """计算综合指标"""
        base_metrics = super().calculate_comprehensive_metrics(task_results)
        
        # 添加新的指标
        additional_metrics = {
            "spatial_efficiency": self.calculate_spatial_efficiency(task_results),
            "assembly_accuracy": self.calculate_assembly_accuracy(task_results),
            "tool_usage_efficiency": self.calculate_tool_usage_efficiency(task_results),
        }
        
        return {**base_metrics, **additional_metrics}
```

### 步骤 2: 更新评估器

```python
# src/phyvpuzzle/evaluation/evaluator.py
class Evaluator:
    def __init__(self, config: Dict[str, Any]):
        # ... 现有初始化 ...
        self.custom_metrics = config.get("custom_metrics", [])
    
    def evaluate_task_results(self, task_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """评估任务结果"""
        # 基础评估
        base_evaluation = super().evaluate_task_results(task_results)
        
        # 自定义指标评估
        if "spatial_efficiency" in self.custom_metrics:
            spatial_eff = self.metrics_calculator.calculate_spatial_efficiency(task_results)
            base_evaluation["spatial_efficiency"] = spatial_eff
        
        if "assembly_accuracy" in self.custom_metrics:
            assembly_acc = self.metrics_calculator.calculate_assembly_accuracy(task_results)
            base_evaluation["assembly_accuracy"] = assembly_acc
        
        return base_evaluation
```

## 🤖 添加新VLM代理

### 步骤 1: 创建代理类

```python
# src/phyvpuzzle/agents/my_custom_agent.py
import requests
from typing import List, Dict, Any, Tuple, Optional
from .base_agent import VLMAgent

class MyCustomAgent(VLMAgent):
    """自定义VLM代理实现"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # 自定义配置
        self.api_endpoint = config.get("api_endpoint", "https://api.example.com/v1/chat")
        self.api_key = config.get("api_key", "")
        self.custom_params = config.get("custom_params", {})
        
    def _get_model_response(self, messages: List[Dict[str, Any]], 
                          tools: Optional[List[Dict[str, Any]]] = None) -> Tuple[str, List[Dict[str, Any]]]:
        """获取模型响应"""
        
        # 构建请求数据
        request_data = {
            "messages": messages,
            "model": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            **self.custom_params
        }
        
        if tools:
            request_data["tools"] = tools
        
        # 发送请求
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.post(
                self.api_endpoint,
                json=request_data,
                headers=headers,
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            
            # 解析响应
            content = result["choices"][0]["message"]["content"]
            tool_calls = result["choices"][0]["message"].get("tool_calls", [])
            
            return content, tool_calls
            
        except requests.RequestException as e:
            raise RuntimeError(f"API请求失败: {e}")
        except KeyError as e:
            raise RuntimeError(f"响应格式错误: {e}")
    
    def _count_tokens(self, text: str) -> int:
        """计算token数量"""
        # 实现自定义的token计算逻辑
        # 这里使用简单的词数估算
        return len(text.split()) * 1.3  # 粗略估算
```

### 步骤 2: 注册代理

```python
# src/phyvpuzzle/agents/__init__.py
from .my_custom_agent import MyCustomAgent

__all__ = [
    # ... 其他代理
    "MyCustomAgent",
]
```

## 🧪 测试和调试

### 创建测试脚本

```python
# tests/test_my_puzzle_env.py
import unittest
from src.phyvpuzzle.environment.my_puzzle_env import MyPuzzleEnvironment
from src.phyvpuzzle.core.base import TaskDifficulty

class TestMyPuzzleEnvironment(unittest.TestCase):
    
    def setUp(self):
        """测试设置"""
        self.config = {
            "gui": False,
            "puzzle_pieces": 4,
            "difficulty": "easy",
            "render_width": 256,
            "render_height": 256
        }
        self.env = MyPuzzleEnvironment(self.config)
    
    def test_environment_initialization(self):
        """测试环境初始化"""
        self.assertIsNotNone(self.env)
        self.assertEqual(len(self.env.pieces), 4)
        self.assertFalse(self.env.is_solved)
    
    def test_tool_schemas(self):
        """测试工具模式"""
        schemas = self.env.get_tool_schemas()
        tool_names = [schema["function"]["name"] for schema in schemas]
        
        # 检查基础工具
        self.assertIn("pick", tool_names)
        self.assertIn("place", tool_names)
        
        # 检查自定义工具
        self.assertIn("connect_pieces", tool_names)
        self.assertIn("check_fit", tool_names)
    
    def test_tool_execution(self):
        """测试工具执行"""
        # 测试获取拼图块信息
        result = self.env.execute_tool_call("get_piece_info", {"piece_id": "piece_1"})
        self.assertEqual(result["status"], "success")
        
        # 测试连接拼图块
        result = self.env.execute_tool_call("connect_pieces", {
            "piece1_id": "piece_1",
            "piece2_id": "piece_2",
            "connection_type": "edge"
        })
        self.assertEqual(result["status"], "success")
    
    def test_success_evaluation(self):
        """测试成功评估"""
        # 初始状态应该是未完成
        self.assertFalse(self.env._evaluate_success())
        
        # 手动设置正确位置来测试成功检测
        for piece_name, target_pos in self.env.target_positions.items():
            if piece_name in self.env.pieces:
                self.env._tool_move(piece_name, list(target_pos))
        
        # 现在应该检测为成功
        self.assertTrue(self.env._evaluate_success())
    
    def tearDown(self):
        """清理"""
        self.env.close()

if __name__ == "__main__":
    unittest.main()
```

### 调试技巧

```python
# 调试脚本示例
def debug_environment():
    """调试环境功能"""
    config = {
        "gui": True,  # 启用GUI观察
        "puzzle_pieces": 3,
        "render_width": 512,
        "render_height": 512
    }
    
    env = MyPuzzleEnvironment(config)
    
    try:
        # 测试工具功能
        print("=== 测试环境初始化 ===")
        print(f"拼图块数量: {len(env.pieces)}")
        print(f"目标位置: {env.target_positions}")
        
        print("\n=== 测试工具调用 ===")
        result = env.execute_tool_call("get_piece_info", {"piece_id": "piece_1"})
        print(f"获取拼图块信息: {result}")
        
        print("\n=== 测试渲染 ===")
        image = env.render(multi_view=True)
        print(f"渲染图像大小: {image.size if hasattr(image, 'size') else 'Multiple views'}")
        
        print("\n=== 测试状态获取 ===")
        state = env._get_current_state()
        print(f"当前状态: step={state.step}, completed={state.completed}")
        
    finally:
        env.close()

if __name__ == "__main__":
    debug_environment()
```

## 📋 最佳实践

### 1. 代码规范

```python
# 良好的类文档
class MyEnvironment(PhysicsEnvironment):
    """
    我的环境实现
    
    这个环境实现了...功能，支持...操作。
    
    Attributes:
        puzzle_pieces (int): 拼图块数量
        difficulty_level (str): 难度等级
        
    Example:
        >>> config = {"puzzle_pieces": 6, "difficulty": "medium"}
        >>> env = MyEnvironment(config)
        >>> env.reset()
    """

# 类型注解
def my_method(self, param1: str, param2: Optional[int] = None) -> Dict[str, Any]:
    """
    方法说明
    
    Args:
        param1: 参数1说明
        param2: 参数2说明，可选
        
    Returns:
        返回值说明
        
    Raises:
        ValueError: 在什么情况下抛出
    """
    pass
```

### 2. 配置管理

```yaml
# 配置文件示例 (configs/my_puzzle.yaml)
runner:
  experiment_name: "my_puzzle_test"
  max_steps: 20

agent:
  model_name: "gpt-4o"
  temperature: 0.7

environment:
  type: "my_puzzle"
  puzzle_pieces: 6
  difficulty: "medium"
  custom_param: "value"

task:
  name: "my_puzzle_task"
  type: "my_puzzle"
  difficulty: "medium"
  time_limit: 300
```

### 3. 错误处理

```python
def robust_tool_implementation(self, param: str) -> Dict[str, Any]:
    """健壮的工具实现"""
    try:
        # 参数验证
        if not param:
            return {"status": "error", "message": "参数不能为空"}
        
        if param not in self.valid_params:
            return {"status": "error", "message": f"无效参数: {param}"}
        
        # 执行操作
        result = self._do_operation(param)
        
        return {"status": "success", "data": result}
        
    except Exception as e:
        self.logger.error(f"工具执行失败: {e}")
        return {"status": "error", "message": f"执行失败: {str(e)}"}
```

### 4. 性能优化

```python
# 缓存机制
from functools import lru_cache

class OptimizedEnvironment(PhysicsEnvironment):
    
    @lru_cache(maxsize=128)
    def _calculate_expensive_metric(self, state_hash: str) -> float:
        """昂贵计算的缓存版本"""
        # 计算逻辑
        pass
    
    def _batch_operations(self, operations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """批量操作优化"""
        results = []
        
        # 批量处理而不是逐个处理
        for batch in self._create_batches(operations, batch_size=10):
            batch_results = self._process_batch(batch)
            results.extend(batch_results)
        
        return results
```

### 5. 测试覆盖

```python
# 全面的测试用例
class ComprehensiveTest(unittest.TestCase):
    
    def test_edge_cases(self):
        """测试边界情况"""
        # 空输入
        result = self.env.execute_tool_call("my_tool", {})
        self.assertEqual(result["status"], "error")
        
        # 无效输入
        result = self.env.execute_tool_call("my_tool", {"invalid": "param"})
        self.assertEqual(result["status"], "error")
    
    def test_performance(self):
        """性能测试"""
        import time
        
        start_time = time.time()
        for _ in range(100):
            self.env.execute_tool_call("my_tool", {"valid": "param"})
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 100
        self.assertLess(avg_time, 0.1)  # 平均执行时间应小于100ms
```

## 🚀 部署和集成

### 1. 环境注册

在主配置中注册新组件：

```python
# src/phyvpuzzle/core/registry.py
ENVIRONMENT_REGISTRY = {
    "pybullet": PhysicsEnvironment,
    "domino": DominoEnvironment,
    "luban": LubanEnvironment,
    "my_puzzle": MyPuzzleEnvironment,  # 新环境
}

TASK_REGISTRY = {
    "domino": DominoTask,
    "luban": LubanTask,
    "my_puzzle": MyPuzzleTask,  # 新任务
}

AGENT_REGISTRY = {
    "openai": OpenAIAgent,
    "vllm": VLLMAgent,
    "my_custom": MyCustomAgent,  # 新代理
}
```

### 2. CLI集成

```python
# src/phyvpuzzle/cli.py
def add_environment_specific_args(parser, env_type: str):
    """添加环境特定参数"""
    if env_type == "my_puzzle":
        parser.add_argument("--puzzle-pieces", type=int, default=6,
                          help="Number of puzzle pieces")
        parser.add_argument("--puzzle-difficulty", type=str, default="medium",
                          choices=["easy", "medium", "hard"],
                          help="Puzzle difficulty level")
```

恭喜！现在你已经掌握了扩展 PhyVPuzzle 框架的完整方法。记住始终遵循模块化设计原则，保持代码清晰和可测试性。🎉
