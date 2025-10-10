"""
Simple stacking task implementation for puzzle translater test.
这是一个简单的堆叠任务，只需要操作2个拼图块。
"""
import math
import os
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
from phyvpuzzle.core.base import ObjectInfo, TaskDifficulty, TaskResult
from phyvpuzzle.core import register_task, register_task_config, TaskConfig
from phyvpuzzle.tasks.base_task import PhysicsTask
from phyvpuzzle.environment.puzzle_env import PuzzleEnvironment, p


@register_task_config("simple_stacking")
@dataclass
class SimpleStackingTaskConfig(TaskConfig):
    """Configuration for simple stacking task."""
    num_pieces: int = 2  # 只操作2个拼图块
    puzzle_size: Tuple = (2, 1)
    piece_size: float = 0.08
    ruled_evaluation: bool = False


@register_task("simple_stacking")
class SimpleStackingTask(PhysicsTask):
    """Simple stacking task: stack two specific puzzle pieces."""
    
    def __init__(self, config: SimpleStackingTaskConfig):
        super().__init__(config)
        
    def _calculate_optimal_steps(self) -> int:
        """Calculate optimal steps for simple stacking task."""
        # 简单任务：移动第一个块到容器 + 移动第二个块到第一个块上面 = 2步
        return 2
            
    def _configure_environment(self) -> None:
        """Configure environment for simple stacking task."""
        if not isinstance(self.environment, PuzzleEnvironment):
            raise ValueError("SimpleStackingTask requires PuzzleEnvironment")
        
        self._load_puzzle_models()
        
    def _load_puzzle_models(self) -> None:
        """Load puzzle models for simple stacking task."""
        puzzle_base_path = os.path.join(self.environment.config.urdf_local_path, "3x3-stacking-puzzle")
        
        if not os.path.exists(puzzle_base_path):
            print(f"Warning: 3x3 puzzle models not found at {puzzle_base_path}")
            print("Creating simple puzzle pieces instead")
            self._create_simple_puzzle_pieces()
            return
        
        available_parts = []
        if os.path.exists(puzzle_base_path):
            for item in os.listdir(puzzle_base_path):
                puzzle_dir = os.path.join(puzzle_base_path, item)
                urdf_path = os.path.join(puzzle_dir, "urdf", f"{item}.urdf")
                if os.path.isdir(puzzle_dir) and os.path.exists(urdf_path):
                    available_parts.append(urdf_path)
                    
        if not available_parts:
            print("No puzzle URDF files found, creating simple pieces")
            self._create_simple_puzzle_pieces()
            return
        
        # 对于simple_stacking任务，我们只需要加载特定的几个对象：
        # - container (obj_8)
        # - Object #7 (ID: 2) - 对应 obj_2
        # - Object #6 (ID: 3) - 对应 obj_3
        
        target_objects = ["obj_8", "obj_2", "obj_3"]  # container + 两个目标拼图块
        selected_parts = []
        
        for urdf_path in available_parts:
            for target in target_objects:
                if target in urdf_path:
                    selected_parts.append(urdf_path)
                    break
        
        if len(selected_parts) < 3:
            print(f"Warning: Could not find all required objects. Found {len(selected_parts)}/3")
            print("Creating simple puzzle pieces instead")
            self._create_simple_puzzle_pieces()
            return
        
        # 确保container是第一个
        selected_parts.sort(key=lambda x: 0 if "obj_8" in x else 1)
        
        # 加载对象
        print(f"Loading {len(selected_parts)} objects for simple stacking task...")
        
        table_loaded = self.environment.config.load_table
        if table_loaded and self.environment.config.table_position:
            base_x, base_y, base_z = self.environment.config.table_position
        else:
            base_x, base_y, base_z = 0.0, 0.0, 0.0
            
        ground_z = base_z + 0.05
        
        # 计算container scale（使用与3x3_stacking相同的逻辑）
        container_scale = 2.0  # 简单任务使用固定scale
        
        # 布局：container在左侧，两个pieces在右侧
        spacing = 0.3
        
        for i, urdf_path in enumerate(selected_parts):
            if "obj_8" in urdf_path:
                # Container放在左侧
                position = (base_x - spacing, base_y, ground_z)
                piece_name = "container"
                properties = {"index": 8, "is_container": True}
                scale_factor = container_scale
            elif "obj_2" in urdf_path:
                # Object #7 (ID: 2) 放在中间
                position = (base_x + spacing * 0.5, base_y - spacing * 0.3, ground_z)
                piece_name = "piece_2"
                properties = {"index": 2, "is_container": False, "target_order": 1}
                scale_factor = 1.0
            elif "obj_3" in urdf_path:
                # Object #6 (ID: 3) 放在右侧
                position = (base_x + spacing * 0.5, base_y + spacing * 0.3, ground_z)
                piece_name = "piece_3"
                properties = {"index": 3, "is_container": False, "target_order": 2}
                scale_factor = 1.0
            else:
                continue

            self.environment.add_object(
                object_name=piece_name,
                urdf_path=urdf_path,
                position=position,
                orientation=(0, 0, 0, 1),
                object_type="puzzle_piece",
                properties=properties,
                scale=scale_factor
            )

        print(f"✅ Loaded {len(selected_parts)} objects successfully for simple stacking task.")
        print(f"→ Container scaled by {container_scale:.2f}x")
        print(f"→ Target pieces spawned on ground")
    
    def _create_simple_puzzle_pieces(self) -> None:
        """Create simple colored cubes for the stacking task."""
        
        table_pos = self.environment.config.table_position
        table_x, table_y, table_z = table_pos
        
        # 创建container（黑色框）- 使用质量为0确保完全固定
        container_id = self.environment.create_primitive_object(
            object_name="container",
            shape_type="box",
            size=(0.15, 0.15, 0.15),
            position=(table_x - 0.3, table_y, table_z + 0.05),
            color=(0.1, 0.1, 0.1, 1.0),  # 黑色
            mass=0.0,  # 质量为0，完全固定不动
        )
        
        # 标记为容器并更新属性
        for obj in self.environment.objects:
            if obj.object_id == container_id:
                obj.properties['is_container'] = True
                break
        
        # 创建 piece_2 (Object #7, ID: 2) - 红色，增加质量使其更重
        self.environment.create_primitive_object(
            object_name="piece_2",
            shape_type="box",
            size=(0.04, 0.04, 0.04),
            position=(table_x + 0.15, table_y - 0.1, table_z + 0.1),
            color=(1.0, 0.0, 0.0, 1.0),  # 红色
            mass=0.5,  # 增加质量（从0.05到0.5），物体更重更稳定
        )
        
        # 创建 piece_3 (Object #6, ID: 3) - 蓝色，增加质量使其更重
        self.environment.create_primitive_object(
            object_name="piece_3",
            shape_type="box",
            size=(0.04, 0.04, 0.04),
            position=(table_x + 0.15, table_y + 0.1, table_z + 0.1),
            color=(0.0, 0.0, 1.0, 1.0),  # 蓝色
            mass=0.5,  # 增加质量（从0.05到0.5），物体更重更稳定
        )
    
    def _evaluate_success(self, task_results: List[TaskResult]) -> List[TaskResult]:
        """Evaluate task success based on stacking completion."""
        if self.config.ruled_evaluation:
            return self._evaluate_success_ruled(task_results)
        else:
            return self._evaluate_success_agent(task_results)
        
    def _evaluate_success_ruled(self, task_results: List[TaskResult]) -> List[TaskResult]:
        """Rule-based evaluation for simple stacking task."""
        for task_result in task_results:
            last_step = task_result.trajectory[-1]
            if isinstance(last_step, dict):
                observations = last_step.get("observations", [])
                if not observations:
                    task_result.success = False
                    continue
                final_observation = observations[-1]
                objects = final_observation.state.objects
            else:
                final_observation = last_step[1]
                objects = final_observation.state.objects
            
            # 查找container和两个拼图块
            container = None
            piece_2 = None
            piece_3 = None
            
            for obj in objects:
                if obj.properties.get('is_container', False) or 'container' in obj.name.lower():
                    container = obj
                elif 'piece_2' in obj.name or obj.properties.get('index') == 2:
                    piece_2 = obj
                elif 'piece_3' in obj.name or obj.properties.get('index') == 3:
                    piece_3 = obj
            
            if not all([container, piece_2, piece_3]):
                task_result.success = False
                continue
            
            # 检查 piece_2 是否在 container 内
            container_aabb_min, container_aabb_max = p.getAABB(container.object_id)
            piece_2_aabb_min, piece_2_aabb_max = p.getAABB(piece_2.object_id)
            
            tolerance = 0.01
            piece_2_in_container = (
                piece_2_aabb_min[0] >= container_aabb_min[0] - tolerance and
                piece_2_aabb_max[0] <= container_aabb_max[0] + tolerance and
                piece_2_aabb_min[1] >= container_aabb_min[1] - tolerance and
                piece_2_aabb_max[1] <= container_aabb_max[1] + tolerance and
                piece_2_aabb_min[2] >= container_aabb_min[2] - tolerance
            )
            
            # 检查 piece_3 是否在 piece_2 上面
            piece_3_aabb_min, piece_3_aabb_max = p.getAABB(piece_3.object_id)
            piece_3_above_piece_2 = piece_3_aabb_min[2] > piece_2_aabb_max[2] - tolerance * 2
            
            # 检查 piece_3 是否也在 container 内
            piece_3_in_container = (
                piece_3_aabb_min[0] >= container_aabb_min[0] - tolerance and
                piece_3_aabb_max[0] <= container_aabb_max[0] + tolerance and
                piece_3_aabb_min[1] >= container_aabb_min[1] - tolerance and
                piece_3_aabb_max[1] <= container_aabb_max[1] + tolerance and
                piece_3_aabb_min[2] >= container_aabb_min[2] - tolerance
            )
            
            task_result.success = piece_2_in_container and piece_3_above_piece_2 and piece_3_in_container
            
        return task_results
    
    def _evaluate_success_agent(self, task_results: List[TaskResult]) -> List[TaskResult]:
        """VLM-based evaluation for simple stacking task."""
        for task_result in task_results:
            last_step = task_result.trajectory[-1]
            if isinstance(last_step, dict):
                observations = last_step.get("observations", [])
                if not observations:
                    task_result.success = False
                    continue
                final_observation = observations[-1]
            else:
                final_observation = last_step[1]
            
            task_success_criteria = """这是一个简单的拼图块堆叠任务。

所有stack都在框中，并且是堆叠的状态就算成功"""
            
            try:
                judge_metrics = self.evaluator._evaluate_with_judge(task_result, task_success_criteria)
                task_result.success = judge_metrics.get("judge_success", False)
                task_result.metadata["judge_metrics"] = judge_metrics
            except Exception as e:
                print(f"Judge evaluation failed: {e}")
                task_result.success = False
                task_result.metadata["judge_metrics"] = {
                    "judge_success": False, 
                    "judge_confidence": 0.0, 
                    "judge_reasoning": f"Judge evaluation failed: {e}"
                }
        
        return task_results
        
    def _get_initial_system_prompt(self) -> str:
        """Get system prompt for the simple stacking task."""
        
        return """你是一个在基于物理的3D仿真环境中运行的智能AI代理。

角色定位：
- 你是一个空间推理和问题解决代理
- 你可以观察3D场景，理解物体几何，并规划物理交互
- 你通过逻辑性的、循序渐进的推理来做出决策

行为准则：
- 在操作物体时系统性地思考空间关系
- 使用观察结果来指导精确的、物理上有效的动作
- 在排列或组装物体时考虑稳定性、接触和适配
- 总是在行动前进行推理，并根据环境反馈调整计划

工具使用：
- 你**只能通过提供给你的可用工具**与环境交互
- 根据需要使用这些工具与物体交互，以推进目标的实现

你的总体使命是通过仔细推理和连续的物理动作来解决复杂的3D操作和拼图任务。
"""

    def _get_initial_instruction(self) -> str:
        """Get task instruction for simple stacking."""
        
        return """
场景说明：
- 你面前有1个黑色容器（container）和2个拼图块
- 所有物体的详细信息（object_id、位置、颜色）会在下方的 OBJECT MAPPING 中实时提供
- OBJECT MAPPING 会在每一步之后更新，显示物体的当前位置

任务目标：
1. 首先将第一个拼图块（piece_3，通常是绿色）移动到黑色容器内的任意一个角落位置，同时尽可能的贴边，长的边贴近于container边缘，同时避免超出container，你需要观测物体的形状和长边
2. 然后将第二个拼图块（piece_2，通常是蓝色）放置在第一个块的上方，需要将砖块对齐，即底部两个和绿色的砖块对齐，能稳稳的放在绿色的砖块上方

操作指南：
- 使用 object_id（整数）来操作物体，不要使用名称
- 仔细参考 OBJECT MAPPING 中的实时位置信息来规划动作
- 考虑容器的位置和边界
- 按顺序完成两个步骤，确保每一步的精确性，只允许使用 move_object 的操作不断的移动以完成任务

提示：
- 先观察 OBJECT MAPPING 了解所有物体的当前位置
- 计算好目标位置，确保块完全在容器内，同时不要有明显的堆叠
- 考虑物体的尺寸和容器的空间
- 每次移动后检查新的位置是否符合要求
- 如果第一操作基本完成了 就可以进行操作2了
- 可能需要依靠上一步行动结果做一些简单的微调，以真正完成任务
"""

