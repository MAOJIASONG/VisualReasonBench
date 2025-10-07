"""
Puzzle task implementation for jigsaw puzzle assembly.
"""
import math
import os
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
from phyvpuzzle.core.base import ObjectInfo, TaskDifficulty, TaskResult
from phyvpuzzle.core import register_task, register_task_config, TaskConfig
from phyvpuzzle.tasks.base_task import PhysicsTask
from phyvpuzzle.environment.puzzle_env import PuzzleEnvironment, p


@register_task_config("3x3_stacking")
@dataclass
class ThreeByThreeStackingTaskConfig(TaskConfig):
    """Configuration for puzzle assembly task."""
    num_pieces: int = 9
    puzzle_size: Tuple = (3, 3)
    piece_size: float = 0.08  # Size of each puzzle piece cube
    ruled_evaluation: bool = False  # Whether to use rule-based evaluation


@register_task("3x3_stacking")
class ThreeByThreeStackingTask(PhysicsTask):
    """Task for jigsaw puzzle assembly."""
    
    def __init__(self, config: ThreeByThreeStackingTaskConfig):
        super().__init__(config)
        
    def _calculate_optimal_steps(self) -> int:
        """Calculate optimal steps for puzzle assembly task."""
        # TODO: predefine 10 steps to solve the task
        return 10
            
    def _configure_environment(self) -> None:
        """Configure environment for puzzle assembly task."""
        # Ensure we have a puzzle environment
        if not isinstance(self.environment, PuzzleEnvironment):
            raise ValueError("ThreeByThreeStackingTask requires PuzzleEnvironment")
        
        # Setup the puzzle environment
        self._load_puzzle_models()
        
    def _load_puzzle_models(self) -> None:
        """Load puzzle models."""
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
            print("No domino URDF files found, creating simple dominoes")
            self._create_simple_puzzle_pieces()
            return
        
        n = len(available_parts)
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)

        table_loaded = self.environment.config.load_table
        if table_loaded and self.environment.config.table_position:
            base_x, base_y, base_z = self.environment.config.table_position
        else:
            base_x, base_y, base_z = 0.0, 0.0, 0.0
            
            
        # ---------- 第一阶段：测量每个URDF尺寸 ----------
        tmp_ids = []
        max_w = max_d = max_h = 0.0
        piece_dimensions = []

        for i, urdf_path in enumerate(available_parts):
            # 临时加载高空以测量尺寸（在更高的位置避免干扰）
            tmp_id = p.loadURDF(
                urdf_path,
                basePosition=(base_x + i * 0.5, base_y, base_z + 1.5 + 0.1 * i),
                useFixedBase=False,
            )
            tmp_ids.append(tmp_id)

            aabb_min, aabb_max = p.getAABB(tmp_id)
            w = aabb_max[0] - aabb_min[0]
            d = aabb_max[1] - aabb_min[1]
            h = aabb_max[2] - aabb_min[2]
            piece_dimensions.append((w, d, h))
            
            # 如果不是container，更新最大尺寸
            if "obj_8" not in urdf_path:
                max_w = max(max_w, w)
                max_d = max(max_d, d)
                max_h = max(max_h, h)

        # 计算安全间距（保持紧凑但不重叠）
        margin_w = max(0.02, 0.1 * max_w)  # 减小到10%，更紧凑
        margin_d = max(0.02, 0.1 * max_d)
        cell_w = max_w + 2 * margin_w
        cell_d = max_d + 2 * margin_d

        # 删除临时对象
        for tmp_id in tmp_ids:
            p.removeBody(tmp_id)
        
        # 计算container需要的尺寸（应该能容纳所有pieces）
        # 获取container原始尺寸
        container_original_size = 0.0
        for i, (w, d, h) in enumerate(piece_dimensions):
            if "obj_8" in available_parts[i]:
                container_original_size = max(w, d, h)
                break
        
        # 计算所有非container pieces的总体积和平均单元尺寸
        non_container_volumes = [w * d * h for i, (w, d, h) in enumerate(piece_dimensions) 
                                if "obj_8" not in available_parts[i]]
        total_volume = sum(non_container_volumes)
        
        # 假设3x3x3=27个单元方块，估算单个方块边长
        unit_cube_size = (total_volume / 27) ** (1/3)
        
        # 3x3x3立方体需要的边长（加20%余量方便放入）
        required_container_size = unit_cube_size * 3 * 1.2
        
        # 计算container scale factor
        if container_original_size > 0:
            container_scale = required_container_size / container_original_size
        else:
            # 如果没找到container，根据最大piece尺寸估算
            container_scale = (max(max_w, max_d, max_h) * 3 * 1.2) / 0.3  # 假设原始container约0.3m
        
        # 限制scale在合理范围内
        container_scale = max(1.5, min(container_scale, 5.0))
        
        print(f"→ Container original size: ~{container_original_size:.3f}m")
        print(f"→ Estimated unit cube size: ~{unit_cube_size:.3f}m")
        print(f"→ Required container size (3x3x3): ~{required_container_size:.3f}m per side")
        print(f"→ Calculated container scale: {container_scale:.2f}x")
        print(f"→ Max piece dimensions: W={max_w:.3f}m, D={max_d:.3f}m, H={max_h:.3f}m")

        # ---------- 第二阶段：正式添加对象 ----------
        print(f"Loading {n} puzzle parts from {puzzle_base_path} ...")
        
        # 所有物体都直接放在地面（或桌面）上，避免从空中掉落
        ground_z = base_z + 0.05  # 稍微抬高一点避免卡进地面
        
        # Container放在一侧，pieces放在网格中
        # 计算网格的实际宽度，让container更靠近
        grid_width = (cols - 1) * cell_w / 2.0
        container_offset_x = grid_width - cell_w * 0.7  # container紧邻pieces网格

        for i, urdf_path in enumerate(available_parts):
            if "obj_8" in urdf_path:
                # Container单独放在一侧
                position = (base_x + container_offset_x, base_y, ground_z)
                piece_name = "container"
                properties = {"index": i, "is_container": True}
                scale_factor = container_scale
            else:
                # 其他pieces按网格分布在地面
                # 重新计算索引（跳过container）
                piece_idx = i if i < available_parts.index(urdf_path) else i - 1
                row = piece_idx // cols
                col = piece_idx % cols
                
                offset_x = (col - (cols - 1) / 2.0) * cell_w
                offset_y = (row - (rows - 1) / 2.0) * cell_d
                
                position = (base_x + offset_x, base_y + offset_y, ground_z)
                piece_name = f"piece_{i+1}"
                properties = {"index": i, "is_container": False}
                scale_factor = 1.0

            self.environment.add_object(
                object_name=piece_name,
                urdf_path=urdf_path,
                position=position,
                orientation=(0, 0, 0, 1),
                object_type="puzzle_piece",
                properties=properties,
                scale=scale_factor
            )

        print(f"✅ Loaded {n} puzzle parts successfully.")
        print(f"→ Container scaled by {container_scale:.2f}x to accommodate all pieces")
        print(f"→ Pieces spawned on ground in {rows}x{cols} grid")
        print(f"→ Grid cell size: ({cell_w:.3f}m, {cell_d:.3f}m), ground z={ground_z:.3f}m")
        print(f"→ Container position: x_offset={container_offset_x:.3f}m")
    
    def _create_simple_puzzle_pieces(self, start_index: int = 0) -> None:
        """Create simple puzzle pieces laid out on a (rows x cols) grid using row/col."""

        rows, cols = self.config.puzzle_size
        piece_colors = self._generate_piece_colors()

        spacing = self.config.piece_size * 1.25

        table_pos = self.environment.config.table_position
        table_x, table_y, table_z = table_pos

        capacity = rows * cols
        num_to_create = min(self.config.num_pieces - start_index, capacity - start_index)

        for i in range(num_to_create):
            abs_idx = start_index + i

            row = abs_idx // cols
            col = abs_idx % cols
            if row >= rows:
                break

            offset_x = (col - (cols - 1) / 2.0) * spacing
            offset_y = (row - (rows - 1) / 2.0) * spacing

            piece_name = f"piece_{abs_idx + 1}"

            self.environment.create_primitive_object(
                object_name=piece_name,
                shape_type="box",
                size=(
                    self.config.piece_size / 2,
                    self.config.piece_size / 2,
                    self.config.piece_size / 2,
                ),
                position=(table_x + offset_x, table_y + offset_y, table_z + 0.1),
                color=piece_colors[abs_idx] if abs_idx < len(piece_colors) else (1, 1, 1, 1),
                mass=0.05,
            )
            
    def _generate_piece_colors(self) -> List[Tuple[float, float, float, float]]:
        """Generate distinct colors for puzzle pieces."""
        colors = []
        num_pieces = self.config.num_pieces
            
        for i in range(num_pieces):
            # Create distinct colors using HSV color space
            hue = (i * 360 / num_pieces) / 360.0
            colors.append(self._hsv_to_rgba(hue, 0.8, 0.9))
        return colors
    
    def _hsv_to_rgba(self, h: float, s: float, v: float) -> Tuple[float, float, float, float]:
        """Convert HSV to RGBA."""
        import colorsys
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        return (r, g, b, 1.0)
    
    def _evaluate_success(self, task_results: List[TaskResult]) -> List[TaskResult]:
        """Evaluate task success based on puzzle completion."""
        if self.config.ruled_evaluation:
            return self._evaluate_success_ruled(task_results)
        else:
            return self._evaluate_success_agent(task_results)
        
    def _evaluate_success_ruled(self, task_results: List[TaskResult]) -> List[TaskResult]:
        for task_result in task_results:
                
            # Get the final observation (support both old and new formats)
            last_step = task_result.trajectory[-1]
            if isinstance(last_step, dict):
                # New format: {"response": str, "actions": List[Action], "observations": List[Observation]}
                observations = last_step.get("observations", [])
                if not observations:
                    task_result.success = False
                    continue
                final_observation = observations[-1]
                # observations are now Observation objects
                objects = final_observation.state.objects
            else:
                # Old format: (action, observation)
                final_observation = last_step[1]
                objects = final_observation.state.objects
            
            # Find the container and puzzle pieces
            container = None
            puzzle_pieces = []
            
            for obj in objects:
                if obj.properties.get('is_container', False):
                    container = obj
                elif obj.object_type == 'puzzle_piece' and not obj.properties.get('is_container', False):
                    puzzle_pieces.append(obj)
            
            # If no container found, task fails
            if container is None:
                task_result.success = False
                continue
            
            # Get container bounding box
            container_aabb_min, container_aabb_max = p.getAABB(container.object_id)
            
            # Add tolerance for checking (5mm tolerance on each side)
            tolerance = 0.005
            container_min = [container_aabb_min[0] - tolerance, 
                           container_aabb_min[1] - tolerance, 
                           container_aabb_min[2] - tolerance]
            container_max = [container_aabb_max[0] + tolerance, 
                           container_aabb_max[1] + tolerance, 
                           container_aabb_max[2] + tolerance]
            
            # Check if all pieces are inside the container
            all_pieces_inside = True
            
            for piece in puzzle_pieces:
                piece_aabb_min, piece_aabb_max = p.getAABB(piece.object_id)
                
                # Check if piece is completely inside container (with tolerance)
                if not (piece_aabb_min[0] >= container_min[0] and piece_aabb_max[0] <= container_max[0] and
                       piece_aabb_min[1] >= container_min[1] and piece_aabb_max[1] <= container_max[1] and
                       piece_aabb_min[2] >= container_min[2] and piece_aabb_max[2] <= container_max[2]):
                    all_pieces_inside = False
                    break
            
            # Check if all pieces fit snugly (no parts sticking out of container)
            # Calculate the union bounding box of all pieces
            if puzzle_pieces and all_pieces_inside:
                all_pieces_min = [float('inf'), float('inf'), float('inf')]
                all_pieces_max = [float('-inf'), float('-inf'), float('-inf')]
                
                for piece in puzzle_pieces:
                    piece_aabb_min, piece_aabb_max = p.getAABB(piece.object_id)
                    for i in range(3):
                        all_pieces_min[i] = min(all_pieces_min[i], piece_aabb_min[i])
                        all_pieces_max[i] = max(all_pieces_max[i], piece_aabb_max[i])
                
                # Check if combined pieces fit within container with tolerance
                fits_snugly = (all_pieces_min[0] >= container_min[0] and all_pieces_max[0] <= container_max[0] and
                              all_pieces_min[1] >= container_min[1] and all_pieces_max[1] <= container_max[1] and
                              all_pieces_min[2] >= container_min[2] and all_pieces_max[2] <= container_max[2])
                
                task_result.success = all_pieces_inside and fits_snugly
            else:
                task_result.success = False
            
        return task_results
    
    def _evaluate_success_agent(self, task_results: List[TaskResult]) -> List[TaskResult]:
        for task_result in task_results:
            # Get the final observation to count actual objects (support both old and new formats)
            last_step = task_result.trajectory[-1]
            if isinstance(last_step, dict):
                # New format: {"response": str, "actions": List[Action], "observations": List[Observation]}
                observations = last_step.get("observations", [])
                if not observations:
                    task_result.success = False
                    continue
                final_observation = observations[-1]
                # observations are now Observation objects
                objects = final_observation.state.objects
            else:
                # Old format: (action, observation)
                final_observation = last_step[1]
                objects = final_observation.state.objects
            
            # Count actual puzzle pieces (excluding container)
            num_pieces = len(objects) - 1
            
            task_success_criteria = f"""This is a 3D cube stacking puzzle task.

Task Configuration:
- Total puzzle pieces: {num_pieces}
- Task type: 3×3×3 cube assembly inside a container

Objective:
Fit all {num_pieces} puzzle pieces completely inside the designated container to form a complete 3×3×3 cube structure.

Success Indicators:
- All {num_pieces} puzzle pieces are completely contained within the container boundaries
- No pieces are protruding, sticking out, or extending beyond the container edges
- All pieces are properly positioned inside the container with no overflow
- The container successfully holds all pieces within its physical boundaries

Failure Indicators:
- One or more pieces are outside the container
- Pieces are sticking out or protruding from the container
- Pieces are scattered or lying far from the container
- Container boundaries are violated by any piece"""
            
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
        """Get general system prompt defining the AI agent's role and reasoning style for 3D puzzle-solving tasks."""
        
        return """You are an intelligent AI agent operating in a physics-based 3D simulation environment.

ROLE:
- You act as a spatial reasoning and problem-solving agent.
- You can observe 3D scenes, understand object geometry, and plan physical interactions.
- You make decisions through logical, step-by-step reasoning.

BEHAVIOR GUIDELINES:
- Think systematically and spatially when manipulating objects.
- Use observations to guide precise, physically valid actions.
- Consider stability, contact, and fit when arranging or assembling objects.
- Always reason before acting and adjust plans based on feedback from the environment.

TOOL USAGE:
- You interact with the environment **only through the available tools** provided to you.
- Use these tools to interact with objects as needed to progress toward the goal.

Your overall mission is to solve complex 3D manipulation and puzzle-solving tasks through careful reasoning and sequential physical actions.
"""

    def _get_initial_instruction(self) -> str:
        """Get concise task instruction for the current 3×3×3 puzzle."""
        
        num_pieces = len(self.environment.objects) - 1  # exclude the container
        
        return f"""3D CUBE STACKING PUZZLE

You have {num_pieces} 3D puzzle pieces and one container.

TASK:
Assemble all pieces into the container to form a solid 3×3×3 cube (27 unit cubes total).

GOAL:
- Fit every piece completely inside the container.
- No gaps, overlaps, or floating pieces.
- The final structure must be stable and form a perfect cube.

ACTION RULE:
- You can move or rotate one piece at a time.
- Continue placing pieces until the cube is fully assembled.
"""

