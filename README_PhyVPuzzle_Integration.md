# PhyVPuzzle-VLM Integration

这是一个完整的物理拼图环境与视觉语言模型(VLM)的评测集成，支持复杂的观察-动作-反馈循环。

## 🎯 项目概述

PhyVPuzzle环境已成功整合到`src/phyvpuzzle/environment`中，实现了：

- **复杂物理拼图**：鲁班锁、宝塔等需要多步操作的拼图任务
- **VLM评测框架**：支持GPT-4o等视觉语言模型的完整评测流程
- **观察-动作-反馈循环**：模型观察→输出动作→环境执行→获得反馈→继续循环
- **多维度成功判断**：连接性、稳定性、配置正确性等综合评估

## 📁 项目结构

```
src/phyvpuzzle/environment/
├── __init__.py                 # 模块导出
├── physics_env.py              # 基础物理环境框架
├── phyvpuzzle_env.py           # PhyVPuzzle专用环境
├── vlm_benchmark.py            # VLM评测控制器
├── success_metrics.py          # 成功判断和评估指标
├── example_usage.py            # 使用示例
└── phobos_models/              # 3D模型和URDF文件
    ├── lego-pagoda-setup-v2/   # 宝塔拼图模型
    ├── luban-simple-prismatic/ # 鲁班锁模型
    └── ...

# 测试和演示脚本
test_gpt4o_phyvpuzzle.py       # GPT-4o API测试
test_gpt4o_real_env.py         # 真实环境测试
test_gpt4o_robust.py           # 健壮性测试
demo_phyvpuzzle_integration.py # 离线集成演示
```

## 🚀 快速开始

### 1. 离线演示（推荐开始）

运行完整的集成演示，无需外部API：

```bash
python demo_phyvpuzzle_integration.py
```

这将展示完整的VLM-环境交互流程，生成可视化结果。

### 2. GPT-4o API测试

确保`.env`文件包含API配置：

```bash
# .env
base_url = "http://your-api-endpoint/v1/"
api_key = "your-api-key"
```

运行API测试：

```bash
# 基础API连接测试
python test_gpt4o_robust.py

# 模拟环境测试
python test_gpt4o_phyvpuzzle.py
```

### 3. 真实物理环境测试

安装PyBullet依赖：

```bash
pip install pybullet
```

运行真实环境测试：

```bash
python test_gpt4o_real_env.py
```

## 🧩 支持的拼图类型

### 鲁班锁 (Luban Lock)
- **目标**：拆解或组装互锁的木制块
- **动作**：移动、旋转、滑动、提升拼图块
- **评估维度**：连接性、空间配置、稳定性、完整性

### 宝塔 (Pagoda)
- **目标**：堆叠构建稳定的塔形结构
- **动作**：移动拼图块、调整中心杆位置
- **评估维度**：结构稳定性、高度达成、平衡质量、对称性

## 🔄 VLM交互流程

```
1. 环境观察 → 渲染RGB图像 + 状态描述
2. VLM分析 → 分析图像和状态，输出动作决策
3. 动作解析 → 将VLM输出解析为结构化命令
4. 环境执行 → 在物理仿真中执行动作
5. 反馈生成 → 提供执行结果和状态变化
6. 成功评估 → 多维度评估任务完成状态
7. 循环继续 → 直到任务完成或达到限制
```

## 📊 评估指标

### 鲁班锁评估
- **连接评分 (40%)**：拼图块间的物理连接质量
- **配置评分 (30%)**：空间排列的紧密度和对齐度
- **稳定评分 (20%)**：结构的物理稳定性
- **完整评分 (10%)**：所有拼图块的有效性

### 宝塔评估
- **稳定评分 (40%)**：结构的动态稳定性
- **高度评分 (30%)**：达到的建筑高度
- **平衡评分 (20%)**：重心和支撑点平衡
- **对称评分 (10%)**：几何对称性

## 🛠️ 使用API

### 基础环境使用

```python
from phyvpuzzle.environment import create_luban_lock_environment

# 创建环境
env = create_luban_lock_environment("./phobos_models", gui=True)
env.setup_environment()

# 获取观察
observation = env.get_observation_for_vlm()
# observation包含:
# - image: PIL图像
# - state_description: 文本状态描述 
# - available_actions: 可用动作列表

# 执行动作
from phyvpuzzle.core.translator import EnvironmentCommand
command = EnvironmentCommand("move_piece", {
    "piece_id": "obj_1",
    "target_position": [0.1, 0.2, 0.6]
})
success = env.execute_command(command)

# 检查成功状态
is_complete = env.is_task_complete()
success, reason = env.get_success_status()
```

### VLM评测使用

```python
from phyvpuzzle.environment import VLMBenchmarkController, VLMBenchmarkConfig, PuzzleType

# 配置评测
config = VLMBenchmarkConfig(
    model_name="gpt-4o",
    puzzle_type=PuzzleType.LUBAN_LOCK,
    max_steps=50,
    time_limit=300.0,
    output_dir="./results"
)

# 运行评测
controller = VLMBenchmarkController(config, your_vlm_processor)
result = controller.run_benchmark()

print(f"成功: {result.success}")
print(f"步数: {result.total_steps}")
print(f"效率分数: {result.efficiency_score}")
```

## 📈 演示结果

运行`demo_phyvpuzzle_integration.py`后，查看生成的结果：

- `demo_results/step_XX_image.png` - 每步的可视化图像
- `demo_results/step_XX_data.json` - 每步的详细数据
- `demo_results/complete_demo_results.json` - 完整评测结果

示例输出：
```json
{
  "final_results": {
    "success": true,
    "reason": "Puzzle successfully solved", 
    "total_steps": 4
  },
  "step_details": [
    {
      "step": 1,
      "action": "move_piece",
      "parameters": {"piece_id": "obj_1", "target_position": [0.2, 0.3, 0.6]},
      "status": "Success"
    }
  ]
}
```

## 🔧 技术特性

### 多模态输入处理
- **视觉输入**：512x512 RGB图像，相机可配置
- **状态输入**：结构化文本描述，包含位置、旋转、速度信息
- **历史上下文**：最近几步的动作和结果

### 动作空间
- `move_piece`: 移动拼图块到目标位置
- `rotate_piece`: 绕轴旋转拼图块
- `slide_piece`: 沿方向滑动拼图块
- `lift_piece`: 垂直提升拼图块
- `insert_piece`: 插入拼图块到槽位
- `remove_piece`: 从位置移除拼图块
- `check_solution`: 检查拼图解决状态

### 物理仿真
- **引擎**：PyBullet 3D物理仿真
- **碰撞检测**：精确的mesh-based碰撞
- **重力模拟**：真实的重力和动力学
- **关节约束**：复杂的机械约束

## 🎯 应用场景

1. **VLM能力评估**：测试视觉推理和序列规划能力
2. **机器人控制**：评估具身AI的操作技能
3. **多步推理**：评估复杂任务的分解和执行
4. **物理直觉**：测试对物理约束和力学的理解

## 🚧 后续开发

### 短期目标
- [ ] 安装PyBullet依赖并测试真实物理环境
- [ ] 修复API连接问题，测试GPT-4o集成
- [ ] 优化提示工程，提高求解成功率
- [ ] 添加更多拼图类型和难度级别

### 长期目标
- [ ] 支持多模态模型（Claude、Gemini等）
- [ ] 添加人机协作模式
- [ ] 实现强化学习baseline
- [ ] 部署Web界面进行在线评测

## 📝 注意事项

1. **依赖管理**：某些导入可能因为缺少transformers组件而失败，这是正常的
2. **API可用性**：当前API端点可能不稳定，建议先运行离线演示
3. **性能优化**：大型模型文件加载可能较慢，建议先使用简化模型测试
4. **GPU要求**：PyBullet渲染建议使用GPU，但CPU也可运行

## 🤝 贡献

该集成框架为VLM物理推理评测提供了完整的基础设施。欢迎基于此框架扩展更多拼图类型、评估指标和模型支持。

---