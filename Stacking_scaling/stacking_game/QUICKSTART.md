# 🎮 快速入门指南

## 5分钟上手3D Polycube Stacking Game

### 1️⃣ 安装依赖 (30秒)

```bash
pip install numpy matplotlib
```

### 2️⃣ 验证安装 (1分钟)

```bash
cd /mnt/moonfs/wuyuhao-m2/wyh/VisualReasonBench/Stacking_scaling/stacking_game

# 运行核心测试
python test_game.py

# 运行3D可视化测试
python test_3d.py
```

**预期结果:** 看到 `✅ All tests passed!`

### 3️⃣ 查看3D演示 (2分钟)

```bash
# 生成3D可视化图片
python demo_3d.py

# 查看生成的图片
ls -lh /tmp/demo_*.png
```

**生成的图片:**
- 📊 `demo_initial_state.png` - 初始状态(pieces在地上)
- 📊 `demo_partial.png` - 部分完成
- 📊 `demo_complete.png` - 完成!
- 📊 `demo_piece_rotations.png` - 旋转展示
- 📊 `demo_3x3x3_initial.png` - 3x3x3初始状态

### 4️⃣ 玩交互式游戏 (无限分钟 😊)

```bash
# 启动3D交互式游戏
python game_3d.py
```

**示例游戏流程:**

```
> load 2x2x2 puzzle_001 42
✓ Loaded puzzle with 2 pieces
[自动显示3D窗口]

> status
Box: 2x2x2
Occupied: 0/8 cells
Placed: 0 pieces
Unplaced: 2 pieces

> place 0
Mode [cells/rot]: rot
Position (x y z): 1 1 1
Rotation (0-2): 0
✓ Piece placed successfully
[3D窗口自动更新]

> place 1
Mode [cells/rot]: rot
Position (x y z): 1 1 2
Rotation (0-2): 0
✓ Piece placed successfully
[3D窗口自动更新]

> status
🎉 PUZZLE COMPLETE! 🎉

> quit
Goodbye!
```

---

## 🎯 主要命令

| 命令 | 说明 | 示例 |
|------|------|------|
| `load <size> <id> [seed]` | 加载puzzle | `load 2x2x2 puzzle_001 42` |
| `show` | 显示/刷新3D窗口 | `show` |
| `status` | 查看当前状态 | `status` |
| `place <id>` | 放置piece | `place 0` |
| `pickup <id>` | 取出piece | `pickup 0` |
| `random <id>` | 随机化旋转 | `random 1` |
| `help` | 查看帮助 | `help` |
| `quit` | 退出游戏 | `quit` |

---

## 🌟 核心特性

### ✅ 3D可视化
- 彩色体素渲染
- 双视图(盒子内+外)
- 实时更新
- 高质量导出

### ✅ 智能初始化
- pieces在盒子外地面
- 随机旋转
- 自动布局
- 可复现(种子)

### ✅ 两种放置模式

**模式1: 按格子 (cells)**
```
> place 0
Mode: cells
Cell 1/4: 1,1,1
Cell 2/4: 2,1,1
Cell 3/4: 1,2,1
Cell 4/4: 2,2,1
```

**模式2: 按旋转 (rot) - 推荐!**
```
> place 0
Mode: rot
Position (x y z): 1 1 1
Rotation (0-23): 0
```

---

## 📚 更多示例

### 简单Python示例
```bash
python example.py
```
展示如何用代码创建和求解puzzle

### 自动求解演示
```bash
python demo.py
```
自动加载和求解2x2x2 puzzle

### 传统2D界面
```bash
python game_cli.py
```
文本版交互界面(无3D窗口)

---

## 🐛 常见问题

### Q: matplotlib窗口不显示?
**A:** 检查显示环境,或使用demo_3d.py生成图片

### Q: 如何保存当前状态的图片?
**A:** 在代码中使用:
```python
from visualizer_3d import visualize_state_3d, save_visualization
fig = visualize_state_3d(state)
save_visualization(fig, "my_puzzle.png", dpi=200)
```

### Q: 如何查看所有可用puzzle?
**A:** 在game_3d.py中输入`list`命令(未实现),或查看:
```bash
ls /mnt/moonfs/wuyuhao-m2/wyh/VisualReasonBench/Stacking_scaling/puzzles_full_v9/
```

### Q: 放置失败怎么办?
**A:** 检查错误信息:
- `ShapeMismatch` - 形状不匹配,试试其他旋转
- `OutOfBounds` - 超出边界,检查坐标
- `Collision` - 碰撞,位置已被占用
- `Floating` - 悬空,需要支撑

---

## 🚀 下一步

1. **学习算法** - 查看`task.md`了解设计原理
2. **阅读代码** - 从`game_core.py`开始
3. **修改扩展** - 添加新功能或改进现有功能
4. **创建关卡** - 设计自己的puzzle

---

## 📞 获取帮助

- **文档:** `README.md` (完整功能说明)
- **设计:** `task.md` (算法和架构)
- **更新:** `UPDATE_SUMMARY.md` (3D功能说明)
- **总结:** `SUMMARY.md` (项目总结)

**Have Fun! 🎉**
