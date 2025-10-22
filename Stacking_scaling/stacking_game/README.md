# 3D Polycube Stacking Game

一个3D多联通块装箱游戏,支持形状匹配、碰撞检测、支撑验证和**3D可视化**。

## 🌟 新功能

- **3D可视化** - 使用matplotlib生成精美的3D视图
- **智能初始化** - 所有piece初始放置在地面外,并随机旋转
- **交互式游戏** - 带3D可视化的交互式命令行界面
- **多视角显示** - 同时显示盒子内部和外部的pieces

## 功能特性

- ✅ 24个旋转支持 (立方体旋转群)
- ✅ 形状匹配算法 (自动识别piece的任意旋转)
- ✅ 碰撞检测
- ✅ 支撑检查 (防止悬空)
- ✅ 边界检查
- ✅ 移动和撤销功能
- ✅ 从puzzles_full_v9加载关卡
- ✅ **3D可视化 (matplotlib)**
- ✅ **随机旋转初始化**
- ✅ **交互式3D游戏界面**
- ✅ 简单的2D可视化
- ✅ 完整的测试套件

## 文件结构

```
stacking_game/
├── game_core.py         # 核心数据结构和基础功能
├── rotation.py          # 24个旋转矩阵生成
├── loader.py            # 关卡加载器
├── placement.py         # 形状匹配和放置判定
├── initialization.py    # 🆕 初始化工具(地面外+随机旋转)
├── visualizer_3d.py     # 🆕 3D可视化模块
├── game_cli.py          # CLI交互界面(2D)
├── game_3d.py           # 🆕 3D交互式游戏界面
├── test_game.py         # 核心功能测试
├── test_3d.py           # 🆕 3D可视化测试
├── example.py           # 简单示例
├── demo.py              # 自动求解演示
├── demo_3d.py           # 🆕 3D可视化演示
├── task.md              # 设计文档
└── README.md            # 本文档
```

## 快速开始

### 1. 安装依赖

```bash
pip install numpy matplotlib
```

### 2. 运行测试

```bash
cd /mnt/moonfs/wuyuhao-m2/wyh/VisualReasonBench/Stacking_scaling/stacking_game

# 核心功能测试
python test_game.py

# 3D可视化测试
python test_3d.py
```

预期输出:
```
✅ All tests passed!
```

### 3. 运行3D可视化演示

```bash
python demo_3d.py
```

这会生成多个3D可视化图片:
- `demo_initial_state.png` - 初始状态(pieces在盒子外)
- `demo_partial.png` - 部分完成状态
- `demo_complete.png` - 完成状态
- `demo_piece_rotations.png` - piece旋转展示
- `demo_3x3x3_initial.png` - 3x3x3初始状态

### 4. 运行简单示例

```bash
python example.py
```

### 5. 交互式游戏

#### 🖥️ 检查显示环境
```bash
python check_display.py
```

这个脚本会诊断您的环境并推荐合适的版本。

#### 选项A: GUI窗口模式 (需要图形界面)
```bash
python game_3d.py
```

**要求:**
- 本地有图形界面,或
- SSH连接时使用 `ssh -X` 启用X11转发
- 安装了tkinter: `sudo apt-get install python3-tk`

#### 选项B: 文件保存模式 (推荐用于服务器)
```bash
python game_3d_file.py
```

**特点:**
- ✅ 无需GUI环境
- ✅ 每次操作自动保存PNG图片到 `/tmp`
- ✅ 功能完全相同
- ✅ 适合SSH远程连接

**示例:**
```
> load 2x2x2 puzzle_001 42
✓ 已保存到: /tmp/puzzle_initial_0_152030.png

> place 0
Mode: rot
Position: 1 1 1
Rotation: 0
✓ 已保存到: /tmp/puzzle_placed_0_1_152045.png

> save my_state
✓ 已保存到: /tmp/puzzle_my_state_2_152100.png
```

#### 传统2D文本界面
```bash
python game_cli.py
```

示例会话:
```
> load 2x2x2 puzzle_001 42
=== Loaded puzzle: 2x2x2/puzzle_001 ===
Box size: 2x2x2
Pieces: 2

> show
[显示3D窗口]

> place 0
Mode [cells/rot]: rot
Position (x y z): 1 1 1
Rotation (0-2): 0
✓ Piece placed successfully
[更新3D窗口]

> status
=== Status ===
Box: 2x2x2
Occupied: 4/8 cells
Placed: 1 pieces
Unplaced: 1 pieces
```

### 游戏命令

- `help` - 显示帮助信息
- `list` - 列出所有可用关卡
- `load <size> <puzzle_id>` - 加载关卡 (例: `load 2x2x2 puzzle_001`)
- `state` - 显示当前游戏状态
- `view` - 显示2D俯视图
- `piece <id>` - 查看piece信息
- `place <id>` - 交互式放置piece
- `pickup <id>` - 取出已放置的piece
- `quit` / `exit` - 退出游戏

## 游戏示例

```
> load 2x2x2 puzzle_001
=== Loaded puzzle: 2x2x2/puzzle_001 ===
Box size: 2x2x2
Pieces: 2

> state
=== Current State ===
Box: 2x2x2
Occupied: 0/8 cells
Placed pieces: 0
Unplaced pieces: 2

Unplaced:
  Piece 0: 4 cells
  Piece 1: 4 cells

> place 0
Placing piece 0 (4 cells)
Enter target cells (format: x,y,z), one per line
Enter empty line when done:
Cell 1/4: 1,1,1
Cell 2/4: 2,1,1
Cell 3/4: 1,2,1
Cell 4/4: 2,2,1
✓ Piece placed successfully

> view
=== Top View (Z-layers) ===

Layer z=2:
   .  .

Layer z=1:
  [0][0]
  [0][0]

> state
=== Current State ===
Box: 2x2x2
Occupied: 4/8 cells
Placed pieces: 1
Unplaced pieces: 1
```

## 核心算法

### 1. 形状匹配

使用规范化签名算法:
1. 预处理: 为每个piece生成24个旋转的规范形签名
2. 实时匹配: 将玩家选择的目标格子规范化后与签名库比对
3. 反推变换: 找到匹配的旋转后,计算平移向量

### 2. 放置判定流程

1. **形状匹配** - 检查目标格子是否与piece的某个旋转匹配
2. **边界检查** - 确保所有格子在盒子范围内
3. **碰撞检查** - 确保目标位置没有被占用
4. **支撑检查** - 至少有一个体素接触底部或其他piece
5. **提交放置** - 更新游戏状态

### 3. 支撑检查

一个piece被认为有支撑,当且仅当至少满足以下条件之一:
- 至少有一个体素在底部 (z=1)
- 至少有一个体素的六邻域中存在其他piece的体素

## 坐标系统

- **对外接口**: 1-based坐标 (x, y, z ∈ [1..A]×[1..B]×[1..C])
- **内部实现**: 0-based坐标用于计算
- **重力方向**: -z方向,底面为z=1

## 扩展功能 (TODO)

- [ ] 3D可视化 (使用matplotlib或plotly)
- [ ] 自动求解器
- [ ] 提示系统 (显示可能的合法放置)
- [ ] 撤销/重做栈
- [ ] 保存/加载游戏进度
- [ ] GUI界面

## 开发说明

### 添加新的测试

在 `test_game.py` 中添加新的测试函数,并在 `run_all_tests()` 中注册。

### 扩展可视化

可以在 `game_cli.py` 中的 `visualize_2d()` 方法基础上添加更多视图。

### 性能优化

- 形状签名使用字符串比较,可以优化为哈希比较
- 可以使用numpy数组批量处理碰撞检测
- 支撑检查可以缓存邻接关系

## License

MIT
