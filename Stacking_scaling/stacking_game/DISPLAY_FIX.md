# 显示问题解决方案

## 问题

使用 `show` 命令时没有看到可视化窗口。

## 原因

您的环境是**无GUI的服务器环境**（通常是SSH远程连接），matplotlib无法打开图形窗口。

诊断结果：
```bash
$ python check_display.py
✗ DISPLAY未设置 (无法显示GUI窗口)
✗ 没有可用的交互式后端 (TkAgg, Qt5Agg等)
```

## ✅ 解决方案

### 方案1: 使用文件保存模式 (推荐 ⭐)

使用 `game_3d_file.py` 代替 `game_3d.py`：

```bash
python game_3d_file.py
```

**特点:**
- ✅ 每次操作**自动保存**PNG图片
- ✅ 无需任何GUI环境
- ✅ 功能**完全相同**
- ✅ 图片保存在 `/tmp` 目录

**示例会话:**
```
$ python game_3d_file.py

> load 2x2x2 puzzle_001 42
=== Loaded puzzle: 2x2x2/puzzle_001 ===
✓ 已保存到: /tmp/puzzle_initial_0_152030.png

> place 0
Mode [cells/rot]: rot
Position (x y z): 1 1 1
Rotation (0-2): 0
✓ Piece placed successfully
✓ 已保存到: /tmp/puzzle_placed_0_1_152045.png

> save checkpoint
✓ 已保存到: /tmp/puzzle_checkpoint_2_152100.png

> status
Box: 2x2x2
Occupied: 4/8 cells

> quit
```

**查看生成的图片:**
```bash
# 列出所有图片
ls -lt /tmp/puzzle_*.png

# 下载到本地查看 (如果是远程服务器)
scp user@server:/tmp/puzzle_*.png ./
```

### 方案2: 使用demo生成图片

```bash
# 生成预设的演示图片
python demo_3d.py

# 生成的图片在 /tmp 目录:
# - demo_initial_state.png
# - demo_partial.png
# - demo_complete.png
# - demo_piece_rotations.png
# - demo_3x3x3_initial.png
```

### 方案3: 启用X11转发 (如果要用GUI)

如果您想使用 `game_3d.py` 的GUI窗口模式：

```bash
# SSH连接时启用X11转发
ssh -X user@server

# 或在 ~/.ssh/config 中添加:
Host myserver
    ForwardX11 yes
    ForwardX11Trusted yes

# 然后安装tkinter (Ubuntu/Debian)
sudo apt-get install python3-tk

# 测试
python game_3d.py
```

## 🎯 推荐工作流程

### 在服务器上使用 game_3d_file.py

```bash
# 1. 运行游戏
python game_3d_file.py

# 2. 玩游戏 (所有操作自动保存图片)
> load 2x2x2 puzzle_001
> place 0
> place 1
> status
> quit

# 3. 下载图片到本地
scp user@server:/tmp/puzzle_*.png ./local_folder/

# 4. 本地查看图片
open ./local_folder/puzzle_*.png
```

## 📋 命令对比

| 功能 | game_3d.py | game_3d_file.py | game_cli.py |
|------|------------|-----------------|-------------|
| 3D可视化 | ✅ 窗口 | ✅ 保存图片 | ❌ |
| 需要GUI | ✅ 是 | ❌ 否 | ❌ 否 |
| 实时更新 | ✅ 是 | ✅ 自动保存 | ❌ |
| 所有游戏功能 | ✅ | ✅ | ✅ |
| 适用环境 | 本地/X11 | **所有环境** | 所有环境 |

## 🔍 诊断工具

运行诊断脚本检查您的环境：

```bash
python check_display.py
```

这会告诉您:
- matplotlib版本
- 可用的后端
- DISPLAY环境变量
- 推荐使用哪个版本

## 📚 相关文件

- `game_3d.py` - GUI窗口版本
- `game_3d_file.py` - 文件保存版本 ⭐
- `game_cli.py` - 传统2D文本版本
- `check_display.py` - 环境诊断脚本
- `demo_3d.py` - 自动生成演示图片

## 💡 小贴士

1. **文件保存模式的优势:**
   - 可以保存游戏进度快照
   - 高质量PNG图片(150 DPI)
   - 文件名包含操作类型和时间戳
   - 可以回顾整个游戏过程

2. **文件命名规则:**
   ```
   puzzle_<label>_<counter>_<timestamp>.png

   例如:
   puzzle_initial_0_152030.png       # 初始状态
   puzzle_placed_0_1_152045.png      # 放置piece 0后
   puzzle_checkpoint_2_152100.png    # 手动保存
   ```

3. **清理旧文件:**
   ```bash
   # 删除所有临时puzzle图片
   rm /tmp/puzzle_*.png
   ```

## ✅ 总结

- **您的情况:** 服务器环境,无GUI
- **推荐方案:** 使用 `game_3d_file.py`
- **效果:** 功能完全相同,图片自动保存
- **体验:** 完整的3D可视化 + 无缝游戏体验

现在您可以愉快地玩游戏了! 🎮
