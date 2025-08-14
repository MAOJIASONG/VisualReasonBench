# PhyVPuzzle 交互式测试脚本运行指南

## 概述

`run_interactive_test.sh` 是一个全自动的 VLM（视觉语言模型）交互式基准测试运行器，用于测试模型在物理推理任务中的表现。

## 脚本功能

该脚本执行以下主要功能：
- 运行 VLM 与 PyBullet 物理环境的交互式测试
- 自动管理测试环境的设置和清理
- 保存测试结果、日志和截图
- 提供详细的测试报告和分析

## 使用方法

### 基本用法

```bash
# 运行所有交互式任务（默认）
./run_interactive_test.sh

# 或者指定具体任务类型
./run_interactive_test.sh [TASK_TYPE]
```

### 任务类型

- `domino` - 仅运行多米诺骨牌链式反应任务
- `luban` - 仅运行鲁班锁组装任务  
- `all` - 运行所有交互式任务（默认）

### 示例命令

```bash
# 运行鲁班锁任务
./run_interactive_test.sh luban

# 运行多米诺任务
./run_interactive_test.sh domino

# 运行所有任务
./run_interactive_test.sh all
```

### 获取帮助

```bash
./run_interactive_test.sh --help
```

## 运行流程详解

### 1. 前置检查阶段
- 检查 Python 环境是否可用
- 验证交互式脚本文件是否存在
- 确认所需依赖项

### 2. 环境准备阶段
- 创建当前状态的备份
- 设置测试结果目录
- 切换到 PhyVPuzzle 工作目录

### 3. 交互式测试阶段
测试过程包含以下循环：
```
VLM 接收环境初始图像和指令
    ↓
VLM 从可用动作中选择操作
    ↓
在 PyBullet 中执行动作
    ↓
VLM 接收更新后的环境图像
    ↓
重复直到任务完成或超时
    ↓
计算最终得分
```

### 4. 结果保存阶段
- 保存测试日志文件
- 保存 JSON 格式的结果数据
- 移动截图到结果目录
- 生成测试摘要报告

### 5. 环境清理阶段
- 清理临时文件和目录
- 保留测试结果用于分析
- 恢复原始工作目录

## 输出文件说明

### 结果目录结构
```
PhyVPuzzle/interactive_results/
├── interactive_[task]_[timestamp].log      # 测试日志
├── interactive_results_[task]_[timestamp].json  # 结果数据
└── vlm_interactive_screenshots/            # 测试截图
    ├── step_001.png
    ├── step_002.png
    └── ...
```

### 日志文件内容
- 详细的执行步骤记录
- VLM 的决策过程
- 环境状态变化
- 错误信息（如有）

### 结果 JSON 文件
包含以下关键信息：
- `final_score`: 最终得分
- `status`: 任务完成状态
- `steps_taken`: 实际执行步数
- `max_steps`: 最大允许步数
- 详细的步骤执行记录

## 注意事项

### API 密钥配置
脚本中预配置了 API 密钥，使用前请确认：
- API 密钥有效且有足够额度
- 网络连接正常

### 系统要求
- Python 3.x 环境
- 所需的 Python 依赖包已安装
- 足够的磁盘空间存储结果文件

### 运行时间
- 单个任务通常需要几分钟到十几分钟
- 具体时间取决于任务复杂度和网络延迟

## 故障排除

### 常见问题

1. **脚本权限错误**
   ```bash
   chmod +x run_interactive_test.sh
   ```

2. **Python 环境问题**
   - 确保 Python 3.x 已安装
   - 检查所需依赖包是否安装完整

3. **API 密钥问题**
   - 验证密钥有效性
   - 检查网络连接

4. **路径问题**
   - 确保在项目根目录运行脚本
   - 检查 PhyVPuzzle 子目录是否存在

### 查看详细错误信息
检查生成的日志文件：
```bash
tail -f PhyVPuzzle/interactive_results/interactive_[task]_[timestamp].log
```

## 结果分析

脚本会自动显示测试摘要，包括：
- 成功率
- 平均得分
- 任务总数
- 详细的执行步骤统计

通过分析结果文件，可以深入了解 VLM 在物理推理任务中的表现和决策过程。