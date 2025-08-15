

# Task: Merge, Integrate, and Extend VLM Multi-turn Evaluation Framework

## 1. 文件整合与合并

* **输入文件**：

  * **VisualReasonBench**: `/mnt/moonfs/wuyuhao-m2/wyh/VisualReasonBench`
  * **PhyVPuzzle**: `/mnt/moonfs/wuyuhao-m2/wyh/PhyVPuzzle`
* **目标**：
  将 **PhyVPuzzle** 的功能模块、数据集和相关代码合并到 **VisualReasonBench** 中，保持代码结构清晰、可维护。
  合并后所有功能统一在 `VisualReasonBench` 下维护与运行。

---

## 2. 多轮次 VLM 操作的输入/输出记录

* 在 VLM 执行多轮推理、操作环境过程中，必须记录：每一个model 的每一个task的

  1. **每轮 Input**（模型输入内容）
  2. **每轮 Output**（模型输出内容）
  3. **环境图片结果**（操作前/后的环境状态截图）
* 报错信息必须**详细且可复现**，用于后续定义交互格式和 Debug。

---

## 3. OpenRouter 接口调用设置

* 使用本地 `openai_key` 文件读取 API Key：

  ```python
  import os
  from openai import OpenAI

  api_key = "你的密钥"  # 本地加载
  base_url = "https://openai.app.msh.team/v1"

  client = OpenAI(api_key=api_key, base_url=base_url)

  # 查看支持的模型
  models = client.models.list()

  chat_completion = client.chat.completions.create(
      messages=[
          {
              "role": "user",
              "content": "Say this is a test",
          }
      ],
      model="o4-mini"
  )
  ```
* 注意确保 **API Key 从本地安全读取**，不要硬编码。

---

## 4. 模型功能要求

* **模型指令**

  * 能调用工具（Function） 操作于pybullet 环境
  * 工具作用于环境并能正常使用
* **任务合并**
  * 选择可以对于 环境操作的原子操作 比如 ： 
    Puzzle (Prismic movement)
    Input func:
    Observe (angle)
    Left, right, up, down, front, back
    Move (colour_idx, distance, axis)
    Press (colour_idx, axis)
    Out func:
    Current view of Image
    Lego/Domino (Torque movement)
    Input func:
    Observe (angle)
    Pick(3d_pos, type, colour_idx)
    Move (3d_pos)
    Turn (rotation_angle)
    Out func:
    Current view of Image
# 可以一定参考 /mnt/moonfs/wuyuhao-m2/wyh/VisualReasonBench/run_interactive_test.sh
* **评测指标 (Eval metric)**

  * 能与环境交互并计算指标 -- 参考metric 部分
 

---

## 5. 最终 Demo

* 使用一个**最简单的多米诺场景** 进行测试，验证：

  1. 模型多轮输入输出记录正常
  2. 工具调用流程正常
  3. 环境状态图片记录完整
  4. Eval metric 与环境交互正常

---



## 6. 验收标准

* **功能集成完成**：`PhyVPuzzle` 与 `VisualReasonBench` 无缝整合
* **交互可追踪**：每轮 VLM 调用都有 input、output、环境截图
* **工具可用**：模型能调用 Function 并在环境中执行动作
* **评测可运行**：Eval metric 与环境无错误交互
* **Demo 可运行**：最简单多米诺，推到每一个即可测试完整通过
* **代码清晰有逻辑，保存的中间状态干净方便读取** 

