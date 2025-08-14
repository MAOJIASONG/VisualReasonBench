# VLM Multi-turn Evaluation Framework Integration TODO List

本文档根据 `Task.md` 的要求，结合 `VisualReasonBench` 的现有代码结构，制定了详细的整合与功能扩展计划。

---

## 阶段一：核心功能增强与集成 (Milestone 1)

**目标**：在现有框架基础上，增强 VLM 能力、完善环境交互，并将 `PhyVPuzzle` 的独有资源无缝集成。

- [ ] **1.1 升级 VLM 处理器以支持 OpenRouter 和工具调用**
  - **当前状态**：`vllm_processor.py` 中的 `OpenAIVLLMProcessor` 仅支持标准 OpenAI API，缺乏工具调用能力和灵活的 `base_url` 配置。
  - **操作**：
    1.  **增强 `OpenAIVLLMProcessor`**：
        -   在 `__init__` 方法中增加 `base_url` 参数，以支持 `OpenRouter`。
        -   修改 `process_input` 方法，使其能够接收一个 `tools` 参数（包含工具的 JSON Schema 描述），并在 API 调用中传递 `tools` 和 `tool_choice`。
        -   重构返回值，使其能同时返回`content`（模型的自然语言回复）和`tool_calls`（模型请求的工具调用）。
    2.  **实现安全的 API Key 管理**：在 `OpenAIVLLMProcessor` 中，实现从本地文件（如 `~/.config/openai_key` 或项目根目录下的 `.env` 文件）安全加载 API Key 的逻辑，移除代码中的硬编码风险。

- [ ] **1.2 迁移并整合 `PhyVPuzzle` 的物理模型资源**
  - **当前状态**：`VisualReasonBench` 缺少 `PhyVPuzzle` 中的多米诺骨牌、鲁班锁等关键物理模型。
  - **操作**：
    1.  将 `/mnt/moonfs/wuyuhao-m2/wyh/PhyVPuzzle/phobos_models` 目录下的所有模型资源，完整复制到 `/mnt/moonfs/wuyuhao-m2/wyh/VisualReasonBench/src/phyvpuzzle/environment/phobos_models`。
    2.  检查并更新环境代码（`physics_env.py`），确保资源路径正确，模型能被成功加载。

- [ ] **1.3 统一并完善项目依赖**
  - **当前状态**：两个项目的 `requirements.txt` 可能存在差异和冲突。
  - **操作**：
    1.  对比 `PhyVPuzzle/requirements.txt` 和 `VisualReasonBench/requirements.txt`。
    2.  合并两者内容，去除重复项，并统一版本号，最终更新 `VisualReasonBench` 的 `requirements.txt`。
    3.  在一个新的虚拟环境中通过 `pip install -r requirements.txt` 验证所有依赖均可正常安装。

---

## 阶段二：实现端到端的数据追踪与日志记录 (Milestone 2)

**目标**：建立一套完善的日志系统，详细记录 VLM 交互的每一个环节，确保实验的可复现性。

- [ ] **2.1 设计并实现结构化的日志记录器**
  - **需求**：需要一个独立的日志模块来管理实验数据。
  - **操作**：
    1.  在 `/mnt/moonfs/wuyuhao-m2/wyh/VisualReasonBench/src/phyvpuzzle/utils/` 目录下创建一个新的 `logger.py` 文件。
    2.  在该模块中实现一个 `ExperimentLogger` 类，负责为每次实验（按模型、任务和时间戳）创建唯一的输出目录。目录结构建议如下：
        ```
        logs/{model_name}/{task_name}/{timestamp}/
        ├── trial_info.json
        └── rounds/
            ├── round_01/
            │   ├── input.json
            │   ├── output.json
            │   ├── pre_action.png
            │   └── post_action.png
            └── ...
        ```
- [ ] **2.2 将日志记录集成到核心 Pipeline**
  - **当前状态**：`pipeline.py` 中的日志记录较为简单，缺少详细的输入输出和图像记录。
  - **操作**：
    1.  在 `PhysicalReasoningPipeline` 的 `__init__` 方法中，实例化 `ExperimentLogger`。
    2.  在 `_execute_single_step` 方法的关键位置插入日志记录调用：
        -   **VLM 输入**：在调用 `vllm_processor.process_input` 之前，记录完整的 `messages`（包含 prompt 和图像数据）到 `input.json`。
        -   **VLM 输出**：记录 `vllm_processor` 返回的原始响应（包括 `content` 和 `tool_calls`）到 `output.json`。
        -   **环境截图**：在执行工具调用前后，分别调用 `environment.render()` 并保存为 `pre_action.png` 和 `post_action.png`。
    3.  **错误处理**：在 `try...except` 块中捕获异常，并将详细的错误信息和 traceback 记录到日志目录中的 `errors.log` 文件。

---

## 阶段三：定义任务与实现工具 (Milestone 3)

**目标**：基于 `BaseTask` 实现一个具体的多米诺任务，并开发相应的原子操作工具，打通 VLM 与环境的交互闭环。

- [ ] **3.1 实现具体的多米诺任务类 (DominoTask)**
  - **当前状态**：`base_task.py` 定义了任务的抽象接口，但缺少具体实现。
  - **操作**：
    1.  在 `/mnt/moonfs/wuyuhao-m2/wyh/VisualReasonBench/src/phyvpuzzle/tasks/` 目录下创建 `domino_task.py`。
    2.  在该文件中定义 `DominoTask` 类，继承自 `BaseTask`，并实现其所有抽象方法：
        -   `setup_task`: 负责在 PyBullet 环境中加载多米诺骨牌模型。
        -   `get_task_description`: 提供清晰的任务描述，例如“推倒所有多米诺骨牌”。
        -   `check_completion`: 检查所有骨牌是否都已倒下。
        -   `evaluate_state`: 根据倒下的骨牌数量计算当前得分。

- [ ] **3.2 定义并实现原子操作工具集**
  - **需求**：需要一套 VLM 可调用的、与环境交互的原子操作。
  - **操作**：
    1.  在 `physics_env.py` 中，定义一系列简单的 Python 函数作为原子操作，例如：
        -   `pick(object_id)`: 模拟抓取一个物体。
        -   `place(object_id, position, orientation)`: 将物体放置到指定位置。
        -   `push(object_id, force, direction)`: 对物体施加一个力。
    2.  为每个函数编写详细的 docstring，并使用 `inspect` 或其他库自动生成其 JSON Schema 描述。这将作为 `tools` 参数传递给 VLM。

- [ ] **3.3 在 Pipeline 中集成工具调用逻辑**
  - **当前状态**：`pipeline.py` 的决策逻辑只区分 `finish` 和 `action`，无法处理具体的工具调用。
  - **操作**：
    1.  修改 `_execute_single_step` 方法，在 VLM 返回 `tool_calls` 后：
        -   遍历 `tool_calls` 列表。
        -   根据 `function.name` 动态地从 `physics_env` 实例中查找并调用对应的工具函数。
        -   将 `function.arguments`（一个 JSON 字符串）解析为字典，并作为参数传递给工具函数。
        -   将工具函数的执行结果追加到 `messages` 历史中，以便 VLM 进行下一步决策。

---

## 阶段四：端到端测试与评测 (Milestone 4)

**目标**：创建一个完整的演示脚本，运行多米诺任务，并确保所有功能模块（日志、工具调用、评测）都能协同工作。

- [ ] **4.1 整合评测模块**
  - **当前状态**：评测逻辑 (`evaluation/metrics.py`) 与任务状态 (`tasks/base_task.py`) 分离。
  - **操作**：
    1.  确保 `DominoTask` 的 `evaluate_state` 方法能够准确反映任务进度。
    2.  在 `PhysicalReasoningPipeline` 的 `execute_task` 方法结束时，调用 `task.get_result()`，并将返回的 `TaskResult` 对象保存到日志目录的 `trial_info.json` 中。

- [ ] **4.2 创建并运行最终的演示脚本**
  - **需求**：需要一个顶层脚本来启动和运行整个流程。
  - **操作**：
    1.  在 `VisualReasonBench` 根目录下创建一个 `run_domino_demo.py` 脚本。
    2.  该脚本应完成以下工作：
        -   初始化 `PipelineConfig` 和 `TaskConfiguration`。
        -   创建 `PhysicalReasoningPipeline` 和 `DominoTask` 实例。
        -   调用 `pipeline.execute_task(task)` 来运行实验。
        -   打印最终的评测结果。

- [ ] **4.3 最终验收**
  - **操作**：执行 `python run_domino_demo.py` 并检查：
    -   [ ] **日志完整性**：确认 `logs` 目录下生成了完整的实验数据，包括输入/输出的 JSON 文件和每个动作前后的截图。
    -   [ ] **工具调用**：确认 VLM 能够成功调用 `push` 或其他工具，并在环境中产生预期的物理效果。
    -   [ ] **任务完成度**：确认 `DominoTask` 能够正确判断任务是否完成，并计算出合理的最终得分。
    -   [ ] **代码质量**：代码结构清晰，逻辑明确，易于扩展。
