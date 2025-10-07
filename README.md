# PhyVPuzzle: 物理视觉推理基准（以 Domino 示例为例）

> 通过 3D 物理仿真评测 VLM 的感知、推理、规划、执行与视觉判断能力。

## 目录

- 快速开始
- Domino 示例配置（YAML）
- 运行示例（example / CLI）
- 运行基准（Benchmark：代码 / CLI）
- 扩展：如何添加新环境（Environment）
- 扩展：如何添加新任务（Task）
- 指标与结果产物

---

## 快速开始

### 依赖与环境变量

```bash
# 克隆并进入项目
git clone https://github.com/your-org/VisualReasonBench.git
cd VisualReasonBench

# 安装依赖
conda install -c conda-forge pybullet
pip install vllm==0.10.2
pip install -e .

# 设置 OpenAI API（任选其一）
export OPENAI_API_KEY="your-openai-api-key"
# 或创建 .env 文件
printf "OPENAI_API_KEY=your-openai-api-key\n" > .env
```

> 提示：本仓库的示例与 CLI 默认使用 OpenAI 模型作为 Agent 和 Judge；你也可以在 YAML 中改为自定义 Agent。

---

## Domino 示例配置（YAML）

示例配置文件：`eval_configs/domino_quick.yaml`

```yaml
runner:
  experiment_name: domino_quick_test
  log_dir: "logs"
  results_excel_path: "experiment_results.xlsx"
  save_images: true
  save_multi_view: true
  save_trajectory: true
  history_length: 5

agent:
  type: "openai"
  model_name: "gpt-4o"
  # API key / base_url 从环境变量加载（.env 或系统环境）
  api_key: null
  base_url: null
  temperature: 0.7
  max_tokens: 500
  timeout: 300.0

judgement:
  type: "openai"
  model_name: "gpt-5-mini"
  api_key: null
  base_url: null
  temperature: 0.1
  max_tokens: 500
  timeout: 300.0

environment:
  type: "domino"
  gui: false
  urdf_local_path: "src/phyvpuzzle/environment/phobos_models"
  render_width: 512
  render_height: 512
  multi_view: true
  load_table: false
  max_steps: 3

task:
  type: "domino_dont_fall"
  name: "domino_dont_fall_demo"
  difficulty: easy
  num_dominoes: 3
  arrangement_pattern: "line"
  domino_spacing: 0.08
  ruled_evaluation: false
```

关键字段说明：

- **runner**：实验名、日志目录、是否保存图像/多视角/轨迹等。
- **agent**：主体模型（如 `gpt-4o`），温度、`max_tokens`、超时等。
- **judgement**：用于 LLM-as-Judge 的判别模型与参数。
- **environment**：环境类型（`domino`）、是否 GUI、渲染大小、最大步数等。
- **task**：任务类型（`domino_dont_fall`）、难度、骨牌数量、排布方式等。

---

## 运行示例

### 方式一：运行脚本（examples）

```bash
# 运行 Domino 快速示例（会自动读取 eval_configs/domino_quick.yaml）
python examples/domino_quick.py
```

该脚本做了以下工作：

- 加载 `.env`（若安装了 python-dotenv），检查 `OPENAI_API_KEY`；
- 读取 `eval_configs/domino_quick.yaml`；
- 初始化 `BenchmarkRunner` 并直接运行 `run_benchmark()`，最终在 `logs/` 下产生日志、图像、报告。

### 方式二：使用 CLI

```bash
# 单次运行（Run）
python -m phyvpuzzle.cli run --config eval_configs/domino_quick.yaml

# 基准评测（Benchmark）
python -m phyvpuzzle.cli benchmark --config eval_configs/domino_quick.yaml --num-runs 5

# 校验配置
python -m phyvpuzzle.cli validate-config eval_configs/domino_quick.yaml

# 列出可用组件（任务/环境/Agent）
python -m phyvpuzzle.cli list-components

# 查看组件详情（例如 domino_dont_fall 任务）
python -m phyvpuzzle.cli show-component --type task --name domino_dont_fall
```

可选覆盖项（部分）：

- `--gui`：启用物理仿真 GUI；
- `--model`：覆盖 `agent.model_name`；
- `--difficulty`：覆盖任务难度（`easy|medium|hard`）。

---

## 运行基准（Benchmark）

### 用代码调用

参考 `examples/domino_quick.py`：

```python
from pathlib import Path
from phyvpuzzle import load_config, BenchmarkRunner, validate_config

config_path = Path("eval_configs/domino_quick.yaml")
config = load_config(str(config_path))
assert not validate_config(config)  # 返回空列表表示通过

runner = BenchmarkRunner(config)
evaluation = runner.run_benchmark(num_runs=5)
```

### 用 CLI 调用

```bash
PYTHONPATH=src python -m phyvpuzzle.cli benchmark --config eval_configs/domino_quick.yaml --num-runs 5
```

Benchmark 流程（核心调用链）：

- `BenchmarkRunner.run_benchmark()`
  - 多次 `run_single_task()`：循环交互（Agent 调用工具 → 环境 `step` → 渲染与记录）
  - `evaluate()`：调用任务的成功判定逻辑 + 评估器聚合指标
  - 导出 Excel 与 JSON 报告

---

## 扩展：如何添加新环境（Environment）

新环境需基于 `PhysicsEnvironment`（PyBullet 封装）。以 Domino 为例：`src/phyvpuzzle/environment/domino_env.py`。

必须实现/可扩展的方法（摘自 `BaseEnvironment` 与 `PhysicsEnvironment`）：

- `reset() -> Observation`：环境重置，返回观测。
- `step(action: Action) -> Observation`：执行工具动作，返回最新观测。
- `render(multi_view: bool) -> PIL.Image | Dict[str, PIL.Image]`：渲染当前视图或多视图。
- `get_tool_schemas() -> List[Dict[str, Any]]`：返回 Agent 可调用的工具 JSON-Schema 列表。
- `execute_tool_call(action: Action) -> Dict[str, Any]`：执行工具调用，返回状态字典。
- `close() -> None`：释放资源。

如果继承 `PhysicsEnvironment`，还需（或可）实现：

- `_get_current_state(metadata) -> State`：打包 `objects`、时间戳和 `metadata`；
- `_get_state_description() -> str`：将最近一次 `tool_call` 与 `tool_result`转为可读文本，供提示词历史；
- `_get_task_specific_tool_schemas() -> List[Dict[str, Any]]`：声明该环境特有工具；
- 可使用装饰器 `@BaseEnvironment.register_tool("tool_name")` 定义工具实现函数，函数签名示例：
  - `def _tool_pick(self, object_id: int) -> Dict[str, Any]`
  - `def _tool_place(self, object_id: int, position: List[float]) -> Dict[str, Any]`
  - `def _tool_push(self, object_id: int, force: float, direction: List[float]) -> Dict[str, Any]`

Domino 环境提供的特有工具（示例）：

- `push_specific_domino(domino_id: str, force: float = 5.0, direction: List[float] = [1,0,0]) -> {status, message}`
- `reset_dominoes() -> {status, message}`

状态与观测的数据结构（见 `phyvpuzzle.core.base`）：

- `State(step, objects: List[ObjectInfo], time_stamp, metadata)`
- `Observation(image: PIL.Image, state: State, description: str)`

注册新环境：

- 使用 `@register_environment_config("env_name")` 为配置注册 dataclass；
- 使用 `@register_environment("env_name")` 注册环境类；
- 在 YAML 中 `environment.type: env_name` 即可启用。

返回值规范：

- 工具实现统一返回 `Dict[str, Any]`，至少包含 `status` 和 `message` 字段；
- `step()` 会将 `tool_call` 与 `tool_result` 写入 `State.metadata`，并用于日志与评估。

---

## 扩展：如何添加新任务（Task）

新任务基于 `BaseTask` 或 `PhysicsTask`。Domino 任务示例：`src/phyvpuzzle/tasks/domino_dont_fall_task.py`。

`PhysicsTask` 需/可实现的方法：

- `_calculate_optimal_steps() -> int`：用于计算最优步数，供效率指标使用；
- `_configure_environment() -> None`：加载物体、布局、设置初态；
- `_evaluate_success(task_results: List[TaskResult]) -> List[TaskResult]`：逐个任务填充 `success` 字段（可规则判断或 LLM 评审）；
- `_get_initial_system_prompt() -> str`：系统提示词；
- `_get_initial_instruction() -> str`：用户任务说明。

`BaseTask` 统一入口（已在基类实现）：

- `configure_environment(environment) -> Observation`：调用 `_configure_environment()` 并返回初始观测；
- `evaluate_tasks(evaluator, task_results) -> EvaluationResult`：先调用本任务的 `_evaluate_success()` 再交给评估器聚合；
- `get_system_prompt() / get_user_prompt()`：分别返回系统与用户提示词。

Domino 任务中的成功判定：

- 规则版（`ruled_evaluation: true`）：根据最终 `Observation.state.objects` 的姿态（欧拉角阈值）判断倒下的骨牌比例是否 ≥ 80%；
- LLM 评审版（`ruled_evaluation: false`）：构造标准描述，调用评审器 `LLMJudge` 判定是否成功，并记录 `judge_success/judge_confidence/judge_reasoning` 至 `TaskResult.metadata`。

注册新任务：

- 使用 `@register_task_config("task_name")` 注册配置 dataclass；
- 使用 `@register_task("task_name")` 注册任务类；
- 在 YAML 中 `task.type: task_name` 即可启用。

返回值规范：

- `_evaluate_success` 必须为每个 `TaskResult` 填充 `success: bool`，并可写入 `metadata`（如 `judge_metrics`）。

---

## 指标与结果产物

评估器：`src/phyvpuzzle/evaluation/evaluator.py`

- `BenchmarkEvaluator.evaluate_metrics(task_results)` 调用 `MetricsCalculator.calculate_comprehensive_metrics` 产出 `EvaluationResult`：
  - `accuracy`: 成功率；
  - `pass_at_k`: 分组后的 Pass@K；
  - `distance_to_optimal`: 成功样本相对最优步数的平均超额；
  - `token_efficiency`: 每次成功的平均 Token；
  - `detailed_metrics`：
    - `step_efficiency`、`time_efficiency`；
    - `success_by_difficulty`、`success_by_task_type`；
    - `trajectory_analysis`（常见首动作、失败步等）。

结果导出：

- Excel（两份）：
  - 详单：`logs/{experiment_name}/experiment_results_Detailed.xlsx`
  - 难度统计：`logs/{experiment_name}/experiment_results_Difficulty.xlsx`
- JSON 报告：
  - 汇总：`logs/{experiment_name}/detailed_reports/{model}_evaluation_report.json`
  - 轨迹：`logs/{experiment_name}/detailed_reports/{model}_trajectories.json`
- 日志与图片：
  - `logs/{experiment_name}/experiment_log.json`
  - `logs/{experiment_name}/images/step_*.png`

控制台展示：

- Run：显示单次任务的执行时间、步数、是否成功；
- Benchmark：汇总 Accuracy、各 `Pass@k` 等关键指标与输出路径。

---

## 常见问题

- 没有 `OPENAI_API_KEY`：请在 `.env` 或环境变量中设置；
- PyBullet 报错：`pip install pybullet` 并确保渲染/驱动可用；
- 结果总失败：检查任务 `_evaluate_success()` 是否与任务目标一致。

---

MIT License — Maojia Song (SUTD)
