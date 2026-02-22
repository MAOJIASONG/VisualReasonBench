# CHAIN: Causal Hierarchy of Actions and Interactions

> An interactive 3D physics-driven benchmark for evaluating whether Vision-Language Models can understand, plan, and execute structured action sequences grounded in physical constraints.

## Overview

Understanding physical structure is essential for real-world applications such as embodied agents, interactive design, and long-horizon manipulation. Yet, prevailing VLM evaluations still center on structure-agnostic, single-turn setups (e.g., VQA), which fail to assess agents' ability to reason about how geometry, contact, and support relations jointly constrain what actions are possible in a dynamic environment.

**CHAIN** shifts evaluation from passive perception to **active problem solving**, spanning tasks such as interlocking mechanical puzzles and causal-chain manipulation.

### Supported Tasks

| Task | Environment | Description |
|------|-------------|-------------|
| **Luban Lock Disassembly** | `luban` (Unity) | Disassemble interlocking wooden puzzles by identifying and executing valid removal sequences |
| **Stacking Game** | `stacking_game` | Pack polycube pieces into a target 3D box under geometric constraints |

---

## Leaderboard

### Diagnosing Frontier Models on CHAIN

We use CHAIN's controlled interactive protocol to localize bottlenecks in perception, planning, and execution as physical constraints tighten.

---

#### Constraint Tightness (Difficulty Stratification)

Accuracy (%) by difficulty tier. Stacking–Easy is largely solved, but performance collapses at Mid/Hard. Puzzle–Easy peaks at 10%, while Puzzle–Mid/Hard remain at 0%.

| Model            | Puzzle Acc ↑ (Easy) | Puzzle Acc ↑ (Mid) | Puzzle Acc ↑ (Hard) | Stacking Acc ↑ (Easy) | Stacking Acc ↑ (Mid) | Stacking Acc ↑ (Hard) |
|------------------|---------------------:|-------------------:|--------------------:|----------------------:|---------------------:|----------------------:|
| GPT-5.2          | 10.0                 | 0.0                | 0.0                 | 100.0                 | 55.0                 | 6.3                   |
| Gemini-3-Pro     | 10.0                 | 0.0                | 0.0                 | 90.0                  | 40.0                 | 6.3                   |
| Claude-Sonnet-4.5| 10.0                 | 0.0                | 0.0                 | 100.0                 | 20.0                 | 0.0                   |

---

#### Intermediate Feedback (Interactive vs. One-shot)

Multi-step interaction consistently outperforms one-shot solving. **Δ = Interactive − One-shot** on overall accuracy.

| Model            | Interactive (%) ↑ (Puzzle) | Interactive (%) ↑ (Stack.) | Interactive (%) ↑ (All) | One-shot (%) ↑ (Puzzle) | One-shot (%) ↑ (Stack.) | One-shot (%) ↑ (All) | Δ |
|------------------|----------------------------:|----------------------------:|-------------------------:|-------------------------:|--------------------------:|----------------------:|--:|
| GPT-5.2          | 3.1                         | 31.2                        | 22.9                     | 0.0                      | 9.1                       | 7.1                   | -15.8 |
| Claude-Sonnet-4.5| 3.1                         | 18.2                        | 13.8                     | 0.0                      | 10.3                      | 8.1                   | -5.7  |
| Gemini-3-Pro     | 3.1                         | 26.0                        | 19.3                     | 0.0                      | 9.1                       | 7.1                   | -12.2 |

---

#### Selection Signal (Reward Models vs. Verification)

Better selection helps, but gains saturate quickly. Reward-model reranking provides limited improvements relative to stronger verifier-style checks.

| Strategy      | All (%) ↑ | Δ vs. Avg@4 |
|---------------|----------:|------------:|
| Avg@4         | 9.3       | —           |
| Pass@1        | 9.4       | +0.1        |
| Pass@2        | 11.2      | +1.9        |
| Pass@4        | 11.2      | +1.9        |
| VLM Judge     | 10.3      | +1.3        |
| Reward Model  | 9.9       | +0.6        |

## Quick Start

### Prerequisites & Environment Variables

```bash
# Clone and enter the project
git clone https://github.com/your-org/CHAIN.git
cd CHAIN

# Install dependencies
pip install -e .

# Set API key (pick one method)
export OPENAI_API_KEY="your-openai-api-key"
# Or create a .env file
printf "OPENAI_API_KEY=your-openai-api-key\n" > .env
```

> **Note:** The benchmark uses OpenAI-compatible models by default for both the agent and the judge. You can override these in the YAML config.

---

## Configuration (YAML)

### Luban Lock example — `eval_configs/luban.yaml`

```yaml
runner:
  experiment_name: luban_disassembly_test
  log_dir: "logs"
  history_length: 5

agent:
  type: "openai"
  model_name: "gpt-5.2"
  temperature: 0.6
  max_tokens: 4096
  timeout: 300.0

judgement:
  type: "openai"
  model_name: "qwen/qwen3-vl-30b-a3b-thinking"
  temperature: 0.1
  max_tokens: 2048
  timeout: 300.0

environment:
  type: "luban"
  urdf_local_path: "assets/pybullet/phobos_models"
  gui: false
  render_width: 512
  render_height: 512
  max_steps: 6

task:
  type: "luban_disassembly"
  name: "luban_test"
  difficulty: "easy"
  urdf_root: "assets/pybullet/phobos_models/luban-6-piece"
  ruled_evaluation: true
```

### Stacking Game example — `eval_configs/stacking_game_222.yaml`

Set `environment.type: stacking_game` and `task.type: stacking_game`. Point `puzzle_dir` to `assets/stacking_game/puzzles_full_v9`.

---

## Running

### Via example scripts

```bash
# Luban Lock demo
python examples/luban_example.py

# Stacking Game demo
python examples/stacking_game_test.py --config eval_configs/stacking_game_222.yaml --k 1 --workers 1
```

### Via CLI

```bash
# Single run
python -m chainbench.cli run --config eval_configs/luban.yaml

# Benchmark (multiple runs)
python -m chainbench.cli benchmark --config eval_configs/luban.yaml --num-runs 5

# Validate a config
python -m chainbench.cli validate-config eval_configs/luban.yaml

# List available components
python -m chainbench.cli list-components

# Show component details
python -m chainbench.cli show-component --type task --name luban_disassembly
```

---

## Project Structure

```
CHAIN/
  assets/                       # Model & dataset assets
    pybullet/phobos_models/     #   URDF / OBJ models for Luban Lock
    stacking_game/              #   Puzzle dataset for Stacking Game
  eval_configs/                 # YAML experiment configurations
  example_logs/                 # Archived experiment logs
  examples/                     # Runnable demo scripts
  notebooks/                    # Analysis notebooks
  scripts/                      # Shell helper scripts
  src/chainbench/               # Main Python package
    agents/                     #   Agent implementations (OpenAI, human, etc.)
    core/                       #   Registry, config, base data classes
    environment/                #   Environment implementations (luban, stacking_game)
    evaluation/                 #   Evaluator, metrics, LLM judge
    tasks/                      #   Task implementations (luban_disassembly, stacking_game)
    utils/                      #   Rendering utilities
    cli.py                      #   Command-line interface
    runner.py                   #   Benchmark orchestration
```

---

## Extending CHAIN

### Adding a New Environment

1. Subclass `BaseEnvironment` (in `chainbench.core.base`).
2. Implement: `reset()`, `step()`, `render()`, `get_tool_schemas()`, `execute_tool_call()`, `close()`.
3. Register with `@register_environment("env_name")` and `@register_environment_config("env_name")`.
4. Set `environment.type: env_name` in your YAML.

### Adding a New Task

1. Subclass `BaseTask` or `PhysicsTask` (in `chainbench.tasks.base_task`).
2. Implement: `_configure_environment()`, `_evaluate_success()`, `_get_initial_system_prompt()`, `_get_initial_instruction()`.
3. Register with `@register_task("task_name")` and `@register_task_config("task_name")`.
4. Set `task.type: task_name` in your YAML.

---

## Metrics & Output

The evaluator (`chainbench.evaluation.evaluator`) produces:

- **Accuracy** — success rate
- **Pass@K** — grouped pass-at-k
- **Distance to Optimal** — average excess steps over optimal
- **Token Efficiency** — average tokens per success
- **Detailed metrics** — step/time efficiency, success by difficulty, trajectory analysis

Output files are saved under `logs/{experiment_name}/`:

- Excel reports (detailed + difficulty breakdown)
- JSON reports (evaluation + trajectories)
- Images (`images/step_*.png`)
- Experiment log (`experiment_log.json`)

---

## FAQ

- **No API key?** Set `OPENAI_API_KEY` in `.env` or as an environment variable.
- **Stacking game dataset missing?** Place puzzle JSON files under `assets/stacking_game/puzzles_full_v9/`. A built-in 2x2x2 demo loads automatically as fallback.
- **Luban Unity server not running?** The Luban environment connects to a Unity process via socket. Make sure the Unity server is running before starting the benchmark.

---

MIT License — Maojia Song (SUTD)
