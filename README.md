# PhyVPuzzle: Physics-based Visual Reasoning Benchmark

> A comprehensive benchmark for evaluating physical visual reasoning capabilities of Vision-Language Models (VLMs) through interactive 3D physics simulation.

## 📋 Table of Contents

- [Quick Start](#-quick-start)
- [Task Types](#-supported-task-types)
- [CLI Usage](#-cli-usage-guide)
- [Evaluation](#-detailed-evaluation-metrics)
- [Development](#-development-guide)

PhyVPuzzle evaluates Vision-Language Models on physics-based reasoning tasks through 3D simulation, testing perception, reasoning, planning, execution, and visual judgment capabilities.

## 🚀 Quick Start

```bash
# Install
git clone https://github.com/your-org/VisualReasonBench.git
cd VisualReasonBench
conda install -c conda-forge pybullet
pip install -e ".[dev]"

# Run example
python examples/domino_quick.py

# Run benchmark
phyvpuzzle benchmark --config eval_configs/domino_quick.yaml --num-runs 5
```

## 📊 Supported Task Types

| Task | Goal | Difficulty | Evaluation |
|------|------|------------|------------|
| 🎲 **Domino** | Trigger chain reaction | 3-21 pieces | >80% fallen |
| 🔒 **Luban Lock** | Assemble/disassemble puzzles | 3-15 pieces | Correct interlocking |
| 🏗️ **Pagoda** | Build stable tower | Variable height | Stability & symmetry |

## ⚙️ Configuration

Key configuration parameters in `eval_configs/domino_quick.yaml`:
- **Agent**: Model (gpt-4o), temperature (0.7), max_tokens (500)
- **Environment**: PyBullet simulation with multi-view rendering
- **Task**: Type (domino), difficulty (very_easy), parameters (3 dominoes, line pattern)

## 🛠️ CLI Usage Guide

```bash
# Basic usage
phyvpuzzle run --config config.yaml
phyvpuzzle evaluate --results-dir logs/
phyvpuzzle benchmark --config config.yaml --num-runs 5

# With GUI and parameter overrides
phyvpuzzle run --config config.yaml --gui --model gpt-4 --difficulty medium
```

## 📁 Structure

```
src/phyvpuzzle/     # Core framework: agents, environments, tasks, evaluation
eval_configs/       # YAML configuration files  
examples/           # Usage examples
scripts/            # Benchmark and evaluation scripts
logs/               # Results and experiment data
```

## 🔧 Development Guide

Extend the framework by inheriting from base classes:
- **Tasks**: `PuzzleTask` → define prompts and validation
- **Environments**: `PhysicsEnvironment` → setup 3D scenes  
- **Agents**: `VLMAgent` → implement model interfaces
- **Metrics**: `MetricsCalculator` → add evaluation functions

## 📊 Evaluation Metrics

**Core Metrics:**
- **Accuracy**: Success rate
- **Pass@K**: Success probability in K attempts  
- **Efficiency**: Steps/tokens per success
- **Optimality**: Distance from optimal solution

**Outputs:**
- Excel reports with detailed results (`logs/experiment_results.xlsx`)
- JSON trajectory data (`logs/detailed_reports/`)
- Multi-view screenshots (`logs/{experiment_name}/images/`)

## 📄 License & Contact

MIT License - **Maojia Song** (SUTD) - maojia_song@mymail.sutd.edu.sg

---

*PhyVPuzzle: Let AI think and act in the physical world* 🤖🧩
