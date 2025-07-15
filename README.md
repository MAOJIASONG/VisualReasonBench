# PhyVPuzzle: Physical Visual Reasoning Benchmark

A comprehensive benchmark for evaluating Vision-Language Models (VLMs) on physical visual reasoning tasks using 3D interactive physics simulation.

## 🎯 Overview

PhyVPuzzle evaluates VLMs' ability to perform **pure visual-based sequential reasoning** on multi-turn interactive physical tasks without relying on text descriptions. The benchmark includes three main task categories:

- **Puzzle Tasks**: Kongming Lock, and other logic puzzles
- **Lego Tasks**: Block construction and assembly challenges  
- **Dominoes Tasks**: Sequential interaction and chain reaction planning

## 🏗️ Architecture

The system follows a pipeline architecture that mirrors the reasoning process:

```
Task → VLLM → Decision (finish/action) → Action Description → Translator → Environment
  ↑                                                                            ↓
  ← ← ← ← ← ← ← ← ← 3D Feedback + History ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ←
```

### Core Components

1. **VLLM Processor**: Handles vision-language model inference (OpenAI, HuggingFace)
2. **Action Descriptor**: Parses natural language actions into structured commands
3. **Translator**: Converts actions to environment-executable commands (rule-based or LLM)
4. **Environment**: PyBullet-based 3D physics simulation with visual feedback
5. **Evaluator**: Comprehensive metrics including Accuracy, Pass@8, Distance to optimal

## 🔧 Installation

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (optional, for local VLMs)
- OpenAI API key (for OpenAI models)

### Install from Source

```bash
git clone https://github.com/MAOJIASONG/VisualReasonBench.git
cd VisualReasonBench

# Create virtual environment
conda create -n vpr python=3.10 
conda activate vpr

# Install dependencies
pip install -e .

# Install development dependencies (optional)
pip install -e ".[dev]"
```

## 🚀 Quick Start

### Command Line Interface

```bash
# Run evaluation on puzzle tasks
phyvpuzzle evaluate --task-type puzzle --difficulty medium --num-runs 8

# Run single task with GUI if needed
phyvpuzzle run --task-type lego --difficulty easy --gui

# Generate sample tasks
phyvpuzzle generate --task-type dominoes --count 10 --output tasks.json
```

### Python API

```python
import phyvpuzzle

# Quick evaluation
result = phyvpuzzle.quick_evaluate(
    task_type="puzzle",
    difficulty="medium", 
    num_tasks=5,
    num_runs=3
)

print(f"Success Rate: {result.metrics['accuracy']:.3f}")
print(f"Pass@8: {result.metrics.get('pass_at_8', 0):.3f}")
```

### Custom Pipeline

```python
from phyvpuzzle import (
    PipelineConfig, 
    create_pipeline, 
    create_task_config,
    evaluate_model_performance
)

# Configure pipeline
config = PipelineConfig(
    vllm_type="openai",
    vllm_model="gpt-4-vision-preview",
    translator_type="rule_based",
    max_iterations=100,
    timeout=300.0
)

# Create and run pipeline
with create_pipeline(config) as pipeline:
    # Create tasks
    tasks = []
    for i in range(10):
        task_config = create_task_config("puzzle", "medium")
        # Add actual task implementation here
        
    # Evaluate
    results = evaluate_model_performance(pipeline, tasks, num_runs=8)
    print(f"Overall accuracy: {results.metrics['accuracy']:.3f}")
```

## 📊 Evaluation Metrics

### Core Metrics

- **Accuracy**: Percentage of successfully completed tasks
- **Pass@K**: Success rate when allowing K attempts per task
- **Distance to Optimal**: Edit distance from optimal solution sequence
- **Step Efficiency**: Ratio of optimal steps to actual steps taken
- **Time Efficiency**: Ratio of minimum time to actual time taken
- **Robustness**: Performance consistency across difficulty levels

### Example Output

```
==================================================
PHYSICAL VISUAL REASONING EVALUATION REPORT
==================================================

SUMMARY:
Total Tasks: 50
Successful Tasks: 32
Overall Success Rate: 0.640

METRICS:
accuracy: 0.640
pass_at_8: 0.780
distance_to_optimal: 3.240
step_efficiency: 0.521
time_efficiency: 0.445
robustness: 0.588

PER-TASK PERFORMANCE:
Task 1: Success=True, Steps=12.0, Time=45.23s
Task 2: Success=False, Steps=inf, Time=inf
...
```

## 🎮 Task Types

### Puzzle Tasks

- **Kongming Lock**: 3D interlocking puzzle requiring spatial reasoning
- **Block Sorting**: Arrange colored blocks according to patterns

### Lego Tasks

- **Free Building**: Construct specific shapes from building blocks
- **Instruction Following**: Build according to visual instructions
- **Creative Assembly**: Design and build functional structures

### Dominoes Tasks

- **Chain Setup**: Arrange dominoes for continuous chain reaction
- **Pattern Creation**: Create specific patterns with falling dominoes
- **Obstacle Navigation**: Plan paths around obstacles

## 🔧 Configuration

### Environment Variables

```bash
export OPENAI_API_KEY="your-api-key-here"
export HUGGINGFACE_TOKEN="your-token-here"  # Optional for HF models
```

### Configuration File

```json
{
  "vllm_type": "openai",
  "vllm_model": "gpt-4-vision-preview",
  "translator_type": "rule_based",
  "environment_type": "pybullet",
  "max_iterations": 100,
  "timeout": 300.0,
  "enable_logging": true,
  "log_level": "INFO"
}
```

## 🧪 Development

### Running Tests

```bash
pytest tests/
```

### Code Quality

```bash
# Format code
black src/

# Type checking
mypy src/

# Linting
flake8 src/
```

### Adding New Tasks

1. Create a new task class inheriting from `BaseTask`
2. Implement required abstract methods
3. Add task registration in the task factory
4. Write tests for the new task

```python
from phyvpuzzle.tasks.base_task import BaseTask

class CustomTask(BaseTask):
    def setup_task(self, environment):
        # Initialize task in environment
        pass
    
    def get_task_description(self):
        return "Your custom task description"
    
    def check_completion(self):
        # Check if task is completed
        return False
    
    def evaluate_state(self):
        # Return score 0.0-1.0
        return 0.0
```

## 📈 Benchmarking Results

| Model | Accuracy | Pass@8 | Distance | Step Eff | Time Eff |
|-------|----------|---------|----------|----------|----------|
| GPT-4V | 0 | 0 | 0 | 0 | 0 |
| Claude-3 | 0 | 0 | 0 | 0 | 0 |
| Gemini-Pro | 0 | 0 | 0 | 0 | 0 |

## 📄 Citation

If you use PhyVPuzzle in your research, please cite:

```bibtex
@misc{phyvpuzzle2025,
  title={PhyVPuzzle: A Physical Visual Reasoning Benchmark for Vision-Language Models},
  author={MAOJIASONG},
  year={2025},
  url={https://github.com/MAOJIASONG/VisualReasonBench}
}
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure code passes all tests and linting
5. Submit a pull request


## 🙏 Acknowledgments

- PyBullet for physics simulation
- OpenAI for vision-language models
- HuggingFace for model hosting and tools
- The research community for feedback and contributions


## 🔄 Changelog (FOR DEV ONLY)

### Version 0.1.0
- Initial release
- Core pipeline implementation
- Basic task framework
- Evaluation metrics
- CLI interface
- PyBullet environment integration
