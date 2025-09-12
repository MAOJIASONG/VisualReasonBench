"""
PhyVPuzzle: A Physics-based Visual Reasoning Benchmark

This package provides a comprehensive benchmark for evaluating Vision-Language Models
on complex physics puzzle tasks that require visual understanding, spatial reasoning,
and sequential action planning.

Key Components:
- Physical simulation environments using PyBullet
- Various puzzle tasks (domino, Luban lock, pagoda, etc.)
- VLM agent implementations (OpenAI, VLLM, etc.)
- Comprehensive evaluation metrics
- Command-line interface for easy execution

Example Usage:
```python
from phyvpuzzle.core.config import load_config
from phyvpuzzle.runner import BenchmarkRunner

# Load configuration and run benchmark
config = load_config("config.yaml")
runner = BenchmarkRunner(config)
runner.setup()
runner.run_benchmark(num_runs=5)
```

Command-line Usage:
```bash
# Run single task
phyvpuzzle run --config domino_config.yaml

# Run full benchmark
phyvpuzzle benchmark --config config.yaml --num-runs 5

# Evaluate results
phyvpuzzle evaluate --results-dir logs/
```
"""

# Normal imports instead of lazy loading to ensure proper registry initialization
from phyvpuzzle.core.config import Config, load_config, validate_config
from phyvpuzzle.runner import BenchmarkRunner

__version__ = "0.1.0"
__author__ = "Maojia Song"
__email__ = "maojia_song@mymail.sutd.edu.sg"

__all__ = [
    "Config",
    "load_config",
    "validate_config",
    "BenchmarkRunner"
]
