"""
Evaluation system for PhyVPuzzle benchmark.

This package contains evaluation components including:
- Metrics calculation (accuracy, pass@k, etc.)
- LLM-as-judge evaluation
- Result aggregation and analysis
"""

from phyvpuzzle.evaluation.metrics import MetricsCalculator
from phyvpuzzle.evaluation.evaluator import BenchmarkEvaluator
from phyvpuzzle.evaluation.judge import LLMJudge

__all__ = [
    "MetricsCalculator",
    "BenchmarkEvaluator", 
    "LLMJudge"
]
