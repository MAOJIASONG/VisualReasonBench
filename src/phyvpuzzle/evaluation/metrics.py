"""
Evaluation Metrics Module

This module provides evaluation metrics for physical visual reasoning tasks,
including Accuracy, Pass@8, Distance to optimal steps, and other metrics.
"""

from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
from dataclasses import dataclass
from enum import Enum
import math

from ..tasks.base_task import BaseTask, TaskResult
from ..core.pipeline import PhysicalReasoningPipeline


class MetricType(Enum):
    """Types of evaluation metrics."""
    ACCURACY = "accuracy"
    PASS_AT_K = "pass_at_k"
    DISTANCE_TO_OPTIMAL = "distance_to_optimal"
    SUCCESS_RATE = "success_rate"
    EFFICIENCY = "efficiency"
    TIME_EFFICIENCY = "time_efficiency"
    STEP_EFFICIENCY = "step_efficiency"
    ROBUSTNESS = "robustness"


@dataclass
class EvaluationResult:
    """Result of evaluation on a dataset."""
    total_tasks: int
    successful_tasks: int
    metrics: Dict[str, float]
    task_results: List[TaskResult]
    per_task_metrics: List[Dict[str, float]]
    metadata: Dict[str, Any]


class AccuracyMetric:
    """Accuracy metric evaluator."""
    
    def __init__(self):
        self.name = "accuracy"
    
    def evaluate(self, results: List[TaskResult]) -> float:
        """
        Calculate accuracy as the percentage of successfully completed tasks.
        
        Args:
            results: List of task results
            
        Returns:
            Accuracy score (0.0 to 1.0)
        """
        if not results:
            return 0.0
        
        successful_tasks = sum(1 for result in results if result.success)
        return successful_tasks / len(results)


class PassAtKMetric:
    """Pass@K metric evaluator."""
    
    def __init__(self, k: int = 8):
        self.k = k
        self.name = f"pass_at_{k}"
    
    def evaluate(self, results_groups: List[List[TaskResult]]) -> float:
        """
        Calculate Pass@K metric.
        
        Args:
            results_groups: List of result groups, each group contains K attempts
                          at the same task
            
        Returns:
            Pass@K score (0.0 to 1.0)
        """
        if not results_groups:
            return 0.0
        
        successful_groups = 0
        for group in results_groups:
            if any(result.success for result in group):
                successful_groups += 1
        
        return successful_groups / len(results_groups)
    
    def evaluate_from_single_results(self, results: List[TaskResult]) -> float:
        """
        Calculate Pass@K from single task results (assuming independence).
        
        Args:
            results: List of task results
            
        Returns:
            Pass@K score (0.0 to 1.0)
        """
        if not results:
            return 0.0
        
        # Group results by task (assuming task names are available)
        task_groups = {}
        for result in results:
            task_id = getattr(result, 'task_id', 'default')
            if task_id not in task_groups:
                task_groups[task_id] = []
            task_groups[task_id].append(result)
        
        # Calculate pass@k for each task group
        successful_tasks = 0
        total_tasks = len(task_groups)
        
        for task_id, task_results in task_groups.items():
            if any(result.success for result in task_results[:self.k]):
                successful_tasks += 1
        
        return successful_tasks / total_tasks if total_tasks > 0 else 0.0


class DistanceToOptimalMetric:
    """Distance to optimal solution metric."""
    
    def __init__(self):
        self.name = "distance_to_optimal"
    
    def evaluate(self, results: List[TaskResult], 
                optimal_solutions: List[List[str]]) -> float:
        """
        Calculate average distance to optimal solution.
        
        Args:
            results: List of task results
            optimal_solutions: List of optimal solutions for each task
            
        Returns:
            Average distance to optimal (lower is better)
        """
        if not results or not optimal_solutions:
            return float('inf')
        
        if len(results) != len(optimal_solutions):
            raise ValueError("Number of results must match number of optimal solutions")
        
        total_distance = 0.0
        valid_tasks = 0
        
        for result, optimal_solution in zip(results, optimal_solutions):
            if hasattr(result, 'action_sequence'):
                distance = self._calculate_edit_distance(
                    result.action_sequence, optimal_solution
                )
                total_distance += distance
                valid_tasks += 1
            else:
                # If no action sequence, use step count as proxy
                distance = abs(result.steps_taken - len(optimal_solution))
                total_distance += distance
                valid_tasks += 1
        
        return total_distance / valid_tasks if valid_tasks > 0 else float('inf')
    
    def _calculate_edit_distance(self, sequence1: List[str], 
                               sequence2: List[str]) -> float:
        """Calculate edit distance between two sequences."""
        len1, len2 = len(sequence1), len(sequence2)
        
        # Create DP table
        dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]
        
        # Initialize base cases
        for i in range(len1 + 1):
            dp[i][0] = i
        for j in range(len2 + 1):
            dp[0][j] = j
        
        # Fill DP table
        for i in range(1, len1 + 1):
            for j in range(1, len2 + 1):
                if sequence1[i-1] == sequence2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
        
        return float(dp[len1][len2])


class EfficiencyMetric:
    """Efficiency metric evaluator."""
    
    def __init__(self, metric_type: str = "step"):
        self.metric_type = metric_type
        self.name = f"{metric_type}_efficiency"
    
    def evaluate(self, results: List[TaskResult], 
                optimal_steps: Optional[List[int]] = None) -> float:
        """
        Calculate efficiency metric.
        
        Args:
            results: List of task results
            optimal_steps: List of optimal step counts for each task
            
        Returns:
            Efficiency score (0.0 to 1.0, higher is better)
        """
        if not results:
            return 0.0
        
        if self.metric_type == "step":
            return self._calculate_step_efficiency(results, optimal_steps)
        elif self.metric_type == "time":
            return self._calculate_time_efficiency(results)
        else:
            raise ValueError(f"Unknown efficiency type: {self.metric_type}")
    
    def _calculate_step_efficiency(self, results: List[TaskResult], 
                                 optimal_steps: Optional[List[int]]) -> float:
        """Calculate step efficiency."""
        if optimal_steps is None:
            # Use minimum steps among successful tasks as reference
            successful_results = [r for r in results if r.success]
            if not successful_results:
                return 0.0
            min_steps = min(r.steps_taken for r in successful_results)
            optimal_steps = [min_steps] * len(results)
        
        total_efficiency = 0.0
        valid_tasks = 0
        
        for result, optimal_step in zip(results, optimal_steps):
            if result.success and optimal_step > 0:
                efficiency = min(1.0, optimal_step / result.steps_taken)
                total_efficiency += efficiency
                valid_tasks += 1
        
        return total_efficiency / valid_tasks if valid_tasks > 0 else 0.0
    
    def _calculate_time_efficiency(self, results: List[TaskResult]) -> float:
        """Calculate time efficiency."""
        successful_results = [r for r in results if r.success]
        if not successful_results:
            return 0.0
        
        min_time = min(r.time_taken for r in successful_results)
        if min_time <= 0:
            return 0.0
        
        total_efficiency = 0.0
        for result in successful_results:
            efficiency = min(1.0, min_time / result.time_taken)
            total_efficiency += efficiency
        
        return total_efficiency / len(successful_results)


class RobustnessMetric:
    """Robustness metric evaluator."""
    
    def __init__(self):
        self.name = "robustness"
    
    def evaluate(self, results_by_difficulty: Dict[str, List[TaskResult]]) -> float:
        """
        Calculate robustness across different difficulty levels.
        
        Args:
            results_by_difficulty: Dictionary mapping difficulty level to results
            
        Returns:
            Robustness score (0.0 to 1.0)
        """
        if not results_by_difficulty:
            return 0.0
        
        difficulty_weights = {
            "easy": 0.2,
            "medium": 0.3,
            "hard": 0.3,
            "expert": 0.2
        }
        
        weighted_score = 0.0
        total_weight = 0.0
        
        for difficulty, results in results_by_difficulty.items():
            if not results:
                continue
                
            success_rate = sum(1 for r in results if r.success) / len(results)
            weight = difficulty_weights.get(difficulty, 0.25)
            
            weighted_score += success_rate * weight
            total_weight += weight
        
        return weighted_score / total_weight if total_weight > 0 else 0.0


class ComprehensiveEvaluator:
    """Comprehensive evaluator for physical visual reasoning tasks."""
    
    def __init__(self):
        self.metrics = {
            MetricType.ACCURACY: AccuracyMetric(),
            MetricType.PASS_AT_K: PassAtKMetric(k=8),
            MetricType.DISTANCE_TO_OPTIMAL: DistanceToOptimalMetric(),
            MetricType.STEP_EFFICIENCY: EfficiencyMetric("step"),
            MetricType.TIME_EFFICIENCY: EfficiencyMetric("time"),
            MetricType.ROBUSTNESS: RobustnessMetric()
        }
    
    def evaluate_pipeline(self, pipeline: PhysicalReasoningPipeline,
                         tasks: List[BaseTask],
                         optimal_solutions: Optional[List[List[str]]] = None,
                         num_runs: int = 1) -> EvaluationResult:
        """
        Comprehensive evaluation of pipeline on tasks.
        
        Args:
            pipeline: Pipeline to evaluate
            tasks: List of tasks to evaluate on
            optimal_solutions: List of optimal solutions for each task
            num_runs: Number of runs per task (for Pass@K metric)
            
        Returns:
            EvaluationResult with comprehensive metrics
        """
        all_results = []
        per_task_metrics = []
        
        # Run evaluation
        for task in tasks:
            task_results = []
            for run in range(num_runs):
                result = pipeline.execute_task(task)
                task_results.append(result)
                pipeline.reset_pipeline()
            
            all_results.extend(task_results)
            
            # Calculate per-task metrics
            task_metrics = self._calculate_per_task_metrics(task_results)
            per_task_metrics.append(task_metrics)
        
        # Calculate overall metrics
        overall_metrics = self._calculate_overall_metrics(
            all_results, tasks, optimal_solutions, num_runs
        )
        
        # Group results by difficulty for robustness
        results_by_difficulty = self._group_by_difficulty(all_results, tasks)
        if results_by_difficulty:
            overall_metrics[MetricType.ROBUSTNESS.value] = \
                self.metrics[MetricType.ROBUSTNESS].evaluate(results_by_difficulty)
        
        return EvaluationResult(
            total_tasks=len(tasks),
            successful_tasks=sum(1 for results in per_task_metrics 
                               if results.get("success", False)),
            metrics=overall_metrics,
            task_results=all_results,
            per_task_metrics=per_task_metrics,
            metadata={
                "num_runs": num_runs,
                "evaluation_time": sum(r.time_taken for r in all_results)
            }
        )
    
    def _calculate_per_task_metrics(self, task_results: List[TaskResult]) -> Dict[str, float]:
        """Calculate metrics for a single task."""
        metrics = {}
        
        # Success rate for this task
        metrics["success"] = any(r.success for r in task_results)
        metrics["success_rate"] = sum(1 for r in task_results if r.success) / len(task_results)
        
        # Average metrics for successful runs
        successful_results = [r for r in task_results if r.success]
        if successful_results:
            metrics["avg_steps"] = sum(r.steps_taken for r in successful_results) / len(successful_results)
            metrics["avg_time"] = sum(r.time_taken for r in successful_results) / len(successful_results)
            metrics["avg_score"] = sum(r.final_score for r in successful_results) / len(successful_results)
        else:
            metrics["avg_steps"] = float('inf')
            metrics["avg_time"] = float('inf')
            metrics["avg_score"] = 0.0
        
        return metrics
    
    def _calculate_overall_metrics(self, results: List[TaskResult], 
                                 tasks: List[BaseTask],
                                 optimal_solutions: Optional[List[List[str]]],
                                 num_runs: int) -> Dict[str, float]:
        """Calculate overall metrics."""
        metrics = {}
        
        # Accuracy
        metrics[MetricType.ACCURACY.value] = \
            self.metrics[MetricType.ACCURACY].evaluate(results)
        
        # Pass@K (using num_runs as K)
        if num_runs > 1:
            # Group results by task
            task_groups = []
            for i in range(0, len(results), num_runs):
                task_groups.append(results[i:i+num_runs])
            
            pass_at_k_metric = PassAtKMetric(k=num_runs)
            metrics[f"pass_at_{num_runs}"] = pass_at_k_metric.evaluate(task_groups)
        
        # Distance to optimal
        if optimal_solutions:
            metrics[MetricType.DISTANCE_TO_OPTIMAL.value] = \
                self.metrics[MetricType.DISTANCE_TO_OPTIMAL].evaluate(results, optimal_solutions)
        
        # Efficiency metrics
        optimal_steps = None
        if optimal_solutions:
            optimal_steps = [len(sol) for sol in optimal_solutions] * num_runs
        
        metrics[MetricType.STEP_EFFICIENCY.value] = \
            self.metrics[MetricType.STEP_EFFICIENCY].evaluate(results, optimal_steps)
        
        metrics[MetricType.TIME_EFFICIENCY.value] = \
            self.metrics[MetricType.TIME_EFFICIENCY].evaluate(results)
        
        return metrics
    
    def _group_by_difficulty(self, results: List[TaskResult], 
                           tasks: List[BaseTask]) -> Dict[str, List[TaskResult]]:
        """Group results by task difficulty."""
        if len(results) != len(tasks):
            # If multiple runs, each task appears multiple times
            num_runs = len(results) // len(tasks)
            expanded_tasks = []
            for task in tasks:
                expanded_tasks.extend([task] * num_runs)
            tasks = expanded_tasks
        
        difficulty_groups = {}
        for result, task in zip(results, tasks):
            difficulty = task.config.difficulty.value
            if difficulty not in difficulty_groups:
                difficulty_groups[difficulty] = []
            difficulty_groups[difficulty].append(result)
        
        return difficulty_groups
    
    def generate_report(self, evaluation_result: EvaluationResult) -> str:
        """Generate a comprehensive evaluation report."""
        report = []
        report.append("=" * 50)
        report.append("PHYSICAL VISUAL REASONING EVALUATION REPORT")
        report.append("=" * 50)
        
        # Summary
        report.append(f"\nSUMMARY:")
        report.append(f"Total Tasks: {evaluation_result.total_tasks}")
        report.append(f"Successful Tasks: {evaluation_result.successful_tasks}")
        report.append(f"Overall Success Rate: {evaluation_result.metrics.get('accuracy', 0.0):.3f}")
        
        # Metrics
        report.append(f"\nMETRICS:")
        for metric_name, value in evaluation_result.metrics.items():
            if isinstance(value, float):
                if value == float('inf'):
                    report.append(f"{metric_name}: INF")
                else:
                    report.append(f"{metric_name}: {value:.3f}")
            else:
                report.append(f"{metric_name}: {value}")
        
        # Per-task summary
        report.append(f"\nPER-TASK PERFORMANCE:")
        for i, task_metrics in enumerate(evaluation_result.per_task_metrics):
            report.append(f"Task {i+1}: Success={task_metrics.get('success', False)}, "
                         f"Steps={task_metrics.get('avg_steps', 0):.1f}, "
                         f"Time={task_metrics.get('avg_time', 0):.2f}s")
        
        # Metadata
        report.append(f"\nMETADATA:")
        for key, value in evaluation_result.metadata.items():
            report.append(f"{key}: {value}")
        
        return "\n".join(report)


def evaluate_model_performance(pipeline: PhysicalReasoningPipeline,
                             tasks: List[BaseTask],
                             optimal_solutions: Optional[List[List[str]]] = None,
                             num_runs: int = 8) -> EvaluationResult:
    """
    Convenience function to evaluate model performance.
    
    Args:
        pipeline: Pipeline to evaluate
        tasks: List of tasks to evaluate on
        optimal_solutions: List of optimal solutions for each task
        num_runs: Number of runs per task (for Pass@K metric)
        
    Returns:
        EvaluationResult with comprehensive metrics
    """
    evaluator = ComprehensiveEvaluator()
    return evaluator.evaluate_pipeline(pipeline, tasks, optimal_solutions, num_runs) 