"""
Main evaluator for PhyVPuzzle benchmark system.
"""

from typing import List, Dict, Any, Optional
import json
import os
from datetime import datetime
import pandas as pd

from ..core.base import BaseEvaluator, TaskResult, EvaluationResult
from .metrics import MetricsCalculator
from .judge import LLMJudge


class BenchmarkEvaluator(BaseEvaluator):
    """Main evaluator for PhyVPuzzle benchmarks."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.metrics_calculator = MetricsCalculator()
        self.judge = None
        
        # Initialize LLM judge if configured
        judge_config = config.get("judge")
        if judge_config:
            self.judge = LLMJudge(
                model_name=judge_config.get("model_name", "gpt-4o"),
                config=judge_config
            )
            
    def evaluate_single_task(self, task_result: TaskResult) -> Dict[str, float]:
        """Evaluate a single task result."""
        metrics = {}
        
        # Basic success metric
        metrics["success"] = 1.0 if task_result.success else 0.0
        
        # Step efficiency
        optimal_steps = task_result.metadata.get("optimal_steps", task_result.total_steps)
        if optimal_steps > 0:
            metrics["step_efficiency"] = optimal_steps / task_result.total_steps
        else:
            metrics["step_efficiency"] = 0.0
            
        # Time efficiency
        if task_result.execution_time > 0:
            metrics["time_efficiency"] = 1.0 / task_result.execution_time
        else:
            metrics["time_efficiency"] = 0.0
            
        # Token efficiency
        total_tokens = task_result.metadata.get("total_tokens", 0)
        if total_tokens > 0 and task_result.success:
            metrics["token_efficiency"] = 1.0 / total_tokens
        else:
            metrics["token_efficiency"] = 0.0
            
        # Distance to optimal
        if task_result.success and optimal_steps > 0:
            distance = max(0, task_result.total_steps - optimal_steps) / optimal_steps
            metrics["distance_to_optimal"] = distance
        else:
            metrics["distance_to_optimal"] = float('inf')
            
        # LLM judge evaluation if available
        if self.judge:
            try:
                judge_metrics = self._evaluate_with_judge(task_result)
                metrics.update(judge_metrics)
            except Exception as e:
                print(f"Judge evaluation failed: {e}")
                
        return metrics
        
    def evaluate_multiple_tasks(self, task_results: List[TaskResult]) -> EvaluationResult:
        """Evaluate multiple task results and aggregate metrics."""
        return self.metrics_calculator.calculate_comprehensive_metrics(task_results)
        
    def calculate_pass_at_k(self, task_results: List[TaskResult], k_values: List[int]) -> Dict[int, float]:
        """Calculate pass@k metrics."""
        return self.metrics_calculator.calculate_pass_at_k(task_results, k_values)
        
    def _evaluate_with_judge(self, task_result: TaskResult) -> Dict[str, float]:
        """Evaluate task result using LLM judge."""
        if not self.judge or not task_result.trajectory:
            return {}
            
        # Get final image from trajectory
        final_image = None
        if task_result.trajectory:
            # Try to get image from final state
            final_state = task_result.trajectory[-1][1]  # (action, state, feedback)
            final_image = getattr(final_state, 'image', None)
            
        if not final_image:
            return {}
            
        # Create task description
        task_description = f"Task: {task_result.task_type.value} puzzle"
        if task_result.metadata.get("description"):
            task_description += f". {task_result.metadata['description']}"
            
        # Create trajectory summary
        trajectory_summary = []
        for i, (action, state, feedback) in enumerate(task_result.trajectory):
            trajectory_summary.append(f"Step {i+1}: {action.action_type} - {feedback}")
            
        # Get judge evaluation
        judge_success, confidence, reasoning = self.judge.judge_success(
            final_image, task_description, trajectory_summary
        )
        
        return {
            "judge_success": 1.0 if judge_success else 0.0,
            "judge_confidence": confidence,
            "human_agreement": self._calculate_human_agreement(task_result.success, judge_success)
        }
        
    def _calculate_human_agreement(self, system_success: bool, judge_success: bool) -> float:
        """Calculate agreement between system and judge evaluation."""
        return 1.0 if system_success == judge_success else 0.0
        
    def export_results_to_excel(self, evaluation_result: EvaluationResult, 
                               output_path: str, model_name: str = "unknown") -> None:
        """Export evaluation results to Excel file."""
        # Prepare data for Excel
        results_data = []
        
        for task_result in evaluation_result.task_results:
            single_metrics = self.evaluate_single_task(task_result)
            
            row_data = {
                "model_name": model_name,
                "task_id": task_result.task_id,
                "task_type": task_result.task_type.value,
                "difficulty": task_result.metadata.get("difficulty", "unknown"),
                "success": task_result.success,
                "total_steps": task_result.total_steps,
                "execution_time": task_result.execution_time,
                "optimal_steps": task_result.metadata.get("optimal_steps", 0),
                "total_tokens": task_result.metadata.get("total_tokens", 0),
                "timestamp": datetime.now().isoformat()
            }
            
            # Add calculated metrics
            row_data.update(single_metrics)
            
            results_data.append(row_data)
            
        # Create DataFrame
        df = pd.DataFrame(results_data)
        
        # Add summary row
        summary_data = {
            "model_name": model_name,
            "task_id": "SUMMARY",
            "task_type": "ALL",
            "difficulty": "ALL",
            "success": evaluation_result.accuracy,
            "total_steps": evaluation_result.detailed_metrics.get("step_efficiency", 0),
            "execution_time": evaluation_result.detailed_metrics.get("time_efficiency", 0),
            "optimal_steps": 0,
            "total_tokens": evaluation_result.token_efficiency if evaluation_result.token_efficiency != float('inf') else 0,
            "timestamp": datetime.now().isoformat(),
            "step_efficiency": evaluation_result.detailed_metrics.get("step_efficiency", 0),
            "time_efficiency": evaluation_result.detailed_metrics.get("time_efficiency", 0),
            "token_efficiency": evaluation_result.token_efficiency if evaluation_result.token_efficiency != float('inf') else 0,
            "distance_to_optimal": evaluation_result.distance_to_optimal if evaluation_result.distance_to_optimal != float('inf') else 0
        }
        
        # Add pass@k metrics
        for k, rate in evaluation_result.pass_at_k.items():
            summary_data[f"pass_at_{k}"] = rate
            
        summary_df = pd.DataFrame([summary_data])
        final_df = pd.concat([df, summary_df], ignore_index=True)
        
        # Save to Excel
        if os.path.exists(output_path):
            # Append to existing file
            try:
                existing_df = pd.read_excel(output_path)
                final_df = pd.concat([existing_df, final_df], ignore_index=True)
            except Exception as e:
                print(f"Could not read existing Excel file: {e}")
                
        final_df.to_excel(output_path, index=False)
        print(f"Results exported to {output_path}")
        
    def export_detailed_report(self, evaluation_result: EvaluationResult, 
                              output_dir: str, model_name: str = "unknown") -> None:
        """Export detailed evaluation report."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Main report
        report = {
            "model_name": model_name,
            "evaluation_timestamp": datetime.now().isoformat(),
            "summary": {
                "total_tasks": len(evaluation_result.task_results),
                "successful_tasks": sum(1 for r in evaluation_result.task_results if r.success),
                "accuracy": evaluation_result.accuracy,
                "pass_at_k": evaluation_result.pass_at_k,
                "distance_to_optimal": evaluation_result.distance_to_optimal,
                "token_efficiency": evaluation_result.token_efficiency
            },
            "detailed_metrics": evaluation_result.detailed_metrics,
            "task_breakdown": []
        }
        
        # Add individual task results
        for task_result in evaluation_result.task_results:
            task_data = {
                "task_id": task_result.task_id,
                "task_type": task_result.task_type.value,
                "success": task_result.success,
                "total_steps": task_result.total_steps,
                "execution_time": task_result.execution_time,
                "metadata": task_result.metadata,
                "trajectory_length": len(task_result.trajectory),
                "error_message": task_result.error_message
            }
            report["task_breakdown"].append(task_data)
            
        # Save main report
        report_path = os.path.join(output_dir, f"{model_name}_evaluation_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
            
        # Save trajectory details separately
        trajectories_path = os.path.join(output_dir, f"{model_name}_trajectories.json")
        trajectories_data = {}
        
        for task_result in evaluation_result.task_results:
            trajectory = []
            for action, state, feedback in task_result.trajectory:
                trajectory.append({
                    "action": action.to_dict(),
                    "state": state.to_dict(),
                    "feedback": feedback
                })
            trajectories_data[task_result.task_id] = trajectory
            
        with open(trajectories_path, 'w') as f:
            json.dump(trajectories_data, f, indent=2, default=str)
            
        print(f"Detailed report exported to {output_dir}")
        
    def compare_models(self, evaluation_results: Dict[str, EvaluationResult]) -> Dict[str, Any]:
        """Compare evaluation results across multiple models."""
        comparison = {
            "models": list(evaluation_results.keys()),
            "metrics_comparison": {},
            "rankings": {},
            "statistical_significance": {}
        }
        
        # Compare key metrics
        metrics_to_compare = ["accuracy", "distance_to_optimal", "token_efficiency"]
        
        for metric in metrics_to_compare:
            comparison["metrics_comparison"][metric] = {}
            
            for model_name, result in evaluation_results.items():
                if metric == "accuracy":
                    value = result.accuracy
                elif metric == "distance_to_optimal":
                    value = result.distance_to_optimal
                elif metric == "token_efficiency":
                    value = result.token_efficiency
                else:
                    continue
                    
                comparison["metrics_comparison"][metric][model_name] = value
                
        # Create rankings
        for metric in metrics_to_compare:
            metric_values = comparison["metrics_comparison"][metric]
            
            # Sort by metric value (higher is better for accuracy, lower is better for others)
            if metric == "accuracy":
                sorted_models = sorted(metric_values.items(), key=lambda x: x[1], reverse=True)
            else:
                # For distance_to_optimal and token_efficiency, lower is better
                sorted_models = sorted(metric_values.items(), key=lambda x: x[1])
                
            comparison["rankings"][metric] = [model for model, _ in sorted_models]
            
        return comparison
