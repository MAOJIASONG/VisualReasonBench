#!/usr/bin/env python3
"""
Final Metrics Implementation - Only 4 Required Metrics

This script demonstrates the implementation of exactly the 4 metrics specified:
1. Acc Pass@8
2. Avg Step  
3. Distance left towards the optimal steps (ratio)
4. Efficiency: Tokens used for successfully complete a task
"""

import os
import json
import random
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass


@dataclass
class TaskResult:
    """Result of a task execution."""
    success: bool
    steps_taken: int
    tokens_used: int
    task_id: str


class FourMetricsEvaluator:
    """Evaluator for the 4 required metrics only."""
    
    @staticmethod
    def calculate_acc_pass_at_8(results: List[TaskResult], num_tasks: int) -> float:
        """
        Metric 1: Acc Pass@8
        Calculate the accuracy of Pass@8 - percentage of tasks solved within 8 attempts.
        """
        # Group results by task_id
        task_groups = {}
        for result in results:
            if result.task_id not in task_groups:
                task_groups[result.task_id] = []
            task_groups[result.task_id].append(result)
        
        # Count tasks that succeeded within 8 attempts
        successful_tasks = 0
        for task_id, task_results in task_groups.items():
            # Take first 8 attempts
            attempts = task_results[:8]
            if any(r.success for r in attempts):
                successful_tasks += 1
        
        return successful_tasks / num_tasks if num_tasks > 0 else 0.0
    
    @staticmethod
    def calculate_avg_step(results: List[TaskResult]) -> float:
        """
        Metric 2: Avg Step
        Calculate average number of steps taken across all attempts.
        """
        if not results:
            return 0.0
        
        total_steps = sum(r.steps_taken for r in results)
        return total_steps / len(results)
    
    @staticmethod
    def calculate_distance_to_optimal(results: List[TaskResult], 
                                     optimal_steps: List[int]) -> float:
        """
        Metric 3: Distance left towards the optimal steps
        Calculate (Avg - Optimal) / Optimal as a ratio.
        """
        if not results or not optimal_steps:
            return float('inf')
        
        # Calculate average steps per task
        task_avg_steps = {}
        for result in results:
            task_idx = int(result.task_id.split('_')[-1])
            if task_idx not in task_avg_steps:
                task_avg_steps[task_idx] = []
            task_avg_steps[task_idx].append(result.steps_taken)
        
        # Calculate distance ratio for each task
        total_ratio = 0.0
        count = 0
        
        for task_idx, steps_list in task_avg_steps.items():
            if task_idx < len(optimal_steps):
                avg_steps = sum(steps_list) / len(steps_list)
                optimal = optimal_steps[task_idx]
                if optimal > 0:
                    # Formula: (Avg - Optimal) / Optimal
                    ratio = (avg_steps - optimal) / optimal
                    total_ratio += ratio
                    count += 1
        
        return total_ratio / count if count > 0 else float('inf')
    
    @staticmethod
    def calculate_token_efficiency(results: List[TaskResult]) -> float:
        """
        Metric 4: Efficiency - Tokens used for successfully complete a task
        Calculate average tokens used for successful task completions.
        """
        successful_results = [r for r in results if r.success]
        
        if not successful_results:
            return float('inf')
        
        total_tokens = sum(r.tokens_used for r in successful_results)
        return total_tokens / len(successful_results)


def generate_mock_results(num_tasks: int = 5, 
                         runs_per_task: int = 8,
                         optimal_steps: List[int] = None) -> List[TaskResult]:
    """Generate mock task results for testing."""
    if optimal_steps is None:
        optimal_steps = [6, 8, 10, 7, 9]
    
    results = []
    
    for task_idx in range(num_tasks):
        task_id = f"task_{task_idx}"
        optimal = optimal_steps[task_idx % len(optimal_steps)]
        
        for run_idx in range(runs_per_task):
            # Success probability increases with each attempt
            success_prob = 0.3 + (run_idx * 0.08)
            success = random.random() < success_prob
            
            # Steps taken
            if success:
                steps = optimal + random.randint(-1, 2)
            else:
                steps = optimal + random.randint(2, 6)
            
            # Tokens used (500-1500 per step)
            tokens = steps * random.randint(500, 1500)
            
            results.append(TaskResult(
                success=success,
                steps_taken=steps,
                tokens_used=tokens,
                task_id=task_id
            ))
    
    return results


def main():
    """Main function to demonstrate the 4 metrics."""
    print("\n" + "="*60)
    print("VisualReasonBench - 4 Required Metrics Implementation")
    print("="*60)
    
    # Configuration
    num_tasks = 5
    runs_per_task = 8
    optimal_steps = [6, 8, 10, 7, 9]
    
    print(f"\n📊 Configuration:")
    print(f"  • Number of tasks: {num_tasks}")
    print(f"  • Runs per task: {runs_per_task}")
    print(f"  • Optimal steps per task: {optimal_steps}")
    
    # Generate mock results
    results = generate_mock_results(num_tasks, runs_per_task, optimal_steps)
    
    # Calculate the 4 metrics
    evaluator = FourMetricsEvaluator()
    
    print("\n" + "="*60)
    print("METRIC RESULTS")
    print("="*60)
    
    # Metric 1: Acc Pass@8
    acc_pass_8 = evaluator.calculate_acc_pass_at_8(results, num_tasks)
    print(f"\n1. Acc Pass@8: {acc_pass_8:.3f}")
    print(f"   → {int(acc_pass_8 * num_tasks)}/{num_tasks} tasks solved within 8 attempts")
    
    # Metric 2: Avg Step
    avg_step = evaluator.calculate_avg_step(results)
    print(f"\n2. Avg Step: {avg_step:.2f}")
    print(f"   → Average steps taken across all {len(results)} attempts")
    
    # Metric 3: Distance to Optimal
    distance_ratio = evaluator.calculate_distance_to_optimal(results, optimal_steps)
    print(f"\n3. Distance to Optimal: {distance_ratio:.3f}")
    if distance_ratio != float('inf'):
        print(f"   → Formula: (Avg - Optimal) / Optimal")
        print(f"   → Interpretation: {distance_ratio*100:.1f}% deviation from optimal")
    
    # Metric 4: Token Efficiency
    token_efficiency = evaluator.calculate_token_efficiency(results)
    print(f"\n4. Efficiency (Tokens): {token_efficiency:.0f}")
    if token_efficiency != float('inf'):
        successful = sum(1 for r in results if r.success)
        print(f"   → Average tokens for {successful} successful completions")
    else:
        print(f"   → No successful tasks")
    
    # OpenRouter Integration Status
    print("\n" + "="*60)
    print("OPENROUTER INTEGRATION")
    print("="*60)
    
    # Check for API key
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key_found = bool(
        os.getenv("OPENROUTER_API_KEY") or 
        os.getenv("API_KEY") or
        os.getenv("api_key")
    )
    
    if api_key_found:
        print("✅ API key found - OpenRouter ready for inference")
        print("   → Key loaded from: environment/local .env file")
    else:
        print("⚠️ No API key found - Using mock data")
        print("   → To enable: Add OPENROUTER_API_KEY to .env file")
    
    # Save results
    output = {
        "metrics": {
            "acc_pass_8": acc_pass_8,
            "avg_step": avg_step,
            "distance_to_optimal_ratio": distance_ratio,
            "token_efficiency": token_efficiency
        },
        "configuration": {
            "num_tasks": num_tasks,
            "runs_per_task": runs_per_task,
            "optimal_steps": optimal_steps
        },
        "openrouter_ready": api_key_found
    }
    
    with open("four_metrics_results.json", "w") as f:
        json.dump(output, f, indent=2)
    
    print("\n📁 Results saved to: four_metrics_results.json")
    
    print("\n" + "="*60)
    print("✅ ALL 4 METRICS IMPLEMENTED SUCCESSFULLY")
    print("="*60)
    
    return 0


if __name__ == "__main__":
    exit(main())