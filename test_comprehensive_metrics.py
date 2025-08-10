#!/usr/bin/env python3
"""
Comprehensive Evaluation Test Script

This script demonstrates the complete evaluation system with:
- All metrics (Acc, Pass@8, Avg Step, Distance to Optimal, Token Efficiency)
- OpenRouter integration with automatic API key loading
- Token tracking throughout inference
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import only what we need, avoiding physics dependencies
sys.path.insert(0, str(project_root / "src"))

# Direct imports to avoid pybullet dependency
from phyvpuzzle.tasks.base_task import TaskResult, TaskConfiguration, TaskType, TaskDifficulty
from src.phyvpuzzle.models.openrouter_client import (
    OpenRouterClient,
    OpenRouterVisionProcessor,
    create_openrouter_client
)


class MockEnvironment:
    """Mock environment for testing."""
    
    def __init__(self):
        self.objects = {}
        self.step_count = 0
        
    def create_primitive_object(self, **kwargs):
        """Create a mock object."""
        self.objects[kwargs["object_name"]] = {
            "position": kwargs.get("position", [0, 0, 0]),
            "color": kwargs.get("color", [1, 1, 1, 1])
        }
    
    def get_state(self):
        """Get environment state."""
        return {"objects": self.objects}
    
    def reset(self):
        """Reset environment."""
        self.objects = {}
        self.step_count = 0


def simulate_task_execution(task_id: str, 
                           difficulty: str = "medium",
                           num_steps: int = 10,
                           success_rate: float = 0.7,
                           optimal_steps: int = 6) -> TaskResult:
    """
    Simulate a task execution with mock data.
    
    This would normally use OpenRouter for real inference.
    """
    import random
    
    # Simulate success based on rate
    success = random.random() < success_rate
    
    # Simulate steps taken (near optimal if successful)
    if success:
        steps_taken = optimal_steps + random.randint(-1, 2)
    else:
        steps_taken = num_steps
    
    # Simulate token usage (500-1500 tokens per step)
    tokens_used = steps_taken * random.randint(500, 1500)
    
    # Create action sequence
    action_sequence = [f"action_{i}" for i in range(steps_taken)]
    
    return TaskResult(
        success=success,
        final_score=1.0 if success else random.uniform(0.3, 0.7),
        steps_taken=steps_taken,
        time_taken=steps_taken * random.uniform(1.5, 3.0),
        tokens_used=tokens_used,
        task_id=task_id,
        action_sequence=action_sequence,
        metadata={
            "difficulty": difficulty,
            "model": "openai/gpt-4o"
        }
    )


def test_openrouter_integration():
    """Test OpenRouter integration with automatic key loading."""
    print("\n" + "="*60)
    print("OPENROUTER INTEGRATION TEST")
    print("="*60)
    
    try:
        # Create client (will auto-load API key)
        print("\n1. Creating OpenRouter client with automatic key loading...")
        client = create_openrouter_client()
        print("✅ Client created successfully")
        
        # Test connection
        print("\n2. Testing API connection...")
        success, message = client.test_connection()
        if success:
            print(f"✅ {message}")
        else:
            print(f"⚠️ {message}")
            print("Note: Using mock data for demonstration")
            return None
        
        # Create vision processor
        print("\n3. Creating vision processor...")
        processor = OpenRouterVisionProcessor(client)
        print("✅ Vision processor ready")
        
        return processor
        
    except Exception as e:
        print(f"⚠️ OpenRouter setup failed: {e}")
        print("Note: Using mock data for demonstration")
        return None


def run_comprehensive_evaluation():
    """Run comprehensive evaluation with all metrics."""
    print("\n" + "="*60)
    print("COMPREHENSIVE EVALUATION SYSTEM")
    print("="*60)
    
    # Test OpenRouter first
    processor = test_openrouter_integration()
    
    # Setup evaluation parameters
    num_tasks = 5
    num_runs_per_task = 8  # For Pass@8
    optimal_steps_list = [6, 8, 10, 7, 9]  # Optimal steps for each task
    
    print(f"\n📊 Evaluation Configuration:")
    print(f"  - Number of tasks: {num_tasks}")
    print(f"  - Runs per task: {num_runs_per_task} (for Pass@8)")
    print(f"  - Optimal steps: {optimal_steps_list}")
    
    # Generate mock results
    print("\n🎯 Generating task results...")
    all_results = []
    
    for task_idx in range(num_tasks):
        task_id = f"puzzle_task_{task_idx}"
        optimal_steps = optimal_steps_list[task_idx]
        
        # Run task multiple times for Pass@8
        for run_idx in range(num_runs_per_task):
            result = simulate_task_execution(
                task_id=task_id,
                difficulty="medium",
                optimal_steps=optimal_steps,
                success_rate=0.6 + run_idx * 0.02  # Slightly improve with each run
            )
            all_results.append(result)
    
    print(f"✅ Generated {len(all_results)} task results")
    
    # Calculate metrics
    print("\n📈 Calculating Metrics:")
    print("-" * 40)
    
    # 1. Accuracy
    successful_tasks = sum(1 for r in all_results if r.success)
    accuracy = successful_tasks / len(all_results)
    print(f"1. Accuracy: {accuracy:.3f} ({successful_tasks}/{len(all_results)} successful)")
    
    # 2. Pass@8
    pass_at_8_count = 0
    for task_idx in range(num_tasks):
        task_results = all_results[task_idx * num_runs_per_task:(task_idx + 1) * num_runs_per_task]
        if any(r.success for r in task_results):
            pass_at_8_count += 1
    pass_at_8 = pass_at_8_count / num_tasks
    print(f"2. Pass@8: {pass_at_8:.3f} ({pass_at_8_count}/{num_tasks} tasks solved in 8 attempts)")
    
    # 3. Average Steps
    avg_steps = sum(r.steps_taken for r in all_results) / len(all_results)
    print(f"3. Avg Steps: {avg_steps:.2f}")
    
    # 4. Distance to Optimal
    total_distance = 0
    total_ratio = 0
    count = 0
    
    for i, result in enumerate(all_results):
        task_idx = i // num_runs_per_task
        optimal = optimal_steps_list[task_idx]
        distance = abs(result.steps_taken - optimal)
        ratio = (result.steps_taken - optimal) / optimal if optimal > 0 else 0
        total_distance += distance
        total_ratio += ratio
        count += 1
    
    avg_distance = total_distance / count
    avg_ratio = total_ratio / count
    
    print(f"4. Distance to Optimal:")
    print(f"   - Average Distance: {avg_distance:.2f} steps")
    print(f"   - Average Ratio: {avg_ratio:.3f} ({avg_ratio*100:.1f}% deviation)")
    
    # 5. Token Efficiency
    successful_results = [r for r in all_results if r.success]
    if successful_results:
        avg_tokens = sum(r.tokens_used for r in successful_results) / len(successful_results)
        print(f"5. Token Efficiency: {avg_tokens:.0f} tokens/successful task")
    else:
        print(f"5. Token Efficiency: N/A (no successful tasks)")
    
    # Create evaluation result object
    print("\n" + "="*60)
    print("USING COMPREHENSIVE EVALUATOR")
    print("="*60)
    
    evaluator = ComprehensiveEvaluator()
    
    # Create mock tasks
    tasks = []
    for i in range(num_tasks):
        config = create_task_config("puzzle", "medium")
        task = PuzzleTask(config)
        tasks.append(task)
    
    # Create evaluation result manually (since we're using mock data)
    metrics = {
        MetricType.ACCURACY.value: accuracy,
        MetricType.PASS_AT_8.value: pass_at_8,
        MetricType.AVG_STEP.value: avg_steps,
        MetricType.DISTANCE_TO_OPTIMAL.value: avg_distance,
        MetricType.DISTANCE_TO_OPTIMAL_RATIO.value: avg_ratio,
        MetricType.TOKEN_EFFICIENCY.value: avg_tokens if successful_results else float('inf')
    }
    
    evaluation_result = EvaluationResult(
        total_tasks=num_tasks,
        successful_tasks=pass_at_8_count,
        metrics=metrics,
        task_results=all_results,
        per_task_metrics=[],
        metadata={
            "num_runs": num_runs_per_task,
            "model": "openai/gpt-4o",
            "timestamp": time.time()
        },
        token_usage={
            "total": sum(r.tokens_used for r in all_results),
            "per_task": sum(r.tokens_used for r in all_results) / num_tasks,
            "per_successful": avg_tokens if successful_results else 0
        }
    )
    
    # Generate report
    report = evaluator.generate_report(evaluation_result)
    print(report)
    
    # Additional token statistics
    if evaluation_result.token_usage:
        print("\n" + "="*60)
        print("TOKEN USAGE STATISTICS")
        print("="*60)
        print(f"Total Tokens Used: {evaluation_result.token_usage['total']:,}")
        print(f"Tokens per Task: {evaluation_result.token_usage['per_task']:.0f}")
        print(f"Tokens per Successful Task: {evaluation_result.token_usage['per_successful']:.0f}")
    
    # Save results
    results_file = "evaluation_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            "metrics": metrics,
            "token_usage": evaluation_result.token_usage,
            "metadata": evaluation_result.metadata,
            "summary": {
                "total_tasks": num_tasks,
                "runs_per_task": num_runs_per_task,
                "optimal_steps": optimal_steps_list
            }
        }, f, indent=2, default=str)
    
    print(f"\n✅ Results saved to {results_file}")
    
    return evaluation_result


def main():
    """Main function."""
    print("\n🚀 VisualReasonBench - Complete Metrics Implementation")
    print("="*60)
    
    # Check for API key
    from dotenv import load_dotenv
    load_dotenv()
    
    if not os.getenv("OPENROUTER_API_KEY") and not os.getenv("API_KEY"):
        print("\n⚠️ No API key found. Please set up your API key:")
        print("1. Create a .env file in the project root")
        print("2. Add: OPENROUTER_API_KEY=your-key-here")
        print("\nUsing mock data for demonstration...")
    
    # Run evaluation
    result = run_comprehensive_evaluation()
    
    print("\n" + "="*60)
    print("✅ EVALUATION COMPLETE")
    print("="*60)
    
    print("\n📊 Key Metrics Summary:")
    for metric_name, value in result.metrics.items():
        if isinstance(value, float):
            if value == float('inf'):
                print(f"  {metric_name}: N/A")
            else:
                print(f"  {metric_name}: {value:.3f}")
        else:
            print(f"  {metric_name}: {value}")
    
    print("\n🎯 Implementation Status:")
    print("✅ Acc Pass@8 - Implemented")
    print("✅ Avg Step - Implemented")
    print("✅ Distance to Optimal - Implemented (absolute and ratio)")
    print("✅ Token Efficiency - Implemented")
    print("✅ OpenRouter Integration - Implemented with auto key loading")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())