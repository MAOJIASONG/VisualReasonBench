#!/usr/bin/env python3
"""
Basic Usage Example for PhyVPuzzle

This script demonstrates how to use PhyVPuzzle to evaluate VLM performance
on physical visual reasoning tasks.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

from phyvpuzzle import (
    PipelineConfig,
    create_pipeline,
    create_task_config,
    evaluate_model_performance,
    ComprehensiveEvaluator
)


def main():
    """Main example function."""
    print("PhyVPuzzle Basic Usage Example")
    print("=" * 40)
    
    # Check for API key
    if not os.getenv('OPENAI_API_KEY'):
        print("Warning: OPENAI_API_KEY not set. Using mock evaluation.")
        mock_mode = True
    else:
        mock_mode = False
    
    # Create pipeline configuration
    config = PipelineConfig(
        vllm_type="openai" if not mock_mode else "huggingface",
        vllm_model="gpt-4-vision-preview",
        translator_type="rule_based",
        max_iterations=50,
        timeout=180.0,
        enable_logging=True,
        log_level="INFO"
    )
    
    print(f"Configuration:")
    print(f"  VLLM Type: {config.vllm_type}")
    print(f"  VLLM Model: {config.vllm_model}")
    print(f"  Translator: {config.translator_type}")
    print(f"  Max Iterations: {config.max_iterations}")
    print(f"  Timeout: {config.timeout}s")
    print()
    
    try:
        # Create pipeline
        print("Initializing pipeline...")
        pipeline = create_pipeline(config)
        
        # For demo purposes, we'll create mock tasks
        print("Creating sample tasks...")
        
        # Note: In a real implementation, you would create actual task instances
        # Here we're just showing the structure
        
        mock_tasks = []
        task_types = ["puzzle", "lego", "dominoes"]
        
        for task_type in task_types:
            for difficulty in ["easy", "medium"]:
                task_config = create_task_config(
                    task_type=task_type,
                    difficulty=difficulty,
                    max_steps=30,
                    time_limit=120.0
                )
                print(f"  Created {task_type} task ({difficulty})")
                # mock_tasks.append(create_task_instance(task_config))
        
        print(f"Created {len(mock_tasks)} sample tasks")
        
        # Simulate evaluation results
        print("\nRunning evaluation...")
        if mock_mode:
            print("Mock mode: Simulating evaluation results...")
            
            # Create mock results
            from phyvpuzzle.evaluation.metrics import EvaluationResult
            
            mock_result = EvaluationResult(
                total_tasks=len(mock_tasks) if mock_tasks else 6,
                successful_tasks=4,
                metrics={
                    "accuracy": 0.667,
                    "pass_at_8": 0.800,
                    "distance_to_optimal": 3.5,
                    "step_efficiency": 0.521,
                    "time_efficiency": 0.445,
                    "robustness": 0.588
                },
                task_results=[],
                per_task_metrics=[],
                metadata={"mock_mode": True}
            )
            
            # Generate report
            evaluator = ComprehensiveEvaluator()
            report = evaluator.generate_report(mock_result)
            print(report)
            
        else:
            # Real evaluation would go here
            print("Real evaluation not implemented in this example")
            print("Please implement actual task classes to run real evaluation")
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("\nExample completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main()) 