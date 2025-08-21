"""
Command-line interface for PhyVPuzzle benchmark.
"""

import argparse
import sys
from typing import Optional
from pathlib import Path

from .core.config import Config, load_config, create_default_config, validate_config
from .core.base import TaskType, TaskDifficulty
from .runner import BenchmarkRunner
from .evaluation.evaluator import BenchmarkEvaluator
from .evaluation.metrics import MetricsCalculator


def create_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="PhyVPuzzle: Physics-based Visual Reasoning Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run single domino task
  phyvpuzzle run --config domino_config.yaml
  
  # Run benchmark with multiple trials
  phyvpuzzle benchmark --config config.yaml --num-runs 5
  
  # Evaluate existing results
  phyvpuzzle evaluate --results-dir logs/
  
  # Create default config
  phyvpuzzle create-config --output config.yaml
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Run command
    run_parser = subparsers.add_parser("run", help="Run single task instance")
    run_parser.add_argument("--config", "-c", required=True, help="Path to configuration file")
    run_parser.add_argument("--gui", action="store_true", help="Enable physics simulation GUI")
    run_parser.add_argument("--task-type", choices=[t.value for t in TaskType], help="Override task type")
    run_parser.add_argument("--difficulty", choices=[d.value for d in TaskDifficulty], help="Override difficulty")
    run_parser.add_argument("--model", help="Override model name")
    run_parser.add_argument("--output-dir", help="Override output directory")
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser("benchmark", help="Run complete benchmark")
    benchmark_parser.add_argument("--config", "-c", required=True, help="Path to configuration file")
    benchmark_parser.add_argument("--num-runs", "-n", type=int, default=1, help="Number of task runs")
    benchmark_parser.add_argument("--gui", action="store_true", help="Enable physics simulation GUI")
    benchmark_parser.add_argument("--model", help="Override model name")
    benchmark_parser.add_argument("--output-dir", help="Override output directory")
    
    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate existing results")
    eval_parser.add_argument("--results-dir", required=True, help="Directory containing result files")
    eval_parser.add_argument("--output", help="Output file for evaluation results")
    eval_parser.add_argument("--compare-models", nargs="+", help="Compare specific models")
    
    # Create config command
    config_parser = subparsers.add_parser("create-config", help="Create default configuration file")
    config_parser.add_argument("--output", "-o", default="config.yaml", help="Output configuration file")
    config_parser.add_argument("--task-type", choices=[t.value for t in TaskType], default="domino", help="Default task type")
    config_parser.add_argument("--difficulty", choices=[d.value for d in TaskDifficulty], default="very_easy", help="Default difficulty")
    
    # Validate config command
    validate_parser = subparsers.add_parser("validate-config", help="Validate configuration file")
    validate_parser.add_argument("config", help="Configuration file to validate")
    
    return parser


def run_command(args) -> int:
    """Execute run command."""
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Apply command line overrides
        if args.gui:
            config.environment.gui = True
        if args.task_type:
            config.task.type = TaskType(args.task_type)
        if args.difficulty:
            config.task.difficulty = TaskDifficulty(args.difficulty)
        if args.model:
            config.agent.model_name = args.model
        if args.output_dir:
            config.runner.log_dir = args.output_dir
            
        # Validate configuration
        issues = validate_config(config)
        for issue in issues:
            if issue.startswith("ERROR"):
                print(f"Configuration error: {issue}")
                return 1
            else:
                print(f"Configuration warning: {issue}")
                
        # Create and run benchmark
        runner = BenchmarkRunner(config)
        runner.setup()
        
        result = runner.run_single_task()
        
        print(f"\nTask Result:")
        print(f"Success: {result.success}")
        print(f"Steps: {result.total_steps}")
        print(f"Time: {result.execution_time:.2f}s")
        
        if result.error_message:
            print(f"Error: {result.error_message}")
            return 1
            
        return 0
        
    except Exception as e:
        print(f"Error running task: {e}")
        return 1


def benchmark_command(args) -> int:
    """Execute benchmark command."""
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Apply command line overrides
        if args.gui:
            config.environment.gui = True
        if args.model:
            config.agent.model_name = args.model
        if args.output_dir:
            config.runner.log_dir = args.output_dir
            
        # Validate configuration
        issues = validate_config(config)
        for issue in issues:
            if issue.startswith("ERROR"):
                print(f"Configuration error: {issue}")
                return 1
            else:
                print(f"Configuration warning: {issue}")
                
        # Create and run benchmark
        runner = BenchmarkRunner(config)
        runner.setup()
        
        runner.run_benchmark(num_runs=args.num_runs)
        
        return 0
        
    except Exception as e:
        print(f"Error running benchmark: {e}")
        return 1


def evaluate_command(args) -> int:
    """Execute evaluate command."""
    try:
        import json
        import os
        import glob
        
        results_dir = Path(args.results_dir)
        
        if not results_dir.exists():
            print(f"Results directory not found: {results_dir}")
            return 1
            
        # Find result files
        result_files = list(results_dir.glob("**/benchmark_result.json"))
        result_files.extend(results_dir.glob("**/*_evaluation_report.json"))
        
        if not result_files:
            print(f"No result files found in {results_dir}")
            return 1
            
        print(f"Found {len(result_files)} result files")
        
        # Load and evaluate results
        all_task_results = []
        
        for result_file in result_files:
            try:
                with open(result_file, 'r') as f:
                    data = json.load(f)
                    
                # Extract task results (implementation would depend on file format)
                # This is a simplified version
                print(f"Loaded results from: {result_file}")
                
            except Exception as e:
                print(f"Error loading {result_file}: {e}")
                continue
                
        # Calculate metrics
        calculator = MetricsCalculator()
        
        if all_task_results:
            evaluation_result = calculator.calculate_comprehensive_metrics(all_task_results)
            
            # Print results
            print("\nEvaluation Results:")
            print(f"Accuracy: {evaluation_result.accuracy:.1%}")
            
            if evaluation_result.pass_at_k:
                for k, rate in evaluation_result.pass_at_k.items():
                    print(f"Pass@{k}: {rate:.1%}")
                    
            # Save results if output specified
            if args.output:
                evaluator = BenchmarkEvaluator({})
                evaluator.export_results_to_excel(evaluation_result, args.output)
                print(f"Results saved to: {args.output}")
        else:
            print("No task results to evaluate")
            
        return 0
        
    except Exception as e:
        print(f"Error evaluating results: {e}")
        return 1


def create_config_command(args) -> int:
    """Execute create-config command."""
    try:
        config = create_default_config(args.output)
        
        # Apply overrides
        config.task.type = TaskType(args.task_type)
        config.task.difficulty = TaskDifficulty(args.difficulty)
        
        print(f"Created default configuration: {args.output}")
        print("Please edit the configuration file to set your API keys and other preferences.")
        
        return 0
        
    except Exception as e:
        print(f"Error creating config: {e}")
        return 1


def validate_config_command(args) -> int:
    """Execute validate-config command."""
    try:
        config = load_config(args.config)
        issues = validate_config(config)
        
        if issues:
            print("Configuration validation results:")
            for issue in issues:
                print(f"  {issue}")
                
            # Check if there are any errors
            errors = [issue for issue in issues if issue.startswith("ERROR")]
            if errors:
                print(f"\n{len(errors)} error(s) found. Please fix before running.")
                return 1
            else:
                print(f"\n{len(issues)} warning(s) found, but configuration is valid.")
        else:
            print("Configuration is valid!")
            
        return 0
        
    except Exception as e:
        print(f"Error validating config: {e}")
        return 1


def main() -> int:
    """Main CLI entry point."""
    parser = create_parser()
    
    if len(sys.argv) == 1:
        parser.print_help()
        return 1
        
    args = parser.parse_args()
    
    if args.command == "run":
        return run_command(args)
    elif args.command == "benchmark":
        return benchmark_command(args)
    elif args.command == "evaluate":
        return evaluate_command(args)
    elif args.command == "create-config":
        return create_config_command(args)
    elif args.command == "validate-config":
        return validate_config_command(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
