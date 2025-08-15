"""
Command Line Interface Module

This module provides a command-line interface for the PhyVPuzzle benchmark.
"""

import argparse
import sys
import os
from typing import List, Optional, Dict, Any
import json
from pathlib import Path

from .core.pipeline import PhysicalReasoningPipeline, PipelineConfig, create_pipeline
from .tasks.base_task import BaseTask, create_task_config
from .evaluation.metrics import evaluate_model_performance, ComprehensiveEvaluator


def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="PhyVPuzzle: Physical Visual Reasoning Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run Domino task once with GUI
  phyvpuzzle run --task-type dominoes --difficulty easy --gui

  # Generate sample tasks
  phyvpuzzle generate --task-type dominoes --count 10 --output tasks.json

  # Evaluate with custom configuration
  phyvpuzzle evaluate --config config.json --output results.json
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate model performance')
    eval_parser.add_argument('--task-type', choices=['dominoes'], 
                            required=True, help='Type of task to evaluate')
    eval_parser.add_argument('--difficulty', choices=['very-easy', 'easy', 'medium', 'hard', 'expert'],
                            default='medium', help='Task difficulty level')
    eval_parser.add_argument('--num-runs', type=int, default=8,
                            help='Number of runs per task for Pass@K metric')
    eval_parser.add_argument('--vllm-type', choices=['openai', 'huggingface'],
                            default='openai', help='Type of VLLM to use')
    eval_parser.add_argument('--vllm-model', default='gpt-4o',
                            help='VLLM model name')
    eval_parser.add_argument('--translator-type', choices=['rule_based', 'llm'],
                            default='rule_based', help='Type of translator to use')
    eval_parser.add_argument('--output', '-o', help='Output file for results')
    eval_parser.add_argument('--config', help='Configuration file (JSON)')
    eval_parser.add_argument('--verbose', '-v', action='store_true',
                            help='Verbose output')
    
    # Run command
    run_parser = subparsers.add_parser('run', help='Run a single task')
    run_parser.add_argument('--task-type', choices=['dominoes'],
                           required=True, help='Type of task to run')
    run_parser.add_argument('--difficulty', choices=['very-easy', 'easy', 'medium', 'hard', 'expert'],
                           default='medium', help='Task difficulty level')
    run_parser.add_argument('--gui', action='store_true',
                           help='Show PyBullet GUI')
    run_parser.add_argument('--vllm-type', choices=['openai', 'huggingface'],
                           default='openai', help='Type of VLLM to use')
    run_parser.add_argument('--vllm-model', default='gpt-4o',
                           help='VLLM model name')
    run_parser.add_argument('--config', help='Configuration file (JSON)')
    run_parser.add_argument('--num-runs', type=int, default=1, help='Number of attempts for the task')
    run_parser.add_argument('--max-steps', type=int, default=5,
                           help='Maximum number of steps')
    run_parser.add_argument('--timeout', type=float, default=300.0,
                           help='Timeout in seconds')
    run_parser.add_argument('--physics-settle-time', type=float, default=2.0,
                           help='Time to wait for physics to settle after actions (seconds)')
    run_parser.add_argument('--verbose', '-v', action='store_true',
                           help='Verbose output')
    
    # Generate command
    gen_parser = subparsers.add_parser('generate', help='Generate sample tasks')
    gen_parser.add_argument('--task-type', choices=['dominoes'],
                           required=True, help='Type of task to generate')
    gen_parser.add_argument('--count', type=int, default=10,
                           help='Number of tasks to generate')
    gen_parser.add_argument('--difficulty', choices=['very-easy', 'easy', 'medium', 'hard', 'expert'],
                           default='medium', help='Task difficulty level')
    gen_parser.add_argument('--output', '-o', required=True,
                           help='Output file for generated tasks')
    
    return parser


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_path}' not found.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in configuration file: {e}")
        sys.exit(1)


def save_results(results: Dict[str, Any], output_path: str) -> None:
    """Save results to JSON file."""
    try:
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Results saved to {output_path}")
    except Exception as e:
        print(f"Error saving results: {e}")


def create_sample_tasks(task_type: str, difficulty: str, count: int) -> List[BaseTask]:
    """Create sample tasks for testing."""
    tasks = []
    
    for i in range(count):
        config = create_task_config(
            task_type=task_type,
            difficulty=difficulty,
            max_steps=100,
            parameters={"task_id": f"{task_type}_{difficulty}_{i}"}
        )
        
        # Create task based on type
        if task_type == "puzzle":
            from .tasks.puzzle_task import PuzzleTask
            task = PuzzleTask(config)
        elif task_type == "lego":
            from .tasks.lego_task import LegoTask
            task = LegoTask(config)
        elif task_type == "dominoes":
            from .tasks.domino_task import DominoTask
            task = DominoTask(config)
        else:
            raise ValueError(f"Unknown task type: {task_type}")
        
        tasks.append(task)
    
    return tasks


def run_evaluation(args) -> None:
    """Run evaluation command."""
    # Load configuration
    config_dict = {}
    if args.config:
        config_dict = load_config(args.config)
    
    # Create pipeline configuration
    pipeline_config = PipelineConfig(
        vllm_type=args.vllm_type,
        vllm_model=args.vllm_model,
        translator_type=args.translator_type,
        enable_logging=args.verbose,
        log_level="DEBUG" if args.verbose else "INFO"
    )
    
    # Override with config file values
    for key, value in config_dict.items():
        if hasattr(pipeline_config, key):
            setattr(pipeline_config, key, value)
    
    print(f"Starting evaluation: {args.task_type} tasks, {args.difficulty} difficulty")
    print(f"VLLM: {pipeline_config.vllm_type} ({pipeline_config.vllm_model})")
    print(f"Translator: {pipeline_config.translator_type}")
    print(f"Runs per task: {args.num_runs}")
    
    try:
        # Create pipeline
        with create_pipeline(pipeline_config) as pipeline:
            # Create tasks
            tasks = create_sample_tasks(args.task_type, args.difficulty, 10)
            
            # Run evaluation
            print(f"Evaluating on {len(tasks)} tasks...")
            evaluation_result = evaluate_model_performance(
                pipeline, tasks, num_runs=args.num_runs
            )
            
            # Generate report
            evaluator = ComprehensiveEvaluator()
            report = evaluator.generate_report(evaluation_result)
            print(report)
            
            # Save results if requested
            if args.output:
                results_dict = {
                    "configuration": {
                        "task_type": args.task_type,
                        "difficulty": args.difficulty,
                        "num_runs": args.num_runs,
                        "vllm_type": pipeline_config.vllm_type,
                        "vllm_model": pipeline_config.vllm_model,
                        "translator_type": pipeline_config.translator_type
                    },
                    "results": {
                        "total_tasks": evaluation_result.total_tasks,
                        "successful_tasks": evaluation_result.successful_tasks,
                        "metrics": evaluation_result.metrics,
                        "metadata": evaluation_result.metadata
                    }
                }
                save_results(results_dict, args.output)
                
    except Exception as e:
        print(f"Error during evaluation: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def run_single_task(args) -> None:
    """Run single task command."""
    # Load configuration
    config_dict = {}
    default_config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'configs', 'default_config.json')
    default_config_path = os.path.abspath(default_config_path)
    if os.path.exists(default_config_path):
        try:
            with open(default_config_path, 'r') as f:
                config_dict = json.load(f)
        except Exception:
            config_dict = {}
    if args.config:
        try:
            with open(args.config, 'r') as f:
                config_dict.update(json.load(f))
        except Exception:
            pass

    # Create pipeline configuration with defaults and overrides
    pipeline_config = PipelineConfig(
        vllm_type=config_dict.get('vllm_type', args.vllm_type),
        vllm_model=config_dict.get('vllm_model', args.vllm_model),
        max_iterations=config_dict.get('max_iterations', args.max_steps),
        timeout=config_dict.get('timeout', args.timeout),
        physics_settle_time=config_dict.get('physics_settle_time', getattr(args, 'physics_settle_time', 2.0)),
        environment_type=config_dict.get('environment_type', 'pybullet'),
        gui=bool(args.gui),
        enable_logging=bool(config_dict.get('enable_logging', args.verbose)),
        log_level=config_dict.get('log_level', 'DEBUG' if args.verbose else 'INFO'),
        feedback_history_size=config_dict.get('feedback_history_size', 5),
        retry_attempts=config_dict.get('retry_attempts', 3),
    )
    
    print(f"Running single task: {args.task_type} ({args.difficulty})")
    print(f"VLLM: {pipeline_config.vllm_type} ({pipeline_config.vllm_model})")
    print(f"GUI: {'Enabled' if args.gui else 'Disabled'}")
    print(f"Physics settle time: {pipeline_config.physics_settle_time}s")
    
    try:
        # Create pipeline
        with create_pipeline(pipeline_config) as pipeline:
            # Create single task
            tasks = create_sample_tasks(args.task_type, args.difficulty, 1)
            task = tasks[0]
            
            # Run task
            print("Starting task execution...")
            result = pipeline.execute_task(task)
            
            # Display results
            print(f"\nTask completed!")
            print(f"Success: {result.success}")
            print(f"Final Score: {result.final_score:.3f}")
            print(f"Steps Taken: {result.steps_taken}")
            print(f"Time Taken: {result.time_taken:.2f}s")
            
            if result.error_message:
                print(f"Error: {result.error_message}")
            
            # Show execution summary
            summary = pipeline.get_execution_summary()
            print(f"\nExecution Summary:")
            print(f"Total Steps: {summary['total_steps']}")
            print(f"Successful Steps: {summary['successful_steps']}")
            print(f"Success Rate: {summary['success_rate']:.3f}")
            print(f"Average Step Time: {summary['avg_step_time']:.3f}s")
            
    except Exception as e:
        print(f"Error during task execution: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def generate_tasks(args) -> None:
    """Generate sample tasks command."""
    print(f"Generating {args.count} {args.task_type} tasks ({args.difficulty})")
    
    try:
        # Create tasks
        tasks = create_sample_tasks(args.task_type, args.difficulty, args.count)
        
        # Convert to serializable format
        task_data = []
        for i, task in enumerate(tasks):
            task_info = {
                "id": i,
                "type": args.task_type,
                "difficulty": args.difficulty,
                "description": task.get_task_description(),
                "max_steps": task.config.max_steps,
                "time_limit": task.config.time_limit,
                "parameters": task.config.parameters
            }
            task_data.append(task_info)
        
        # Save to file
        output_data = {
            "metadata": {
                "task_type": args.task_type,
                "difficulty": args.difficulty,
                "count": args.count,
                "generated_at": str(Path(__file__).stat().st_mtime)
            },
            "tasks": task_data
        }
        
        save_results(output_data, args.output)
        print(f"Generated {len(task_data)} tasks successfully!")
        
    except Exception as e:
        print(f"Error generating tasks: {e}")
        sys.exit(1)


def main() -> None:
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Set up environment
    if args.command in ['evaluate', 'run']:
        # Check for required environment variables
        if args.vllm_type == 'openai' and not os.getenv('OPENAI_API_KEY'):
            print("Warning: OPENAI_API_KEY environment variable not set")
    
    # Execute command
    if args.command == 'evaluate':
        run_evaluation(args)
    elif args.command == 'run':
        run_single_task(args)
    elif args.command == 'generate':
        generate_tasks(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main() 