#!/usr/bin/env python3
"""Quick test of very-easy domino task using the new PhyVPuzzle system."""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from phyvpuzzle.core.config import load_config
from phyvpuzzle.runner import BenchmarkRunner

load_dotenv()

def main():
    print("\n" + "="*80)
    print("DOMINO QUICK TEST - PhyVPuzzle v0.1.0")
    print("="*80)
    
    # Load configuration
    config_path = "eval_configs" / "domino_quick.yaml"
    
    if not config_path.exists():
        print(f"Configuration file not found: {config_path}")
        return 1
        
    try:
        # Load config
        config = load_config(str(config_path))
        
        print(f"\nConfiguration:")
        print(f"- Task: {config.task.name} ({config.task.difficulty.value})")
        print(f"- Model: {config.agent.model_name}")
        print(f"- Max steps: {config.task.max_steps}")
        print(f"- Dominoes: {config.task.parameters.get('num_dominoes', 'unknown')}")
        print(f"- Pattern: {config.task.parameters.get('arrangement_pattern', 'unknown')}")
        
        # Create and setup runner
        print("\nInitializing benchmark runner...")
        runner = BenchmarkRunner(config)
        runner.setup()
        
        # Execute task
        print("\nExecuting domino task...")
        print("-" * 60)
        
        result = runner.run_single_task()
        
        print("-" * 60)
        print(f"\nResults:")
        print(f"- Success: {result.success}")
        print(f"- Steps: {result.total_steps}")
        print(f"- Time: {result.execution_time:.2f}s")
        print(f"- Tokens: {result.metadata.get('total_tokens', 'unknown')}")
        
        if result.error_message:
            print(f"- Error: {result.error_message}")
            
        # Show log location
        print(f"\nLogs saved to: {runner.logger.run_dir}")
        
        print("\n" + "="*80)
        
        return 0 if result.success else 1
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())