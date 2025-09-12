#!/usr/bin/env python3
# """A quick demonstration of the PhyVPuzzle system using a simple domino task."""
# import debugpy; debugpy.connect(("localhost", 9501))
import sys
import os
from pathlib import Path

from phyvpuzzle import load_config, BenchmarkRunner, validate_config

# Try to load environment variables from a .env file (optional)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Note: python-dotenv not available. Using system environment variables only.")

def check_api_keys():
    """Check for and print a warning if the API key is not set."""
    if not os.getenv("OPENAI_API_KEY"):
        print("\n" + "="*80)
        print("⚠️  Warning: OPENAI_API_KEY not found in environment variables.")
        print("Please create a .env file in the root directory with your API key:")
        print("  OPENAI_API_KEY='your-key-here'")
        print("Or export it as an environment variable.")
        print("The script will likely fail without it.")
        print("="*80 + "\n")

def main():
    """Main function to run the domino task demo."""
    check_api_keys()

    print("\n" + " Domino Quick Test ".center(80, "="))
    
    # Load configuration from the YAML file
    config_path = Path(__file__).resolve().parent.parent / "eval_configs" / "domino_quick.yaml"
    
    if not config_path.exists():
        print(f"❌ Configuration file not found: {config_path}")
        return 1
        
    try:
        config = load_config(str(config_path))
        if validate_config(config):
            print(f"❌ Configuration is not valid: {validate_config(config)}")
            raise ValueError("Configuration is not valid")
        
        print("\n" + "📋 Experiment Configuration".center(40, "-"))
        print(f"  Experiment Name: {config.runner.experiment_name}")
        print(f"  Agent (subject) : {config.agent.model_name}")
        print(f"  Task Type      : {config.task.type} ({config.task.difficulty.value})")
        print(f"  Num Dominoes   : {config.task.num_dominoes}")
        print(f"  Arrangement    : {config.task.arrangement_pattern}")
        print("-" * 40)
        
        # Initialize and set up the benchmark runner
        print("\n🚀 Initializing benchmark runner...")
        runner = BenchmarkRunner(config)
        
        # --- BENCHMARK ---
        evaluation_result = runner.run_benchmark()

        # --- Final Summary ---
        print("\n" + "🏁 Final Summary".center(80, "="))
        print(f"\n  Log files saved to: {runner.logger.run_dir}")
        print(f"  ➡️  Summary: {runner.logger.run_dir}/summary.txt")
        print(f"  ➡️  Full Log: {runner.logger.run_dir}/experiment_log.json")
        print("=" * 80)
        
        return 0 if evaluation_result.accuracy > 0.5 else 1
        
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())