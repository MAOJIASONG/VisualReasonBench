#!/usr/bin/env python3
"""A quick demonstration of the PhyVPuzzle system using a jigsaw puzzle assembly task."""
# import debugpy;debugpy.connect(("localhost", 9501))
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
    """Main function to run the puzzle assembly task demo."""
    check_api_keys()

    print("\n" + " 🧩 Puzzle Assembly Quick Test ".center(80, "="))
    
    # Load configuration from the YAML file
    config_path = Path(__file__).resolve().parent.parent / "eval_configs" / "puzzle_quick.yaml"
    
    if not config_path.exists():
        print(f"❌ Configuration file not found: {config_path}")
        return 1
        
    try:
        config = load_config(str(config_path))
        validation_errors = validate_config(config)
        if validation_errors:
            print(f"❌ Configuration is not valid: {validation_errors}")
            raise ValueError("Configuration is not valid")
        
        print("\n" + "📋 Experiment Configuration".center(50, "-"))
        print(f"  Experiment Name   : {config.runner.experiment_name}")
        print(f"  Agent (subject)   : {config.agent.model_name}")
        print(f"  Task Type         : {config.task.type} ({config.task.difficulty.value})")
        print(f"  Max Steps         : {config.environment.max_steps}")
        print("-" * 50)
        
        # Initialize and set up the benchmark runner
        print("\n🚀 Initializing puzzle benchmark runner...")
        runner = BenchmarkRunner(config)
        
        print("\n🎯 Task Overview:")
        print("  • Assemble scattered puzzle pieces into correct positions")
        print("  • Each piece has a unique color for identification")
        print("  • Use available tools to place and align pieces")
        print("\n" + "🎮 Starting Puzzle Challenge...".center(60, "-"))
        
        # --- BENCHMARK ---
        try:
            evaluation_result = runner.run_benchmark(num_runs=1)
        except Exception as benchmark_error:
            print(f"\n❌ Benchmark execution failed: {benchmark_error}")
            import traceback
            traceback.print_exc()
            return 1

        # --- Final Summary ---
        print("\n" + "🏁 Puzzle Challenge Results".center(80, "="))
        
        if evaluation_result.accuracy > 0.5:
            print("🎉 SUCCESS! The puzzle was completed successfully!")
        else:
            print("😔 Challenge not completed. Better luck next time!")
            
        print(f"\n📊 Performance Metrics:")
        print(f"  • Success Rate: {evaluation_result.accuracy:.1%}")
        if evaluation_result.pass_at_k:
            for k, rate in evaluation_result.pass_at_k.items():
                print(f"  • Pass@{k}: {rate:.1%}")
        if evaluation_result.token_efficiency != float('inf'):
            print(f"  • Token Efficiency: {evaluation_result.token_efficiency:.0f} tokens/success")
        if evaluation_result.distance_to_optimal != float('inf'):
            print(f"  • Step Efficiency: {evaluation_result.distance_to_optimal:.2f}x optimal")
            
        print(f"\n📁 Output Files:")
        print(f"  ➡️  Log Directory : {runner.logger.run_dir}")
        print(f"  ➡️  Results Excel : {runner.logger.run_dir}/{config.runner.results_excel_path}")
        print(f"  ➡️  Full Log      : {runner.logger.run_dir}/experiment_log.json")
        print(f"  ➡️  Images        : {runner.logger.run_dir}/images/")
        print("=" * 80)
        
        return 0 if evaluation_result.accuracy > 0.5 else 1
        
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    main()
