#!/usr/bin/env python3
"""
Human agent test with predefined actions defined in script.

This demonstrates how to use the human agent with automated action sequences
without modifying configuration files.
"""

import sys
import os
from pathlib import Path

from phyvpuzzle import load_config, BenchmarkRunner, validate_config
from phyvpuzzle.agents import HumanAgent

# Define different action sequences for different scenarios
PUZZLE_BASIC_ACTIONS = [
    # "pick(object_id=1)",
    # "place(object_id=1, position=[0.0, 0.0, 0.45])"
    # "pick(object_id=3)",
    # "place(object_id=3, position=[0.0, 0.0, 0.45])",
    # "pick(object_id=\"piece_3\")",
    # "place(object_id=\"piece_3\", position=[0.0, 0.0, 0.45])",
    "place_piece(piece_name=\"piece_1\", precise=true)",
    "place_piece(piece_name=\"piece_2\", precise=true)", 
    "place_piece(piece_name=\"piece_3\", precise=true)",
    # "align_pieces(piece1=\"piece_1\", piece2=\"piece_2\", direction=\"right\")",
    # "place_piece(piece_name=\"piece_4\", precise=true)",
    # "finish"
]

PUZZLE_ADVANCED_ACTIONS = [
    # # Try manual pick and place first
    # "pick(object_id=\"piece_1\")",
    # "place(object_id=\"piece_1\", position=[0.0, 0.0, 0.45])",
    
    # # Use puzzle-specific tools
    # "place_piece(piece_name=\"piece_2\", precise=true)",
    # "place_piece(piece_name=\"piece_3\", precise=true)",
    
    # # Align pieces
    # "align_pieces(piece1=\"piece_1\", piece2=\"piece_2\", direction=\"right\")",
    # "align_pieces(piece1=\"piece_2\", piece2=\"piece_3\", direction=\"right\")",
    
    # # Wait to observe
    # "wait",
    
    # # Continue with remaining pieces
    # "place_piece(piece_name=\"piece_4\", precise=true)",
    # "place_piece(piece_name=\"piece_5\", precise=true)",
    # "place_piece(piece_name=\"piece_6\", precise=true)",
    # "place_piece(piece_name=\"piece_7\", precise=true)",
    
    # "finish"
]

DOMINO_ACTIONS = [
    "push_specific_domino(domino_id=\"domino_1\", force=3.0, direction=[1, 0, 0])",
    "wait",
    "push_specific_domino(domino_id=\"domino_5\", force=5.0, direction=[0, 1, 0])",
    "wait",
    "reset_dominoes()",
    "push_specific_domino(domino_id=\"domino_1\", force=8.0, direction=[1, 0, 0])",
    "finish"
]

def run_with_actions(config_file: str, actions: list, experiment_name: str):
    """Run human agent with specified actions."""
    
    print(f"\n" + f" 🤖 {experiment_name} ".center(80, "="))
    print(f"Running with {len(actions)} predefined actions...")
    
    # Load base configuration
    config_path = Path(__file__).resolve().parent.parent / "eval_configs" / config_file
    
    if not config_path.exists():
        print(f"❌ Configuration file not found: {config_path}")
        return False
        
    try:
        config = load_config(str(config_path))
        validation_errors = validate_config(config)
        if validation_errors:
            print(f"❌ Configuration is not valid: {validation_errors}")
            return False
        
        # Override experiment name
        config.runner.experiment_name = experiment_name
        
        print("\n" + "📋 Experiment Configuration".center(50, "-"))
        print(f"  Experiment Name   : {config.runner.experiment_name}")
        print(f"  Agent Type        : {config.agent.type}")
        print(f"  Task Type         : {config.task.type} ({config.task.difficulty.value})")
        print(f"  Max Steps         : {config.environment.max_steps}")
        print(f"  Actions Count     : {len(actions)}")
        print("-" * 50)
        
        # Show action preview
        print("\n🎬 Action Sequence Preview:")
        for i, action in enumerate(actions[:5], 1):  # Show first 5 actions
            print(f"  {i}. {action}")
        if len(actions) > 5:
            print(f"  ... and {len(actions) - 5} more actions")
        
        # Initialize benchmark runner
        print("\n🚀 Initializing human agent with action list...")
        runner = BenchmarkRunner(config)
        
        print("\n" + "🤖 Starting Automated Execution...".center(60, "-"))
        
        # Setup the runner (this creates the agent)
        runner.setup()
        
        # CRITICAL: Set the action list BEFORE running the benchmark
        if isinstance(runner.agent, HumanAgent):
            runner.agent.set_action_list(actions)
            print(f"✅ Action list set with {len(actions)} actions")
        else:
            print("⚠️  Warning: Agent is not a HumanAgent, cannot set action list")
            return False
        
        # Run single task to avoid agent recreation
        task_result = runner.run_single_task()
        
        # Create evaluation result
        from phyvpuzzle.core.base import EvaluationResult
        evaluation_result = EvaluationResult(
            accuracy=1.0 if task_result.success else 0.0,
            pass_at_k={1: 1.0 if task_result.success else 0.0},
            token_efficiency=0,
            distance_to_optimal=1.0,
            detailed_metrics={"success_rate": 1.0 if task_result.success else 0.0},
            task_results=[task_result]
        )
        
        # Results
        print("\n" + f"🏁 {experiment_name} Results".center(80, "="))
        
        if evaluation_result.accuracy > 0.5:
            print("🎉 SUCCESS! The automated sequence completed the task!")
        else:
            print("📊 Task completed. Sequence executed but task not fully solved.")
            
        print(f"\n📊 Performance Metrics:")
        print(f"  • Success Rate: {evaluation_result.accuracy:.1%}")
        if evaluation_result.pass_at_k:
            for k, rate in evaluation_result.pass_at_k.items():
                print(f"  • Pass@{k}: {rate:.1%}")
        
        print(f"\n📁 Output Files:")
        print(f"  ➡️  Log Directory : {runner.logger.run_dir}")
        print(f"  ➡️  Results Excel : {runner.logger.run_dir}/{config.runner.results_excel_path}")
        print(f"  ➡️  Full Log      : {runner.logger.run_dir}/experiment_log.json")
        print(f"  ➡️  Images        : {runner.logger.run_dir}/images/")
        
        return evaluation_result.accuracy > 0.5
        
    except Exception as e:
        print(f"\n❌ Execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function with multiple test scenarios."""
    
    print("=" * 80)
    print("🎮 Human Agent Action List Testing Suite")
    print("=" * 80)
    
    # Test scenarios
    scenarios = [
        {
            "name": "Puzzle Basic Test",
            "config": "human_puzzle_test.yaml",
            "actions": PUZZLE_BASIC_ACTIONS,
            "experiment": "human_puzzle_basic_actions"
        },
        {
            "name": "Puzzle Advanced Test", 
            "config": "human_puzzle_test.yaml",
            "actions": PUZZLE_ADVANCED_ACTIONS,
            "experiment": "human_puzzle_advanced_actions"
        }
    ]
    
    # Allow user to choose scenario
    if len(sys.argv) > 1:
        scenario_name = sys.argv[1].lower()
        if scenario_name == "basic":
            scenarios = [scenarios[0]]
        elif scenario_name == "advanced":
            scenarios = [scenarios[1]]
        elif scenario_name == "all":
            pass  # Run all scenarios
        else:
            print(f"Unknown scenario: {scenario_name}")
            print("Available scenarios: basic, advanced, all")
            return 1
    else:
        # Default to basic
        scenarios = [scenarios[0]]
    
    results = []
    for scenario in scenarios:
        print(f"\n{'='*20} {scenario['name']} {'='*20}")
        success = run_with_actions(
            scenario["config"],
            scenario["actions"], 
            scenario["experiment"]
        )
        results.append((scenario["name"], success))
        
        if len(scenarios) > 1:
            input("\nPress Enter to continue to next scenario...")
    
    # Final summary
    print("\n" + "📊 Final Results Summary".center(80, "="))
    for name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"  {name}: {status}")
    
    passed = sum(1 for _, success in results if success)
    print(f"\nOverall: {passed}/{len(results)} scenarios passed")
    print("=" * 80)
    
    return 0 if passed == len(results) else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
