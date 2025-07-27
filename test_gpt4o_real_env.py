#!/usr/bin/env python3
"""
Test PhyVPuzzle Real Environment with GPT-4o API

This script tests the actual PhyVPuzzle physics environment with GPT-4o,
requiring PyBullet to be installed.
"""

import os
import sys
import json
import time
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Load environment variables
load_dotenv()

try:
    import pybullet as p
    PYBULLET_AVAILABLE = True
except ImportError:
    PYBULLET_AVAILABLE = False
    print("Warning: PyBullet not available. Install with: pip install pybullet")

from test_gpt4o_phyvpuzzle import GPT4oVLMProcessor, GPT4oDecisionParser


def test_real_phyvpuzzle_environment():
    """Test with real PhyVPuzzle environment if PyBullet is available."""
    
    if not PYBULLET_AVAILABLE:
        print("❌ PyBullet not available. Cannot run real environment test.")
        return False
    
    try:
        # Import PhyVPuzzle components with fallback for import issues
        try:
            from phyvpuzzle.environment.phyvpuzzle_env import (
                PuzzleType, PhyVPuzzleConfig, PhyVPuzzleEnvironment
            )
            from phyvpuzzle.environment.physics_env import CameraConfig
        except ImportError as e:
            print(f"❌ Cannot import PhyVPuzzle components: {e}")
            print("This may be due to missing core dependencies.")
            return False
        
        print("✓ PhyVPuzzle imports successful")
        
        # Create environment configuration
        models_base = os.path.join(os.path.dirname(__file__), "src", "phyvpuzzle", "environment", "phobos_models")
        config = PhyVPuzzleConfig(
            puzzle_type=PuzzleType.LUBAN_LOCK,
            urdf_base_path=models_base,
            meshes_path=os.path.join(models_base, "luban-simple-prismatic/base_link/meshes/stl"),
            initial_camera_config=CameraConfig(
                position=(-1.5, 2.2, 1.8),
                target=(0, 0, 0.5),
                fov=60.0,
                image_width=512,
                image_height=512
            ),
            max_steps=20,
            time_limit=300.0
        )
        
        print("✓ Environment configuration created")
        
        # Create environment
        env = PhyVPuzzleEnvironment(config, gui=False)  # Start without GUI for testing
        
        try:
            # Setup environment
            print("Setting up PhyVPuzzle environment...")
            env.setup_environment()
            print("✓ Environment setup successful")
            
            # Test observation
            observation = env.get_observation_for_vlm()
            print("✓ Observation obtained")
            print(f"  Image size: {observation['image'].size}")
            print(f"  Available actions: {len(observation['available_actions'])}")
            
            return True
            
        except Exception as e:
            print(f"❌ Environment setup failed: {e}")
            return False
        finally:
            env.close()
            
    except Exception as e:
        print(f"❌ Real environment test failed: {e}")
        return False


def run_gpt4o_with_real_environment():
    """Run GPT-4o with real PhyVPuzzle environment."""
    
    if not PYBULLET_AVAILABLE:
        print("❌ PyBullet required for real environment test")
        return False
    
    try:
        # Import PhyVPuzzle components
        from phyvpuzzle.environment.phyvpuzzle_env import (
            PuzzleType, PhyVPuzzleConfig, PhyVPuzzleEnvironment
        )
        from phyvpuzzle.environment.physics_env import CameraConfig
        
        # API setup
        api_key = os.getenv("api_key")
        base_url = os.getenv("base_url")
        
        if not api_key or not base_url:
            print("❌ Missing API credentials")
            return False
        
        processor = GPT4oVLMProcessor(api_key, base_url)
        parser = GPT4oDecisionParser()
        
        # Environment setup
        models_base = os.path.join(os.path.dirname(__file__), "src", "phyvpuzzle", "environment", "phobos_models")
        config = PhyVPuzzleConfig(
            puzzle_type=PuzzleType.LUBAN_LOCK,
            urdf_base_path=models_base,
            meshes_path=os.path.join(models_base, "luban-simple-prismatic/base_link/meshes/stl"),
            initial_camera_config=CameraConfig(
                position=(-1.5, 2.2, 1.8),
                target=(0, 0, 0.5),
                fov=60.0,
                image_width=512,
                image_height=512
            ),
            max_steps=15,
            time_limit=180.0
        )
        
        env = PhyVPuzzleEnvironment(config, gui=True)  # Enable GUI for visual feedback
        
        try:
            print("Setting up real PhyVPuzzle environment...")
            env.setup_environment()
            print("✓ Real environment ready")
            
            # Luban lock instruction
            instruction = """You are controlling a robotic system to solve a Luban lock puzzle. 
The Luban lock consists of interlocking wooden pieces that must be manipulated in a specific sequence.

Your goal is to either:
1. Disassemble the lock by carefully removing pieces in the correct order
2. Assemble the lock by placing pieces in their proper interlocking positions

Available actions:
- move_piece: Move a piece to a target position [x, y, z]
- rotate_piece: Rotate a piece around an axis (x, y, or z) by degrees
- slide_piece: Slide a piece in a direction [dx, dy, dz] by distance
- lift_piece: Lift a piece vertically by specified height
- insert_piece: Insert a piece into a slot position
- remove_piece: Remove a piece from current position
- check_solution: Check if the puzzle is solved

When specifying coordinates:
- X-axis: left(-) to right(+)
- Y-axis: back(-) to front(+) 
- Z-axis: down(-) to up(+)

Analyze the current state, identify key pieces and their constraints, then choose the most logical next action."""

            max_steps = 10
            results = []
            
            for step in range(max_steps):
                if env.is_task_complete():
                    break
                
                print(f"\n🔄 Step {step + 1}/{max_steps}")
                
                try:
                    # Get observation
                    observation = env.get_observation_for_vlm()
                    
                    # Prepare VLM input
                    vlm_input = {
                        "image": observation["image"],
                        "instruction": instruction,
                        "current_state": observation["state_description"],
                        "available_actions": observation["available_actions"],
                        "step_number": step + 1,
                        "max_steps": max_steps
                    }
                    
                    # Add recent context
                    if results:
                        recent = results[-2:]  # Last 2 steps
                        context = "Recent actions:\\n"
                        for i, r in enumerate(recent):
                            context += f"Step {len(results) - len(recent) + i + 1}: {r['action']} → {r['result']}\\n"
                        vlm_input["recent_context"] = context
                    
                    # Query GPT-4o
                    print("🤖 Querying GPT-4o...")
                    response = processor.process_input(vlm_input)
                    print("📝 GPT-4o Analysis:")
                    print(response[:200] + "..." if len(response) > 200 else response)
                    
                    # Parse decision
                    parsed_action = parser.parse_decision(response)
                    
                    if not parsed_action:
                        print("❌ Failed to parse action from response")
                        results.append({
                            "step": step + 1,
                            "action": "parse_failed",
                            "result": "Failed to parse GPT-4o response"
                        })
                        continue
                    
                    print(f"🎯 Parsed Action: {parsed_action['action_type']}")
                    if parsed_action.get("parameters"):
                        print(f"📋 Parameters: {parsed_action['parameters']}")
                    
                    # Create environment command
                    from phyvpuzzle.core.translator import EnvironmentCommand
                    command = EnvironmentCommand(
                        command_type=parsed_action["action_type"],
                        parameters=parsed_action.get("parameters", {}),
                        timestamp=time.time()
                    )
                    
                    # Execute command
                    print("⚙️ Executing command...")
                    success = env.execute_command(command)
                    result = "Success" if success else "Failed"
                    print(f"✅ Execution: {result}")
                    
                    # Record result
                    results.append({
                        "step": step + 1,
                        "action": parsed_action["action_type"],
                        "parameters": parsed_action.get("parameters", {}),
                        "result": result,
                        "response": response
                    })
                    
                    # Check task completion
                    task_success, reason = env.get_success_status()
                    if task_success:
                        print(f"🎉 Task completed: {reason}")
                        break
                    
                    # Pause for visualization
                    time.sleep(2)
                    
                except Exception as e:
                    print(f"❌ Error in step {step + 1}: {e}")
                    results.append({
                        "step": step + 1,
                        "action": "error",
                        "result": f"Error: {str(e)}"
                    })
                    continue
            
            # Final evaluation
            final_success, final_reason = env.get_success_status()
            
            print(f"\n{'='*60}")
            print("🏁 FINAL RESULTS")
            print(f"{'='*60}")
            print(f"📊 Total Steps: {len(results)}")
            print(f"🎯 Task Success: {final_success}")
            print(f"💭 Reason: {final_reason}")
            
            print(f"\n📋 Action Summary:")
            for result in results:
                action = result['action']
                status = result['result']
                print(f"  Step {result['step']:2d}: {action:15s} → {status}")
            
            # Save detailed results
            output_file = "gpt4o_phyvpuzzle_results.json"
            with open(output_file, 'w') as f:
                json.dump({
                    "success": final_success,
                    "reason": final_reason,
                    "total_steps": len(results),
                    "step_results": results,
                    "config": {
                        "puzzle_type": "luban_lock",
                        "max_steps": max_steps,
                        "model": "gpt-4o"
                    }
                }, f, indent=2, default=str)
            
            print(f"\n💾 Detailed results saved to: {output_file}")
            
            return final_success
            
        finally:
            env.close()
            print("🔒 Environment closed")
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("This may be due to missing dependencies or import path issues.")
        return False
    except Exception as e:
        print(f"❌ Real environment test failed: {e}")
        return False


def main():
    """Main test function."""
    print("🧩 GPT-4o Real PhyVPuzzle Environment Test")
    print("=" * 60)
    
    # Check PyBullet availability
    if not PYBULLET_AVAILABLE:
        print("⚠️ PyBullet not available. Please install:")
        print("   pip install pybullet")
        print("   Then run this test again.")
        return 1
    
    # Test 1: Environment creation
    print("\n🔧 Testing environment creation...")
    if not test_real_phyvpuzzle_environment():
        print("❌ Environment test failed. Check dependencies and file paths.")
        return 1
    
    # Test 2: Full GPT-4o integration
    print("\n🤖 Running GPT-4o with real environment...")
    try:
        success = run_gpt4o_with_real_environment()
        
        if success:
            print("\n🎉 SUCCESS! GPT-4o solved the PhyVPuzzle!")
        else:
            print("\n🎯 Test completed. GPT-4o attempted to solve the puzzle.")
            print("   Check the results for performance analysis.")
        
        print("\n📈 Performance Notes:")
        print("• Adjust prompt engineering for better results")
        print("• Fine-tune action parameters and thresholds")
        print("• Consider multi-step reasoning strategies")
        
        return 0
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())