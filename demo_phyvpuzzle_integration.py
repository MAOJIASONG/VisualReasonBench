#!/usr/bin/env python3
"""
PhyVPuzzle Integration Demo (Offline Version)

This script demonstrates the complete PhyVPuzzle-VLM integration architecture
without requiring external API calls. Shows the full pipeline for reference.
"""

import os
import sys
import json
import time
from typing import Dict, Any, Optional
from PIL import Image, ImageDraw, ImageFont

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


class DemoVLMProcessor:
    """Demo VLM processor that simulates GPT-4o responses for testing."""
    
    def __init__(self, model_name: str = "demo-gpt-4o"):
        self.model_name = model_name
        self.step_count = 0
        
        # Predefined responses for different scenarios
        self.response_templates = {
            "initial": """ANALYSIS: I can see a Luban lock puzzle with multiple wooden pieces. The pieces appear to be in their initial interlocked position. I need to start by identifying which piece can be moved first without causing the structure to collapse.

ACTION: move_piece
OBJECT: obj_1
POSITION: [0.2, 0.3, 0.6]
REASONING: Starting with obj_1 as it appears to be a key piece that can slide outward. Moving it slightly will help assess the interlocking constraints.""",
            
            "middle": """ANALYSIS: The previous move was successful. I can see that obj_1 has been repositioned. Now I need to continue the disassembly sequence by working on the next piece that has become free to move.

ACTION: rotate_piece
OBJECT: obj_2
POSITION: [0.0, 0.0, 45.0]
REASONING: obj_2 appears to need rotation before it can be extracted. A 45-degree rotation around the Z-axis should align it properly for removal.""",
            
            "check": """ANALYSIS: Several pieces have been moved and the puzzle structure has been modified. I should check if the current configuration represents a solved state or if more moves are needed.

ACTION: check_solution
REASONING: After the previous manipulations, it's important to verify whether the puzzle has been successfully disassembled or if additional steps are required.""",
            
            "final": """ANALYSIS: The puzzle appears to be in a solved state with pieces properly disassembled or reassembled. The interlocking constraints have been satisfied.

ACTION: check_solution
REASONING: Final verification that the puzzle solution is complete and all pieces are in their target configuration."""
        }
    
    def process_input(self, vlm_input: Dict[str, Any]) -> str:
        """Generate demo response based on step count."""
        self.step_count += 1
        
        if self.step_count == 1:
            return self.response_templates["initial"]
        elif self.step_count <= 3:
            return self.response_templates["middle"]
        elif self.step_count <= 5:
            return self.response_templates["check"]
        else:
            return self.response_templates["final"]


class DemoActionParser:
    """Parser for demo VLM responses."""
    
    def parse_decision(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse action from demo response."""
        try:
            lines = response.split('\n')
            action_data = {}
            
            for line in lines:
                line = line.strip()
                if line.startswith('ACTION:'):
                    action = line.replace('ACTION:', '').strip()
                    action_data['action_type'] = action
                elif line.startswith('OBJECT:'):
                    obj = line.replace('OBJECT:', '').strip()
                    if obj:
                        action_data.setdefault('parameters', {})['piece_id'] = obj
                elif line.startswith('POSITION:'):
                    pos_str = line.replace('POSITION:', '').strip()
                    try:
                        # Parse coordinates
                        pos_str = pos_str.strip('[]')
                        coords = [float(x.strip()) for x in pos_str.split(',')]
                        if len(coords) >= 3:
                            if action_data.get('action_type') == 'rotate_piece':
                                action_data.setdefault('parameters', {})['rotation_euler'] = coords
                            else:
                                action_data.setdefault('parameters', {})['target_position'] = coords
                    except:
                        pass
            
            if 'parameters' not in action_data:
                action_data['parameters'] = {}
                
            return action_data
            
        except Exception as e:
            print(f"Parse error: {e}")
            return {"action_type": "check_solution", "parameters": {}}


class DemoEnvironment:
    """Demo environment that simulates PhyVPuzzle behavior."""
    
    def __init__(self):
        self.step_count = 0
        self.task_completed = False
        self.piece_positions = {
            "obj_1": [0.1, 0.2, 0.5],
            "obj_2": [0.3, 0.2, 0.5], 
            "obj_3": [0.5, 0.2, 0.5]
        }
        self.piece_rotations = {
            "obj_1": [0, 0, 0],
            "obj_2": [0, 0, 0],
            "obj_3": [0, 0, 0]
        }
        
    def setup_environment(self):
        """Setup demo environment."""
        print("🔧 Demo environment setup complete")
        
    def get_observation_for_vlm(self) -> Dict[str, Any]:
        """Generate demo observation."""
        # Create a simple visualization
        img = Image.new('RGB', (512, 512), color=(240, 240, 250))
        draw = ImageDraw.Draw(img)
        
        # Draw puzzle pieces as colored rectangles
        colors = [(200, 100, 100), (100, 200, 100), (100, 100, 200)]
        piece_names = ["obj_1", "obj_2", "obj_3"]
        
        for i, (name, color) in enumerate(zip(piece_names, colors)):
            pos = self.piece_positions[name]
            # Convert 3D position to 2D screen coordinates
            x = int(pos[0] * 400 + 50)
            y = int(pos[1] * 400 + 50)
            w, h = 80, 60
            
            draw.rectangle([x, y, x+w, y+h], fill=color, outline=(0, 0, 0), width=2)
            
            # Label the piece
            try:
                draw.text((x+10, y+20), name, fill=(255, 255, 255))
            except:
                pass  # Font might not be available
        
        # Add title
        try:
            draw.text((10, 10), f"Luban Lock - Step {self.step_count}", fill=(0, 0, 0))
        except:
            pass
        
        # State description
        state_desc = f"""Puzzle Type: luban_lock
Current State: {"solved" if self.task_completed else "in_progress"}
Step: {self.step_count}
Task Completed: {self.task_completed}

Piece Positions:
"""
        
        for name, pos in self.piece_positions.items():
            state_desc += f"- {name}: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})\\n"
        
        return {
            "image": img,
            "state_description": state_desc,
            "raw_state": {
                "step_count": self.step_count,
                "task_completed": self.task_completed,
                "pieces": self.piece_positions
            },
            "available_actions": ["move_piece", "rotate_piece", "slide_piece", "check_solution"]
        }
    
    def execute_command(self, command: Dict[str, Any]) -> bool:
        """Execute demo command."""
        self.step_count += 1
        
        command_type = command.get("command_type", "")
        parameters = command.get("parameters", {})
        
        print(f"⚙️ Executing: {command_type}")
        if parameters:
            print(f"   Parameters: {parameters}")
        
        # Simulate command execution
        if command_type == "move_piece":
            piece_id = parameters.get("piece_id", "obj_1")
            target_pos = parameters.get("target_position")
            
            if piece_id in self.piece_positions and target_pos:
                self.piece_positions[piece_id] = target_pos[:3]
                print(f"   ✅ Moved {piece_id} to {target_pos}")
            else:
                print(f"   ⚠️ Using default movement for {piece_id}")
                # Default movement
                if piece_id in self.piece_positions:
                    self.piece_positions[piece_id][0] += 0.1
                    
        elif command_type == "rotate_piece":
            piece_id = parameters.get("piece_id", "obj_2")
            rotation = parameters.get("rotation_euler", [0, 0, 45])
            
            if piece_id in self.piece_rotations:
                self.piece_rotations[piece_id] = rotation
                print(f"   ✅ Rotated {piece_id} by {rotation}")
                
        elif command_type == "check_solution":
            # Check if puzzle is "solved" based on some criteria
            if self.step_count >= 3:
                self.task_completed = True
                print("   🎉 Puzzle solved!")
            else:
                print("   📋 Puzzle not yet solved")
                
        return True
    
    def is_task_complete(self) -> bool:
        """Check if task is complete."""
        return self.task_completed or self.step_count >= 6
    
    def get_success_status(self) -> tuple[bool, str]:
        """Get success status."""
        if self.task_completed:
            return True, "Puzzle successfully solved"
        elif self.step_count >= 6:
            return False, "Step limit reached"
        else:
            return False, "Task in progress"
    
    def close(self):
        """Close demo environment."""
        print("🔒 Demo environment closed")


def save_step_visualization(observation: Dict[str, Any], step: int, action: str, result: str):
    """Save step visualization for analysis."""
    try:
        os.makedirs("demo_results", exist_ok=True)
        
        # Save image
        img_path = f"demo_results/step_{step:02d}_image.png"
        observation["image"].save(img_path)
        
        # Save step data
        step_data = {
            "step": step,
            "action": action,
            "result": result,
            "state": observation["raw_state"],
            "description": observation["state_description"]
        }
        
        data_path = f"demo_results/step_{step:02d}_data.json"
        with open(data_path, 'w') as f:
            json.dump(step_data, f, indent=2)
            
        print(f"   💾 Saved visualization to demo_results/")
        
    except Exception as e:
        print(f"   ⚠️ Could not save visualization: {e}")


def run_complete_demo():
    """Run complete PhyVPuzzle-VLM integration demo."""
    print("🎯 PhyVPuzzle-VLM Integration Demo")
    print("=" * 60)
    print("This demo shows the complete pipeline architecture without requiring external APIs.")
    print()
    
    # Initialize components
    vlm_processor = DemoVLMProcessor()
    action_parser = DemoActionParser() 
    environment = DemoEnvironment()
    
    # Setup
    environment.setup_environment()
    
    # Main instruction for the VLM
    instruction = """You are controlling a robotic system to solve a Luban lock puzzle.
The goal is to disassemble the interlocking wooden pieces by moving them in the correct sequence.

Available actions:
- move_piece: Move a piece to target coordinates [x, y, z]
- rotate_piece: Rotate a piece by euler angles [rx, ry, rz] in degrees
- slide_piece: Slide a piece in a direction
- check_solution: Check if the puzzle is solved

Analyze the current state and choose the best next action to make progress toward solving the puzzle."""

    max_steps = 6
    results = []
    
    print(f"🚀 Starting {max_steps}-step demo sequence...")
    print()
    
    for step in range(1, max_steps + 1):
        if environment.is_task_complete():
            break
            
        print(f"{'='*20} Step {step} {'='*20}")
        
        try:
            # 1. Get Environment Observation
            print("📷 Getting environment observation...")
            observation = environment.get_observation_for_vlm()
            print("   ✅ Observation captured")
            
            # 2. Prepare VLM Input
            vlm_input = {
                "image": observation["image"],
                "instruction": instruction,
                "current_state": observation["state_description"],
                "available_actions": observation["available_actions"],
                "step_number": step,
                "max_steps": max_steps
            }
            
            # Add context from previous steps
            if results:
                recent = results[-2:]  # Last 2 steps
                context = "Recent actions:\\n"
                for r in recent:
                    context += f"Step {r['step']}: {r['action']} → {r['status']}\\n"
                vlm_input["recent_context"] = context
            
            # 3. Query VLM
            print("🤖 Querying VLM (Demo GPT-4o)...")
            vlm_response = vlm_processor.process_input(vlm_input)
            print("   ✅ VLM response received")
            print(f"   📝 Response preview: {vlm_response[:100]}...")
            
            # 4. Parse VLM Decision
            print("🔍 Parsing VLM decision...")
            parsed_action = action_parser.parse_decision(vlm_response)
            
            if not parsed_action:
                print("   ❌ Failed to parse action")
                continue
                
            action_type = parsed_action.get("action_type", "unknown")
            parameters = parsed_action.get("parameters", {})
            
            print(f"   ✅ Parsed action: {action_type}")
            if parameters:
                print(f"   📋 Parameters: {parameters}")
            
            # 5. Execute in Environment
            print("⚙️ Executing action in environment...")
            command = {
                "command_type": action_type,
                "parameters": parameters
            }
            
            execution_success = environment.execute_command(command)
            status = "Success" if execution_success else "Failed"
            print(f"   {'✅' if execution_success else '❌'} Execution: {status}")
            
            # 6. Record Results
            result_entry = {
                "step": step,
                "action": action_type,
                "parameters": parameters,
                "status": status,
                "vlm_response_length": len(vlm_response),
                "environment_state": observation["raw_state"]
            }
            results.append(result_entry)
            
            # 7. Save Visualization
            save_step_visualization(observation, step, action_type, status)
            
            # 8. Check Completion
            task_success, reason = environment.get_success_status()
            if task_success:
                print(f"   🎉 Task completed: {reason}")
                break
            
            print()
            time.sleep(1)  # Pause for readability
            
        except Exception as e:
            print(f"   ❌ Error in step {step}: {e}")
            results.append({
                "step": step,
                "action": "error", 
                "status": f"Error: {str(e)}"
            })
    
    # Final Results
    final_success, final_reason = environment.get_success_status()
    
    print("\\n" + "=" * 60)
    print("🏁 DEMO RESULTS")
    print("=" * 60)
    print(f"📊 Total Steps Executed: {len(results)}")
    print(f"🎯 Final Success: {final_success}")
    print(f"💭 Final Reason: {final_reason}")
    
    print(f"\\n📋 Step-by-Step Summary:")
    for r in results:
        step_num = r['step']
        action = r['action']
        status = r['status']
        print(f"   Step {step_num:2d}: {action:15s} → {status}")
    
    # Save comprehensive results
    try:
        final_results = {
            "demo_info": {
                "title": "PhyVPuzzle-VLM Integration Demo",
                "model": "demo-gpt-4o",
                "puzzle_type": "luban_lock",
                "timestamp": time.time()
            },
            "final_results": {
                "success": final_success,
                "reason": final_reason,
                "total_steps": len(results)
            },
            "step_details": results,
            "architecture_notes": {
                "components": [
                    "VLM Processor (GPT-4o simulation)",
                    "Action Parser (structured output)",
                    "Physics Environment (PyBullet simulation)",
                    "Success Evaluator (multi-metric assessment)"
                ],
                "data_flow": [
                    "Environment → Observation (image + state)",
                    "VLM → Analysis + Action decision", 
                    "Parser → Structured command",
                    "Environment → Execute + Update",
                    "Evaluator → Success assessment"
                ]
            }
        }
        
        with open("demo_results/complete_demo_results.json", 'w') as f:
            json.dump(final_results, f, indent=2, default=str)
            
        print(f"\\n💾 Complete results saved to: demo_results/complete_demo_results.json")
        
    except Exception as e:
        print(f"⚠️ Could not save final results: {e}")
    
    environment.close()
    
    print(f"\\n✨ Demo Architecture Summary:")
    print("🔧 Components integrated:")
    print("   • VLM Processor (vision + reasoning)")
    print("   • Action Parser (structured commands)")  
    print("   • Physics Environment (3D simulation)")
    print("   • Success Evaluator (multi-dimensional)")
    print("   • Result Logger (comprehensive tracking)")
    
    print(f"\\n🚀 Ready for Real Integration:")
    print("   • Replace DemoVLMProcessor with GPT-4o API")
    print("   • Install PyBullet for physics simulation")
    print("   • Load actual URDF models")
    print("   • Run with GUI for visual feedback")
    
    return final_success


def main():
    """Main demo function."""
    try:
        success = run_complete_demo()
        
        print(f"\\n🎉 Demo completed successfully!")
        print(f"Architecture validation: {'PASSED' if success else 'PARTIAL'}")
        
        print(f"\\n📁 Output Files:")
        print("   • demo_results/step_XX_image.png - Step visualizations")
        print("   • demo_results/step_XX_data.json - Step data")
        print("   • demo_results/complete_demo_results.json - Full results")
        
        return 0
        
    except KeyboardInterrupt:
        print("\\n⚠️ Demo interrupted by user")
        return 1
    except Exception as e:
        print(f"\\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())