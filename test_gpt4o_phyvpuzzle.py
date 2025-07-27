#!/usr/bin/env python3
"""
Test PhyVPuzzle Environment with GPT-4o API

This script demonstrates the complete VLM evaluation pipeline using GPT-4o
for complex physics puzzle solving tasks.
"""

import os
import sys
import json
import base64
import time
from io import BytesIO
from typing import Dict, Any, Optional

import requests
from PIL import Image
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Load environment variables
load_dotenv()

class GPT4oVLMProcessor:
    """VLM processor using GPT-4o API for vision-language tasks."""
    
    def __init__(self, api_key: str, base_url: str):
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.model = "gpt-4o"
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        })
    
    def encode_image(self, image: Image.Image) -> str:
        """Encode PIL Image to base64 string."""
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        image_data = buffer.getvalue()
        return base64.b64encode(image_data).decode('utf-8')
    
    def process_input(self, vlm_input: Dict[str, Any]) -> str:
        """
        Process VLM input and return model response.
        
        Args:
            vlm_input: Dictionary containing image, instruction, and context
            
        Returns:
            Model response string
        """
        try:
            # Prepare the prompt
            instruction = vlm_input.get("instruction", "")
            current_state = vlm_input.get("current_state", "")
            available_actions = vlm_input.get("available_actions", [])
            step_number = vlm_input.get("step_number", 0)
            recent_context = vlm_input.get("recent_context", "")
            
            # Build the text prompt
            prompt = f"""You are controlling a robotic system to solve a physics puzzle. 

{instruction}

Current State:
{current_state}

Available Actions: {', '.join(available_actions)}

Step: {step_number}

{recent_context}

Please analyze the image and current state, then choose the best next action. 

Format your response as:
ANALYSIS: [Your analysis of the current situation]
ACTION: [action_name]
PARAMETERS: {{"param1": "value1", "param2": "value2"}}
REASONING: [Why you chose this action]

Focus on the physics constraints and puzzle-solving logic. Be specific with coordinates and parameters."""

            # Encode image
            image = vlm_input.get("image")
            if image is None:
                raise ValueError("No image provided in VLM input")
            
            image_base64 = self.encode_image(image)
            
            # Prepare API request
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_base64}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 500,
                "temperature": 0.7
            }
            
            # Make API request
            response = self.session.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                timeout=30
            )
            
            if response.status_code != 200:
                print(f"API Error: {response.status_code} - {response.text}")
                return f"Error: API request failed with status {response.status_code}"
            
            result = response.json()
            
            if "choices" not in result or not result["choices"]:
                return "Error: No response choices returned"
            
            return result["choices"][0]["message"]["content"]
            
        except Exception as e:
            print(f"Error in GPT-4o processing: {e}")
            return f"Error: {str(e)}"


class GPT4oDecisionParser:
    """Parser for extracting structured decisions from GPT-4o responses."""
    
    def parse_decision(self, response: str) -> Optional[Dict[str, Any]]:
        """
        Parse GPT-4o response into structured action.
        
        Args:
            response: Raw response from GPT-4o
            
        Returns:
            Parsed action dictionary or None if parsing fails
        """
        try:
            lines = response.strip().split('\n')
            parsed = {}
            
            for line in lines:
                line = line.strip()
                if line.startswith("ACTION:"):
                    action = line.replace("ACTION:", "").strip()
                    parsed["action_type"] = action
                elif line.startswith("PARAMETERS:"):
                    params_str = line.replace("PARAMETERS:", "").strip()
                    try:
                        # Try to parse as JSON
                        if params_str.startswith('{') and params_str.endswith('}'):
                            params = json.loads(params_str)
                            parsed["parameters"] = params
                        else:
                            # Simple key-value parsing
                            parsed["parameters"] = {}
                    except json.JSONDecodeError:
                        parsed["parameters"] = {}
            
            if "action_type" in parsed:
                return parsed
            
            # Fallback: try to extract action from response
            if "move_piece" in response.lower():
                return {"action_type": "move_piece", "parameters": {}}
            elif "check_solution" in response.lower():
                return {"action_type": "check_solution", "parameters": {}}
            elif "rotate" in response.lower():
                return {"action_type": "rotate_piece", "parameters": {}}
            
            return None
            
        except Exception as e:
            print(f"Error parsing decision: {e}")
            return None


def create_simple_test_environment():
    """Create a simplified test environment without full PhyVPuzzle setup."""
    
    class MockEnvironment:
        def __init__(self):
            self.step_count = 0
            self.task_completed = False
            self.last_action_success = True
            
        def setup_environment(self):
            print("Mock environment setup complete")
            
        def get_observation_for_vlm(self):
            # Create a simple test image
            img = Image.new('RGB', (512, 512), color='lightblue')
            
            state_desc = f"""Puzzle Type: luban_lock
Current State: initial
Step: {self.step_count}
Task Completed: {self.task_completed}
Last Action Success: {self.last_action_success}

Piece Positions:
- obj_1: (0.100, 0.200, 0.500)
- obj_2: (0.300, 0.200, 0.500)
- obj_3: (0.500, 0.200, 0.500)"""
            
            return {
                "image": img,
                "state_description": state_desc,
                "raw_state": {
                    "step_count": self.step_count,
                    "task_completed": self.task_completed,
                    "pieces": {"obj_1": {}, "obj_2": {}, "obj_3": {}}
                },
                "available_actions": ["move_piece", "rotate_piece", "check_solution"]
            }
        
        def execute_command(self, command):
            self.step_count += 1
            # Mock execution
            if command.get("command_type") == "check_solution" and self.step_count >= 3:
                self.task_completed = True
                return True
            return True
            
        def is_task_complete(self):
            return self.task_completed or self.step_count >= 5
            
        def get_success_status(self):
            return self.task_completed, "Task completed" if self.task_completed else "In progress"
            
        def close(self):
            print("Mock environment closed")
    
    return MockEnvironment()


def test_gpt4o_api_connection():
    """Test basic GPT-4o API connection."""
    print("Testing GPT-4o API connection...")
    
    api_key = os.getenv("api_key")
    base_url = os.getenv("base_url")
    
    if not api_key or not base_url:
        print("✗ Missing API credentials in .env file")
        return False
    
    try:
        processor = GPT4oVLMProcessor(api_key, base_url)
        
        # Create a simple test image
        test_img = Image.new('RGB', (256, 256), color='red')
        
        # Simple test input
        test_input = {
            "image": test_img,
            "instruction": "Describe what you see in this image.",
            "current_state": "Test state",
            "available_actions": ["test_action"],
            "step_number": 1
        }
        
        response = processor.process_input(test_input)
        
        if "error" in response.lower():
            print(f"✗ API test failed: {response}")
            return False
        
        print("✓ GPT-4o API connection successful")
        print(f"Sample response: {response[:100]}...")
        return True
        
    except Exception as e:
        print(f"✗ API connection failed: {e}")
        return False


def run_mock_puzzle_test():
    """Run a complete mock puzzle test with GPT-4o."""
    print("\nRunning mock puzzle test with GPT-4o...")
    
    # Setup
    api_key = os.getenv("api_key")
    base_url = os.getenv("base_url")
    
    processor = GPT4oVLMProcessor(api_key, base_url)
    parser = GPT4oDecisionParser()
    env = create_simple_test_environment()
    
    # Setup environment
    env.setup_environment()
    
    # Instruction template
    instruction = """You are controlling a robotic system to solve a Luban lock puzzle. 
The goal is to disassemble or assemble the interlocking wooden pieces by moving them in the correct sequence.

Available actions:
- move_piece: Move a piece to a target position
- rotate_piece: Rotate a piece around an axis  
- check_solution: Check if the puzzle is solved

Observe the current state carefully and plan your next action."""

    max_steps = 5
    step_results = []
    
    for step in range(max_steps):
        if env.is_task_complete():
            break
            
        print(f"\n--- Step {step + 1} ---")
        
        try:
            # Get observation
            observation = env.get_observation_for_vlm()
            
            # Prepare VLM input
            vlm_input = {
                "image": observation["image"],
                "instruction": instruction,
                "current_state": observation["state_description"],
                "available_actions": observation["available_actions"],
                "step_number": step + 1
            }
            
            # Add context from previous steps
            if step_results:
                recent_actions = step_results[-2:]  # Last 2 actions
                context = "Recent actions:\\n"
                for i, result in enumerate(recent_actions):
                    context += f"Step {len(step_results) - len(recent_actions) + i + 1}: {result['action']} - {result['status']}\\n"
                vlm_input["recent_context"] = context
            
            # Get VLM response
            print("Querying GPT-4o...")
            response = processor.process_input(vlm_input)
            print(f"GPT-4o Response: {response}")
            
            # Parse decision
            parsed_action = parser.parse_decision(response)
            print(f"Parsed Action: {parsed_action}")
            
            # Execute action
            if parsed_action:
                command = {
                    "command_type": parsed_action["action_type"],
                    "parameters": parsed_action.get("parameters", {})
                }
                success = env.execute_command(command)
                status = "Success" if success else "Failed"
            else:
                status = "Parse Failed"
                command = None
            
            print(f"Execution: {status}")
            
            # Record step
            step_results.append({
                "step": step + 1,
                "response": response,
                "action": parsed_action["action_type"] if parsed_action else "unknown",
                "command": command,
                "status": status
            })
            
            # Brief pause
            time.sleep(1)
            
        except Exception as e:
            print(f"Error in step {step + 1}: {e}")
            break
    
    # Final results
    success, reason = env.get_success_status()
    
    print(f"\n{'='*50}")
    print("FINAL RESULTS")
    print(f"{'='*50}")
    print(f"Total Steps: {len(step_results)}")
    print(f"Task Success: {success}")
    print(f"Reason: {reason}")
    
    print(f"\nStep Summary:")
    for result in step_results:
        print(f"  Step {result['step']}: {result['action']} - {result['status']}")
    
    env.close()
    return success


def main():
    """Run the complete GPT-4o PhyVPuzzle test."""
    print("GPT-4o PhyVPuzzle Integration Test")
    print("=" * 50)
    
    # Test 1: API Connection
    if not test_gpt4o_api_connection():
        print("Aborting: GPT-4o API connection failed")
        return 1
    
    # Test 2: Mock Puzzle Solving
    try:
        success = run_mock_puzzle_test()
        
        if success:
            print("\n🎉 Test completed successfully!")
            print("GPT-4o successfully solved the mock puzzle.")
        else:
            print("\n⚠️ Test completed but puzzle not solved.")
            print("GPT-4o made valid attempts but didn't complete the task.")
        
        print("\nNext steps:")
        print("1. Install PyBullet for full physics simulation")
        print("2. Run with real PhyVPuzzle environment")
        print("3. Tune GPT-4o prompts for better performance")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())