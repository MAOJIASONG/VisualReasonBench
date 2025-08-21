"""
VLLM-based agent implementation for local model inference.
"""

import json
from typing import Any, Dict, List, Tuple, Optional
import requests
import base64
from io import BytesIO

from .base_agent import VLMAgent


class VLLMAgent(VLMAgent):
    """Agent using VLLM for local model inference."""
    
    def __init__(self, model_name: str, config: Dict[str, Any]):
        super().__init__(model_name, config)
        
        self.vllm_url = config.get("vllm_url", "http://localhost:8000")
        self.temperature = config.get("temperature", 0.7)
        self.max_tokens = config.get("max_tokens", 500)
        self.timeout = config.get("timeout", 60.0)
        
        # Check if VLLM server is running
        self._check_vllm_server()
        
    def _check_vllm_server(self) -> None:
        """Check if VLLM server is accessible."""
        try:
            response = requests.get(f"{self.vllm_url}/health", timeout=5)
            if response.status_code != 200:
                raise ConnectionError("VLLM server health check failed")
        except Exception as e:
            print(f"Warning: Cannot connect to VLLM server at {self.vllm_url}: {e}")
            print("Make sure VLLM server is running with vision model loaded")
    
    def _get_model_response(self, messages: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
        """Get response from VLLM server."""
        try:
            # Convert messages to VLLM format
            prompt = self._messages_to_prompt(messages)
            
            # Prepare request payload
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "stop": ["Human:", "Assistant:", "<|end|>", "<|im_end|>"]
            }
            
            # Make request to VLLM server
            response = requests.post(
                f"{self.vllm_url}/v1/completions",
                json=payload,
                timeout=self.timeout,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code != 200:
                raise Exception(f"VLLM API error: {response.status_code} - {response.text}")
                
            result = response.json()
            
            # Extract text response
            text_response = result["choices"][0]["text"].strip()
            
            # Parse tool calls from text (VLLM doesn't have native function calling)
            tool_calls = self._extract_tool_calls_from_text(text_response)
            
            return text_response, tool_calls
            
        except Exception as e:
            print(f"Error calling VLLM API: {e}")
            return f"Error: {e}", []
    
    def _messages_to_prompt(self, messages: List[Dict[str, Any]]) -> str:
        """Convert messages to prompt format for VLLM."""
        prompt_parts = []
        
        for message in messages:
            role = message["role"]
            content = message["content"]
            
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                # Handle mixed content (text + images)
                if isinstance(content, list):
                    text_parts = []
                    for item in content:
                        if item["type"] == "text":
                            text_parts.append(item["text"])
                        elif item["type"] == "image_url":
                            # For VLLM with vision, we'd need to handle images differently
                            # This is a simplified placeholder
                            text_parts.append("[IMAGE]")
                    
                    user_content = "\n".join(text_parts)
                else:
                    user_content = content
                    
                prompt_parts.append(f"Human: {user_content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        
        # Add final assistant prompt
        prompt_parts.append("Assistant:")
        
        return "\n\n".join(prompt_parts)
    
    def _extract_tool_calls_from_text(self, text: str) -> List[Dict[str, Any]]:
        """Extract tool calls from model text response."""
        tool_calls = []
        
        # Look for function call patterns
        lines = text.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            
            # Pattern 1: Function call format
            if line.startswith("Function:") or line.startswith("Tool:") or line.startswith("Action:"):
                try:
                    # Extract function name and arguments
                    parts = line.split(":", 1)
                    if len(parts) == 2:
                        func_part = parts[1].strip()
                        
                        # Look for function_name(arguments) format
                        if '(' in func_part and ')' in func_part:
                            func_name = func_part.split('(')[0].strip()
                            args_str = func_part[func_part.find('(')+1:func_part.rfind(')')].strip()
                            
                            # Parse arguments
                            args = self._parse_function_args(args_str)
                            
                            tool_calls.append({
                                "id": f"call_{len(tool_calls)}",
                                "type": "function",
                                "function": {
                                    "name": func_name,
                                    "arguments": json.dumps(args)
                                }
                            })
                            
                except Exception as e:
                    print(f"Error parsing function call: {e}")
                    continue
            
            # Pattern 2: JSON format
            elif line.startswith('{') and ('"function"' in line or '"action"' in line):
                try:
                    call_data = json.loads(line)
                    
                    func_name = call_data.get("function") or call_data.get("action")
                    args = call_data.get("arguments") or call_data.get("parameters", {})
                    
                    if func_name:
                        tool_calls.append({
                            "id": f"call_{len(tool_calls)}",
                            "type": "function",
                            "function": {
                                "name": func_name,
                                "arguments": json.dumps(args) if isinstance(args, dict) else args
                            }
                        })
                        
                except Exception as e:
                    print(f"Error parsing JSON function call: {e}")
                    continue
        
        return tool_calls
    
    def _parse_function_args(self, args_str: str) -> Dict[str, Any]:
        """Parse function arguments from string."""
        args = {}
        
        if not args_str.strip():
            return args
        
        try:
            # Try JSON parsing first
            if args_str.startswith('{') and args_str.endswith('}'):
                return json.loads(args_str)
            
            # Try simple key=value parsing
            for arg in args_str.split(','):
                if '=' in arg:
                    key, val = arg.split('=', 1)
                    key = key.strip().strip("\"'")
                    val = val.strip().strip("\"'")
                    
                    # Try to convert to appropriate type
                    try:
                        # Try number conversion
                        if '.' in val:
                            args[key] = float(val)
                        else:
                            args[key] = int(val)
                    except ValueError:
                        # Keep as string
                        args[key] = val
                        
        except Exception as e:
            print(f"Error parsing function arguments '{args_str}': {e}")
            
        return args
    
    def set_vision_support(self, enabled: bool = True) -> None:
        """Configure vision support for VLLM."""
        # This would be used to configure vision model behavior
        self.vision_enabled = enabled
