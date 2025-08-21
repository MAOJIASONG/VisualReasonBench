"""
OpenAI API-based agent implementation.
"""

import json
from typing import Any, Dict, List, Tuple
import openai
from tenacity import retry, stop_after_attempt, wait_exponential

from .base_agent import VLMAgent


class OpenAIAgent(VLMAgent):
    """Agent using OpenAI API (or compatible APIs)."""
    
    def __init__(self, model_name: str, config: Dict[str, Any]):
        super().__init__(model_name, config)
        
        # Initialize OpenAI client
        self.client = openai.OpenAI(
            api_key=config.get("api_key"),
            base_url=config.get("base_url", "https://api.openai.com/v1")
        )
        
        self.temperature = config.get("temperature", 0.7)
        self.max_tokens = config.get("max_tokens", 500)
        self.timeout = config.get("timeout", 30.0)
        
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True
    )
    def _get_model_response(self, messages: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
        """Get response from OpenAI API."""
        try:
            # Check if we have tool schemas available
            tools = self._get_available_tools()
            
            # Make API call
            kwargs = {
                "model": self.model_name,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "timeout": self.timeout
            }
            
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"
                
            response = self.client.chat.completions.create(**kwargs)
            
            # Extract response content
            message = response.choices[0].message
            content = message.content or ""
            
            # Extract tool calls
            tool_calls = []
            if message.tool_calls:
                for tool_call in message.tool_calls:
                    tool_calls.append({
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments
                        }
                    })
            
            return content, tool_calls
            
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            return f"Error: {e}", []
    
    def _get_available_tools(self) -> List[Dict[str, Any]]:
        """Get available tool schemas from context (would be injected by runner)."""
        # In a full implementation, this would be provided by the environment
        # For now, return empty list - tools will be extracted from text response
        return []
    
    def parse_tool_calls_from_text(self, text: str) -> List[Dict[str, Any]]:
        """Parse tool calls from text response when formal tool calling isn't used."""
        tool_calls = []
        
        # Look for action patterns in text
        lines = text.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            
            # Pattern 1: Action: function_name(args)
            if line.startswith("Action:"):
                try:
                    action_part = line[7:].strip()  # Remove "Action: "
                    
                    # Extract function name and arguments
                    if '(' in action_part and ')' in action_part:
                        func_name = action_part.split('(')[0].strip()
                        args_part = action_part[action_part.find('(')+1:action_part.rfind(')')]
                        
                        # Simple argument parsing (would need more robust parsing)
                        args = {}
                        if args_part:
                            # Very basic parsing - in real implementation would use proper parser
                            try:
                                args = json.loads(f"{{{args_part}}}")
                            except:
                                # Fallback to simple key=value parsing
                                for arg in args_part.split(','):
                                    if '=' in arg:
                                        key, val = arg.split('=', 1)
                                        key = key.strip().strip("\"'")
                                        val = val.strip().strip("\"'")
                                        args[key] = val
                        
                        tool_calls.append({
                            "id": f"call_{len(tool_calls)}",
                            "type": "function",
                            "function": {
                                "name": func_name,
                                "arguments": json.dumps(args)
                            }
                        })
                except Exception as e:
                    print(f"Error parsing action from text: {e}")
                    continue
            
            # Pattern 2: {"action": "func_name", "parameters": {...}}
            elif line.startswith('{') and '"action"' in line:
                try:
                    action_data = json.loads(line)
                    if "action" in action_data:
                        tool_calls.append({
                            "id": f"call_{len(tool_calls)}",
                            "type": "function", 
                            "function": {
                                "name": action_data["action"],
                                "arguments": json.dumps(action_data.get("parameters", {}))
                            }
                        })
                except Exception as e:
                    print(f"Error parsing JSON action: {e}")
                    continue
        
        return tool_calls
