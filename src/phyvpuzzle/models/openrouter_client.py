"""
OpenRouter API Client Module

This module provides integration with OpenRouter for model inference,
with automatic local API key loading and token tracking.
"""

import os
import json
import base64
import time
from typing import Dict, Any, Optional, List, Tuple
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image
from dotenv import load_dotenv


class OpenRouterClient:
    """Client for OpenRouter API with automatic key loading and token tracking."""
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        """
        Initialize OpenRouter client.
        
        Args:
            api_key: Optional API key. If not provided, will try to load from environment
            base_url: Optional base URL. Defaults to OpenRouter API endpoint
        """
        # Load environment variables
        load_dotenv()
        
        # Get API key from various sources
        self.api_key = api_key or self._load_api_key()
        if not self.api_key:
            raise ValueError("No API key found. Please set OPENROUTER_API_KEY in .env or environment")
        
        # Set base URL
        self.base_url = base_url or os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        self.base_url = self.base_url.rstrip('/')
        
        # Setup session
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": os.getenv("OPENROUTER_REFERER", "https://github.com/visualreasonbench"),
            "X-Title": os.getenv("OPENROUTER_TITLE", "VisualReasonBench")
        })
        
        # Token tracking
        self.total_tokens_used = 0
        self.tokens_per_request = []
        
    def _load_api_key(self) -> Optional[str]:
        """
        Load API key from various sources in priority order.
        
        Returns:
            API key string or None if not found
        """
        # Priority 1: OPENROUTER_API_KEY environment variable
        key = os.getenv("OPENROUTER_API_KEY")
        if key:
            return key
        
        # Priority 2: Generic API_KEY for compatibility
        key = os.getenv("API_KEY")
        if key:
            return key
        
        # Priority 3: Check .env file in project root
        env_file = Path(__file__).parent.parent.parent.parent / ".env"
        if env_file.exists():
            # Parse .env file manually for specific keys
            with open(env_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        if '=' in line:
                            k, v = line.split('=', 1)
                            k = k.strip()
                            v = v.strip().strip('"').strip("'")
                            if k in ["OPENROUTER_API_KEY", "api_key", "API_KEY"]:
                                return v
        
        # Priority 4: Check home directory for .openrouter file
        home_key_file = Path.home() / ".openrouter"
        if home_key_file.exists():
            with open(home_key_file, 'r') as f:
                return f.read().strip()
        
        return None
    
    def encode_image(self, image: Image.Image, max_size: int = 1024) -> str:
        """
        Encode PIL Image to base64.
        
        Args:
            image: PIL Image object
            max_size: Maximum dimension size (will resize if larger)
            
        Returns:
            Base64 encoded string
        """
        # Resize if needed
        if image.size[0] > max_size or image.size[1] > max_size:
            ratio = min(max_size / image.size[0], max_size / image.size[1])
            new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        # Convert to base64
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    def test_connection(self) -> Tuple[bool, str]:
        """
        Test API connection.
        
        Returns:
            Tuple of (success, message)
        """
        try:
            # Use a cheap model for testing
            payload = {
                "model": "openai/gpt-3.5-turbo",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 10
            }
            
            response = self.session.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                return True, "Connection successful"
            else:
                return False, f"HTTP {response.status_code}: {response.text[:200]}"
                
        except Exception as e:
            return False, f"Connection failed: {str(e)}"
    
    def chat_completion(self, 
                       messages: List[Dict[str, Any]], 
                       model: str = "openai/gpt-4o",
                       max_tokens: int = 1000,
                       temperature: float = 0.7,
                       track_tokens: bool = True) -> Dict[str, Any]:
        """
        Send chat completion request to OpenRouter.
        
        Args:
            messages: List of message dictionaries
            model: Model to use
            max_tokens: Maximum tokens in response
            temperature: Temperature for sampling
            track_tokens: Whether to track token usage
            
        Returns:
            Response dictionary with 'content', 'tokens_used', and 'raw_response'
        """
        try:
            payload = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature
            }
            
            response = self.session.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Extract content
                content = ""
                if "choices" in result and result["choices"]:
                    content = result["choices"][0]["message"]["content"]
                
                # Track tokens if available
                tokens_used = 0
                if track_tokens and "usage" in result:
                    tokens_used = result["usage"].get("total_tokens", 0)
                    self.total_tokens_used += tokens_used
                    self.tokens_per_request.append(tokens_used)
                
                return {
                    "content": content,
                    "tokens_used": tokens_used,
                    "raw_response": result,
                    "success": True
                }
            else:
                return {
                    "content": f"Error: HTTP {response.status_code}",
                    "tokens_used": 0,
                    "raw_response": {"error": response.text},
                    "success": False
                }
                
        except Exception as e:
            return {
                "content": f"Error: {str(e)}",
                "tokens_used": 0,
                "raw_response": {"error": str(e)},
                "success": False
            }
    
    def process_vision_task(self,
                           image: Image.Image,
                           prompt: str,
                           model: str = "openai/gpt-4o",
                           max_tokens: int = 500,
                           system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a vision task with an image and prompt.
        
        Args:
            image: PIL Image object
            prompt: User prompt
            model: Model to use (must support vision)
            max_tokens: Maximum tokens in response
            system_prompt: Optional system prompt
            
        Returns:
            Response dictionary with processed content and token usage
        """
        # Encode image
        image_base64 = self.encode_image(image)
        
        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_base64}"}
                }
            ]
        })
        
        # Send request
        return self.chat_completion(messages, model, max_tokens)
    
    def get_token_statistics(self) -> Dict[str, Any]:
        """
        Get token usage statistics.
        
        Returns:
            Dictionary with token statistics
        """
        if not self.tokens_per_request:
            return {
                "total_tokens": 0,
                "num_requests": 0,
                "avg_tokens_per_request": 0,
                "min_tokens": 0,
                "max_tokens": 0
            }
        
        return {
            "total_tokens": self.total_tokens_used,
            "num_requests": len(self.tokens_per_request),
            "avg_tokens_per_request": self.total_tokens_used / len(self.tokens_per_request),
            "min_tokens": min(self.tokens_per_request),
            "max_tokens": max(self.tokens_per_request)
        }
    
    def reset_token_tracking(self):
        """Reset token tracking counters."""
        self.total_tokens_used = 0
        self.tokens_per_request = []
    
    def list_available_models(self) -> List[str]:
        """
        List available models from OpenRouter.
        
        Returns:
            List of model names
        """
        try:
            response = self.session.get(
                "https://openrouter.ai/api/v1/models",
                timeout=10
            )
            
            if response.status_code == 200:
                models = response.json().get("data", [])
                return [model["id"] for model in models]
            else:
                return []
                
        except Exception:
            return []


class OpenRouterVisionProcessor:
    """Specialized processor for vision tasks using OpenRouter."""
    
    def __init__(self, client: Optional[OpenRouterClient] = None):
        """
        Initialize vision processor.
        
        Args:
            client: Optional OpenRouterClient instance (will create if not provided)
        """
        self.client = client or OpenRouterClient()
        self.task_history = []
        
    def process_puzzle_step(self,
                           image: Image.Image,
                           state_description: str,
                           step_number: int,
                           available_actions: List[str],
                           model: str = "openai/gpt-4o") -> Dict[str, Any]:
        """
        Process a puzzle solving step.
        
        Args:
            image: Current state image
            state_description: Text description of current state
            step_number: Current step number
            available_actions: List of available actions
            model: Model to use
            
        Returns:
            Dictionary with action decision and token usage
        """
        # Build prompt
        actions_str = "\n".join(f"- {action}" for action in available_actions)
        
        prompt = f"""You are solving a physical puzzle at step {step_number}.

Current State:
{state_description}

Available Actions:
{actions_str}

Analyze the image and current state, then choose the best action.

Respond in JSON format:
{{
    "action": "selected_action_name",
    "parameters": {{}},
    "reasoning": "brief explanation",
    "confidence": 0.0-1.0
}}"""

        system_prompt = "You are an expert puzzle solver with strong spatial reasoning skills. Always respond with valid JSON."
        
        # Process with vision model
        response = self.client.process_vision_task(
            image=image,
            prompt=prompt,
            model=model,
            max_tokens=300,
            system_prompt=system_prompt
        )
        
        # Parse response
        if response["success"]:
            try:
                # Try to parse JSON from response
                content = response["content"]
                # Find JSON in response (might have extra text)
                import re
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    action_data = json.loads(json_match.group())
                else:
                    # Fallback parsing
                    action_data = {
                        "action": "unknown",
                        "parameters": {},
                        "reasoning": content,
                        "confidence": 0.5
                    }
            except json.JSONDecodeError:
                action_data = {
                    "action": "parse_error",
                    "parameters": {},
                    "reasoning": response["content"],
                    "confidence": 0.0
                }
        else:
            action_data = {
                "action": "error",
                "parameters": {},
                "reasoning": response["content"],
                "confidence": 0.0
            }
        
        # Record in history
        self.task_history.append({
            "step": step_number,
            "action": action_data,
            "tokens_used": response["tokens_used"]
        })
        
        return {
            "action": action_data,
            "tokens_used": response["tokens_used"],
            "raw_response": response["content"]
        }
    
    def get_task_summary(self) -> Dict[str, Any]:
        """
        Get summary of the current task.
        
        Returns:
            Dictionary with task statistics
        """
        if not self.task_history:
            return {
                "total_steps": 0,
                "total_tokens": 0,
                "actions_taken": []
            }
        
        return {
            "total_steps": len(self.task_history),
            "total_tokens": sum(h["tokens_used"] for h in self.task_history),
            "actions_taken": [h["action"]["action"] for h in self.task_history],
            "avg_confidence": sum(h["action"].get("confidence", 0) for h in self.task_history) / len(self.task_history)
        }
    
    def reset_task(self):
        """Reset task history."""
        self.task_history = []


def create_openrouter_client() -> OpenRouterClient:
    """
    Convenience function to create OpenRouter client with automatic configuration.
    
    Returns:
        Configured OpenRouterClient instance
    """
    return OpenRouterClient()