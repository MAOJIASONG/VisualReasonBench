"""
Base agent implementation with common functionality.
"""

import json
import time
from typing import Any, Dict, List, Optional, Tuple
from abc import abstractmethod

from ..core.base import BaseAgent, Observation
from ..utils.token_counter import count_tokens


class VLMAgent(BaseAgent):
    """Base VLM agent with common functionality."""
    
    def __init__(self, model_name: str, config: Dict[str, Any]):
        super().__init__(model_name, config)
        self.conversation_history = []
        self.system_prompt = ""
        
    def reset_conversation(self) -> None:
        """Reset conversation history."""
        self.conversation_history = []
        self.system_prompt = ""
        
    def set_system_prompt(self, prompt: str) -> None:
        """Set system prompt for the conversation."""
        self.system_prompt = prompt
        
    def process_observation(self, observation: Observation, system_prompt: str, 
                          user_prompt: str) -> Tuple[str, List[Dict[str, Any]]]:
        """Process observation and return response and tool calls."""
        # Update system prompt if different
        if system_prompt != self.system_prompt:
            self.set_system_prompt(system_prompt)
            
        # Prepare messages
        messages = self._prepare_messages(observation, user_prompt)
        
        # Get response from model
        response, tool_calls = self._get_model_response(messages)
        
        # Update conversation history
        self.conversation_history.append({
            "role": "user",
            "content": user_prompt,
            "observation": observation.to_dict(),
            "timestamp": time.time()
        })
        
        self.conversation_history.append({
            "role": "assistant", 
            "content": response,
            "tool_calls": tool_calls,
            "timestamp": time.time()
        })
        
        # Update token count
        self.total_tokens += self._count_tokens_in_messages(messages, response)
        
        return response, tool_calls
    
    def _prepare_messages(self, observation: Observation, user_prompt: str) -> List[Dict[str, Any]]:
        """Prepare message list for model API call."""
        messages = []
        
        # Add system message
        if self.system_prompt:
            messages.append({
                "role": "system",
                "content": self.system_prompt
            })
        
        # Add conversation history (text only)
        for msg in self.conversation_history:
            if msg["role"] in ["user", "assistant"]:
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
                
        # Add current observation with image
        content = [
            {"type": "text", "text": user_prompt}
        ]
        
        # Add main image
        if observation.image:
            content.append({
                "type": "image_url",
                "image_url": {"url": self._image_to_data_url(observation.image)}
            })
            
        # Add multi-view images if available
        if observation.multi_view_images:
            for view_name, img in observation.multi_view_images.items():
                content.append({
                    "type": "image_url", 
                    "image_url": {"url": self._image_to_data_url(img)}
                })
        
        messages.append({
            "role": "user",
            "content": content
        })
        
        return messages
    
    def _image_to_data_url(self, image) -> str:
        """Convert PIL Image to data URL."""
        import base64
        from io import BytesIO
        
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/png;base64,{img_str}"
    
    @abstractmethod
    def _get_model_response(self, messages: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
        """Get response from the specific model implementation."""
        pass
    
    def _count_tokens_in_messages(self, messages: List[Dict[str, Any]], response: str) -> int:
        """Count tokens in messages and response."""
        # Simple approximation - in real implementation would use model-specific tokenizer
        total_text = ""
        
        for msg in messages:
            if isinstance(msg["content"], str):
                total_text += msg["content"]
            elif isinstance(msg["content"], list):
                for item in msg["content"]:
                    if item["type"] == "text":
                        total_text += item["text"]
        
        total_text += response
        
        return count_tokens(total_text, self.model_name)
    
    def get_token_count(self) -> int:
        """Get total tokens used by this agent."""
        return self.total_tokens
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get summary of conversation for logging."""
        return {
            "total_turns": len(self.conversation_history) // 2,
            "total_tokens": self.total_tokens,
            "model_name": self.model_name,
            "last_interaction": self.conversation_history[-1]["timestamp"] if self.conversation_history else None
        }
