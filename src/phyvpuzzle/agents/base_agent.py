"""
Base agent implementation with common functionality.
"""

import json
import time
from typing import Any, Dict, List, Optional, Tuple
from abc import abstractmethod

from phyvpuzzle.core import BaseAgent, AgentConfig
from phyvpuzzle.core.base import Observation
from phyvpuzzle.utils.token_counter import calculate_conversation_tokens


class VLMAgent(BaseAgent):
    """Base VLM agent with common functionality."""
    
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.system_prompt: str = None
        self.total_tokens: int = 0

    def set_system_prompt(self, system_prompt: str) -> None:
        """Set system prompt for this agent."""
        self.system_prompt = system_prompt
        
    def process_observation(self, history: List[Observation], prompt: str, tools: Optional[List[Dict[str, Any]]] = None) -> Tuple[str, List[Dict[str, Any]]]:
            
        """Process observation and return response and tool calls."""

        # Prepare messages
        messages = self._prepare_messages(history, prompt)
        
        # Get response from model
        response, tool_calls = self._get_model_response(messages, tools)

        messages.append({
            "role": "assistant",
            "content": response,
            "tool_calls": tool_calls
        })
        # Update token count
        self.total_tokens += self._count_tokens_in_messages(messages)
        
        return response, tool_calls

    def _count_tokens_in_messages(self, messages: List[Dict[str, Any]]) -> int:
        """Count tokens in messages and response."""
        
        return calculate_conversation_tokens(messages, self.config.model_name)
    
    def get_token_count(self) -> int:
        """Get total tokens used by this agent."""
        return self.total_tokens
    
    @abstractmethod
    def _prepare_messages(self, history: List[Observation], prompt: str) -> List[Dict[str, Any]]:
        """Prepare message list for model API call."""
        pass
    
    @abstractmethod
    def _get_model_response(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None) -> Tuple[str, List[Dict[str, Any]]]:
        """Get response from the specific model implementation."""
        pass

