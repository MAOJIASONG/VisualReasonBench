"""
Agent implementations for VLM models.

This package contains different agent implementations that can process
visual observations and generate actions, including:
- OpenAI API-based agents
- VLLM-based agents  
- Transformers-based agents
- Local model agents
"""

from .base_agent import VLMAgent
from .openai_agent import OpenAIAgent
from .vllm_agent import VLLMAgent

__all__ = [
    "VLMAgent",
    "OpenAIAgent", 
    "VLLMAgent"
]
