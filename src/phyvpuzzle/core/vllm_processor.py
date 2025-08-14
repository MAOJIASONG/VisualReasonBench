"""
VLLM Processor Module

This module handles the interaction with Vision-Language Models (VLLMs) for 
physical visual reasoning tasks.
"""

from typing import Dict, List, Optional, Tuple, Any
from PIL import Image
import numpy as np
from abc import ABC, abstractmethod
import os
from pathlib import Path
import time


class VLLMProcessor(ABC):
    """Abstract base class for Vision-Language Model processors."""
    
    def __init__(self, model_name: str, device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.processor = None
        self.history = []
        
    @abstractmethod
    def load_model(self) -> None:
        """Load the VLLM model and processor."""
        pass
    
    @abstractmethod
    def process_input(self, image: Image.Image, task_description: str, 
                     context: Dict[str, Any]) -> str:
        """Process visual input and generate response."""
        pass
    
    def add_to_history(self, input_data: Dict[str, Any], 
                      response: str) -> None:
        """Add interaction to history."""
        self.history.append({
            'input': input_data,
            'response': response,
            'timestamp': time.time(),
            'step_index': len(self.history)
        })
    
    def get_history(self) -> List[Dict[str, Any]]:
        """Get interaction history."""
        return self.history
    
    def clear_history(self) -> None:
        """Clear interaction history."""
        self.history = []


class OpenAIVLLMProcessor(VLLMProcessor):
    """OpenAI-based VLLM processor with tool calling and configurable base_url."""
    
    def __init__(self, model_name: str = "gpt-4o", 
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None):
        super().__init__(model_name)
        self.api_key = api_key or self._load_api_key()
        self.base_url = base_url.rstrip('/') if base_url else None
        self.client = None
        
    def _load_api_key(self) -> Optional[str]:
        """Load API key from environment, project .env, or home files."""
        # Priority 1: OPENAI_API_KEY
        key = os.getenv("OPENAI_API_KEY")
        if key:
            return key
        # Priority 2: generic keys
        for env_name in ("api_key", "API_KEY"):
            key = os.getenv(env_name)
            if key:
                return key
        # Priority 3: project .env
        env_path = Path(__file__).parent.parent.parent.parent / ".env"
        if env_path.exists():
            try:
                from dotenv import load_dotenv
                load_dotenv(env_path)
                key = os.getenv("OPENAI_API_KEY") or os.getenv("api_key") or os.getenv("API_KEY")
                if key:
                    return key
            except Exception:
                pass
        # Priority 4: home config files
        for candidate in (Path.home()/".openai_key", Path.home()/".config"/"openai_key"):
            if candidate.exists():
                try:
                    return candidate.read_text().strip()
                except Exception:
                    continue
        return None
        
    def load_model(self) -> None:
        """Initialize OpenAI API client with optional base_url (supports OpenRouter-compatible endpoints)."""
        from openai import OpenAI
        if not self.api_key:
            # Allow client to still initialize; requests will fail with clear error later
            self.client = OpenAI()
        else:
            if self.base_url:
                self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            else:
                self.client = OpenAI(api_key=self.api_key)
    
    def process_input(self, image: Image.Image, task_description: str, 
                     context: Dict[str, Any],
                     tools: Optional[List[Dict[str, Any]]] = None,
                     tool_choice: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process input using OpenAI API (or OpenRouter-compatible) with optional tool calling.
        
        Returns a dictionary with keys: 'content', 'tool_calls', 'raw_response', 'used_messages'.
        """
        # Convert PIL Image to base64
        import base64
        import io
        
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        # Create prompt with history context
        history_context = self._format_history_context()
        
        messages: List[Dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    "You are an expert in physical visual reasoning. "
                    f"Task: {task_description}\n"
                    f"Context: {context}\n"
                    f"History: {history_context}\n"
                    "When appropriate, you may call available tools to act in the environment. "
                    "Otherwise, choose either 'finish' if the task is complete, or 'action' followed by a description of the next action."
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What should I do next?"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_str}"}},
                ],
            },
        ]
        
        kwargs: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": 300,
        }
        if tools:
            kwargs["tools"] = tools
            if tool_choice is not None:
                kwargs["tool_choice"] = tool_choice
        
        response = self.client.chat.completions.create(**kwargs)
        message = response.choices[0].message
        content = getattr(message, "content", None) or ""
        tool_calls = []
        # New OpenAI SDK exposes tool_calls as a list on message
        if getattr(message, "tool_calls", None):
            for tc in message.tool_calls:
                try:
                    tool_calls.append({
                        "id": getattr(tc, "id", None),
                        "type": getattr(tc, "type", None),
                        "function": {
                            "name": getattr(tc.function, "name", None),
                            "arguments": getattr(tc.function, "arguments", None),
                        },
                    })
                except Exception:
                    continue
        
        self.add_to_history(
            {
                'image': image,
                'task_description': task_description,
                'context': context,
            },
            content,
        )
        
        return {
            "content": content,
            "tool_calls": tool_calls,
            "raw_response": response,
            "used_messages": messages,
        }
    
    def _format_history_context(self) -> str:
        """Format history for context."""
        if not self.history:
            return "No previous actions."
        
        formatted_history = []
        for i, entry in enumerate(self.history[-5:]):  # Last 5 entries
            formatted_history.append(f"Step {i+1}: {entry['response']}")
        
        return " | ".join(formatted_history)


class HuggingFaceVLLMProcessor(VLLMProcessor):
    """HuggingFace-based VLLM processor."""
    
    def __init__(self, model_name: str = "microsoft/kosmos-2-patch14-224", 
                 device: str = "cuda"):
        super().__init__(model_name, device)
        
    def load_model(self) -> None:
        """Load HuggingFace model and processor."""
        # Lazy imports to avoid hard dependency when not used
        from transformers import AutoProcessor, AutoModelForVision2Seq
        import torch  # type: ignore
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = AutoModelForVision2Seq.from_pretrained(self.model_name)
        self.model.to(self.device)
        
    def process_input(self, image: Image.Image, task_description: str, 
                     context: Dict[str, Any]) -> str:
        """Process input using HuggingFace model."""
        import torch  # type: ignore
        # Create prompt with history context
        history_context = self._format_history_context()
        
        prompt = (f"Task: {task_description}\n"
                 f"Context: {context}\n"
                 f"History: {history_context}\n"
                 f"Choose either 'finish' if the task is complete, "
                 f"or 'action' followed by a description of the next action.")
        
        inputs = self.processor(text=prompt, images=image, 
                              return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=300)
        
        result = self.processor.decode(outputs[0], skip_special_tokens=True)
        
        self.add_to_history({
            'image': image,
            'task_description': task_description,
            'context': context
        }, result)
        
        return result
    
    def _format_history_context(self) -> str:
        """Format history for context."""
        if not self.history:
            return "No previous actions."
        
        formatted_history = []
        for i, entry in enumerate(self.history[-5:]):  # Last 5 entries
            formatted_history.append(f"Step {i+1}: {entry['response']}")
        
        return " | ".join(formatted_history)


class DecisionParser:
    """Parse VLLM decisions into action types."""
    
    @staticmethod
    def parse_decision(response: str) -> Tuple[str, str]:
        """
        Parse VLLM response into decision type and description.
        
        Args:
            response: VLLM response string
            
        Returns:
            Tuple of (decision_type, description)
            decision_type: 'finish' or 'action'
            description: Action description if decision_type is 'action'
        """
        response = response.strip().lower()
        
        if 'finish' in response:
            return 'finish', response
        elif 'action' in response:
            # Extract action description
            action_start = response.find('action') + len('action')
            action_desc = response[action_start:].strip()
            if action_desc.startswith(':'):
                action_desc = action_desc[1:].strip()
            return 'action', action_desc
        else:
            # Default to action if unclear
            return 'action', response


def create_vllm_processor(processor_type: str = "openai", 
                         **kwargs) -> VLLMProcessor:
    """
    Factory function to create VLLM processor.
    
    Args:
        processor_type: Type of processor ("openai" or "huggingface")
        **kwargs: Additional arguments for processor
        
    Returns:
        VLLMProcessor instance
    """
    if processor_type == "openai":
        return OpenAIVLLMProcessor(**kwargs)
    elif processor_type == "huggingface":
        return HuggingFaceVLLMProcessor(**kwargs)
    else:
        raise ValueError(f"Unknown processor type: {processor_type}") 