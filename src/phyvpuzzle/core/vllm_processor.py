"""
VLLM Processor Module

This module handles the interaction with Vision-Language Models (VLLMs) for 
physical visual reasoning tasks.
"""

from typing import Dict, List, Optional, Tuple, Any
import torch
from PIL import Image
import numpy as np
from transformers import AutoProcessor, AutoModelForVision2Seq
from abc import ABC, abstractmethod


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
            'timestamp': torch.tensor([len(self.history)])
        })
    
    def get_history(self) -> List[Dict[str, Any]]:
        """Get interaction history."""
        return self.history
    
    def clear_history(self) -> None:
        """Clear interaction history."""
        self.history = []


class OpenAIVLLMProcessor(VLLMProcessor):
    """OpenAI-based VLLM processor."""
    
    def __init__(self, model_name: str = "gpt-4-vision-preview", 
                 api_key: Optional[str] = None):
        super().__init__(model_name)
        self.api_key = api_key
        
    def load_model(self) -> None:
        """Load OpenAI API client."""
        import openai
        if self.api_key:
            openai.api_key = self.api_key
        self.client = openai.OpenAI()
    
    def process_input(self, image: Image.Image, task_description: str, 
                     context: Dict[str, Any]) -> str:
        """Process input using OpenAI API."""
        # Convert PIL Image to base64
        import base64
        import io
        
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        # Create prompt with history context
        history_context = self._format_history_context()
        
        messages = [
            {
                "role": "system",
                "content": f"You are an expert in physical visual reasoning. "
                          f"Task: {task_description}\n"
                          f"Context: {context}\n"
                          f"History: {history_context}\n"
                          f"Choose either 'finish' if the task is complete, "
                          f"or 'action' followed by a description of the next action."
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What should I do next?"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_str}"}}
                ]
            }
        ]
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=300
        )
        
        result = response.choices[0].message.content
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


class HuggingFaceVLLMProcessor(VLLMProcessor):
    """HuggingFace-based VLLM processor."""
    
    def __init__(self, model_name: str = "microsoft/kosmos-2-patch14-224", 
                 device: str = "cuda"):
        super().__init__(model_name, device)
        
    def load_model(self) -> None:
        """Load HuggingFace model and processor."""
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = AutoModelForVision2Seq.from_pretrained(self.model_name)
        self.model.to(self.device)
        
    def process_input(self, image: Image.Image, task_description: str, 
                     context: Dict[str, Any]) -> str:
        """Process input using HuggingFace model."""
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