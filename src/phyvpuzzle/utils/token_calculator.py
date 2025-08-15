"""
Token Calculator using Qwen Model

This module provides token calculation functionality using the Qwen2.5-72B model
for accurate token counting and cost estimation.
"""

from typing import Dict, List, Optional, Any, Union
import tiktoken
import json


class QwenTokenCalculator:
    """Calculate tokens using Qwen2.5-72B tokenizer."""
    
    def __init__(self):
        """Initialize the token calculator."""
        # Try to use tiktoken with cl100k_base encoding (similar to GPT-4)
        # For actual Qwen tokenizer, you would need the model's specific tokenizer
        try:
            self.encoding = tiktoken.get_encoding("cl100k_base")
        except:
            # Fallback to a simple approximation
            self.encoding = None
            
        # Token usage tracking
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_interactions = 0
            
        # Token pricing (example rates - adjust based on actual pricing)
        self.pricing = {
            "gpt-4o": {
                "input": 0.005,  # per 1K tokens
                "output": 0.015,  # per 1K tokens
            },
            "qwen2.5-72b": {
                "input": 0.002,  # per 1K tokens
                "output": 0.006,  # per 1K tokens
            },
            "default": {
                "input": 0.001,
                "output": 0.002,
            }
        }
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in a text string."""
        if self.encoding:
            return len(self.encoding.encode(text))
        else:
            # Simple approximation: ~4 characters per token
            return len(text) // 4
    
    def count_message_tokens(self, messages: List[Dict[str, Any]]) -> int:
        """Count tokens in a list of messages."""
        total_tokens = 0
        
        for message in messages:
            # Count role tokens
            total_tokens += self.count_tokens(message.get("role", ""))
            
            # Count content tokens
            content = message.get("content", "")
            if isinstance(content, str):
                total_tokens += self.count_tokens(content)
            elif isinstance(content, list):
                # Handle multi-modal content (text + images)
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            total_tokens += self.count_tokens(item.get("text", ""))
                        elif item.get("type") == "image_url":
                            # Approximate tokens for images
                            # High-res image: ~765 tokens, Low-res: ~85 tokens
                            detail = item.get("image_url", {}).get("detail", "auto")
                            if detail == "high":
                                total_tokens += 765
                            else:
                                total_tokens += 170  # Average
        
        # Add overhead for message structure
        total_tokens += 3 * len(messages)
        
        return total_tokens
    
    def count_tool_tokens(self, tools: List[Dict[str, Any]]) -> int:
        """Count tokens in tool definitions."""
        if not tools:
            return 0
        
        # Convert tools to JSON string and count
        tools_str = json.dumps(tools, separators=(',', ':'))
        return self.count_tokens(tools_str)
    
    def calculate_cost(self, 
                      input_tokens: int,
                      output_tokens: int,
                      model: str = "gpt-4o") -> float:
        """Calculate cost based on token usage."""
        pricing = self.pricing.get(model, self.pricing["default"])
        
        input_cost = (input_tokens / 1000) * pricing["input"]
        output_cost = (output_tokens / 1000) * pricing["output"]
        
        return input_cost + output_cost
    
    def accumulate_tokens(self, input_tokens: int, output_tokens: int) -> None:
        """Accumulate token counts for running totals."""
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_interactions += 1
    
    def get_total_usage(self) -> Dict[str, Any]:
        """Get accumulated token usage statistics."""
        total = self.total_input_tokens + self.total_output_tokens
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": total,
            "total_interactions": self.total_interactions,
            "avg_tokens_per_interaction": total / max(1, self.total_interactions),
            "input_output_ratio": self.total_input_tokens / max(1, self.total_output_tokens)
        }
    
    def reset_counters(self) -> None:
        """Reset accumulated token counters."""
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_interactions = 0
    
    def analyze_conversation(self, 
                            messages: List[Dict[str, Any]],
                            tools: Optional[List[Dict[str, Any]]] = None,
                            completions: Optional[List[str]] = None,
                            model: str = "gpt-4o") -> Dict[str, Any]:
        """Analyze a complete conversation for token usage and cost."""
        
        # Count input tokens
        input_tokens = self.count_message_tokens(messages)
        
        # Add tool tokens if present
        if tools:
            input_tokens += self.count_tool_tokens(tools)
        
        # Count output tokens
        output_tokens = 0
        if completions:
            for completion in completions:
                output_tokens += self.count_tokens(completion)
        
        # Calculate total and cost
        total_tokens = input_tokens + output_tokens
        cost = self.calculate_cost(input_tokens, output_tokens, model)
        
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "cost": cost,
            "cost_breakdown": {
                "input_cost": (input_tokens / 1000) * self.pricing.get(model, self.pricing["default"])["input"],
                "output_cost": (output_tokens / 1000) * self.pricing.get(model, self.pricing["default"])["output"],
            },
            "model": model,
            "metrics": {
                "avg_input_tokens_per_message": input_tokens / len(messages) if messages else 0,
                "avg_output_tokens_per_completion": output_tokens / len(completions) if completions else 0,
                "input_output_ratio": input_tokens / output_tokens if output_tokens > 0 else float('inf'),
            }
        }
    
    def estimate_pipeline_usage(self,
                               num_steps: int,
                               has_images: bool = True,
                               has_tools: bool = True,
                               model: str = "gpt-4o") -> Dict[str, Any]:
        """Estimate token usage for a pipeline run."""
        
        # Base estimates per step
        text_tokens_per_step = 200  # Task description + context
        image_tokens_per_step = 170 if has_images else 0  # Per image
        tool_tokens = 500 if has_tools else 0  # Tool definitions
        output_tokens_per_step = 150  # Average response
        
        # Calculate totals
        input_tokens = num_steps * (text_tokens_per_step + image_tokens_per_step + tool_tokens)
        output_tokens = num_steps * output_tokens_per_step
        total_tokens = input_tokens + output_tokens
        
        # Calculate cost
        cost = self.calculate_cost(input_tokens, output_tokens, model)
        
        return {
            "estimated_input_tokens": input_tokens,
            "estimated_output_tokens": output_tokens,
            "estimated_total_tokens": total_tokens,
            "estimated_cost": cost,
            "parameters": {
                "num_steps": num_steps,
                "has_images": has_images,
                "has_tools": has_tools,
                "model": model
            }
        }


def format_token_report(token_data: Dict[str, Any]) -> str:
    """Format token data into a readable report."""
    
    report = []
    report.append("\n" + "="*60)
    report.append("                 TOKEN USAGE REPORT")
    report.append("="*60)
    
    # Handle both actual and estimated token keys
    input_key = 'input_tokens' if 'input_tokens' in token_data else 'estimated_input_tokens'
    output_key = 'output_tokens' if 'output_tokens' in token_data else 'estimated_output_tokens'
    total_key = 'total_tokens' if 'total_tokens' in token_data else 'estimated_total_tokens'
    cost_key = 'cost' if 'cost' in token_data else 'estimated_cost'
    
    report.append(f"\n📊 Token Counts:")
    report.append(f"   Input Tokens:  {token_data.get(input_key, 0):,}")
    report.append(f"   Output Tokens: {token_data.get(output_key, 0):,}")
    report.append(f"   Total Tokens:  {token_data.get(total_key, 0):,}")
    
    if cost_key in token_data:
        report.append(f"\n💰 Cost Analysis:")
        report.append(f"   Total Cost: ${token_data[cost_key]:.4f}")
        if 'cost_breakdown' in token_data:
            breakdown = token_data['cost_breakdown']
            report.append(f"   Input Cost:  ${breakdown['input_cost']:.4f}")
            report.append(f"   Output Cost: ${breakdown['output_cost']:.4f}")
    
    if 'metrics' in token_data:
        metrics = token_data['metrics']
        report.append(f"\n📈 Efficiency Metrics:")
        report.append(f"   Avg Input/Message:  {metrics['avg_input_tokens_per_message']:.1f}")
        report.append(f"   Avg Output/Response: {metrics['avg_output_tokens_per_completion']:.1f}")
        report.append(f"   Input/Output Ratio:  {metrics['input_output_ratio']:.2f}")
    
    report.append("\n" + "="*60)
    
    return "\n".join(report)


# Example usage and testing
if __name__ == "__main__":
    calculator = QwenTokenCalculator()
    
    # Test basic token counting
    test_text = "Push the first domino to topple all five dominoes in sequence."
    tokens = calculator.count_tokens(test_text)
    print(f"Test text tokens: {tokens}")
    
    # Test message token counting
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [
            {"type": "text", "text": "What do you see in this image?"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,...", "detail": "auto"}}
        ]}
    ]
    msg_tokens = calculator.count_message_tokens(messages)
    print(f"Message tokens: {msg_tokens}")
    
    # Test pipeline estimation
    estimate = calculator.estimate_pipeline_usage(
        num_steps=10,
        has_images=True,
        has_tools=True,
        model="gpt-4o"
    )
    print(format_token_report(estimate))