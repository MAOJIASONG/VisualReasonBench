"""
Token counting utilities for different models.
"""

from typing import Optional

# Conditional tiktoken import
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    tiktoken = None


def count_tokens(text: str, model_name: str = "gpt-4") -> int:
    """
    Count tokens in text for a given model.
    
    Args:
        text: Text to count tokens for
        model_name: Model name to use for tokenization
        
    Returns:
        Number of tokens
    """
    if not TIKTOKEN_AVAILABLE:
        # Fallback to approximate count (4 chars per token)
        return len(text) // 4
        
    try:
        # Map model names to tiktoken encodings
        encoding_map = {
            "gpt-4": "cl100k_base",
            "gpt-4o": "cl100k_base", 
            "gpt-4-vision-preview": "cl100k_base",
            "gpt-3.5-turbo": "cl100k_base",
            "text-davinci-003": "p50k_base",
            "text-davinci-002": "p50k_base"
        }
        
        # Get encoding name
        encoding_name = encoding_map.get(model_name, "cl100k_base")
        
        # Get tokenizer
        tokenizer = tiktoken.get_encoding(encoding_name)
        
        # Count tokens
        tokens = tokenizer.encode(text)
        return len(tokens)
        
    except Exception as e:
        print(f"Error counting tokens for model {model_name}: {e}")
        # Fallback to approximate count (4 chars per token)
        return len(text) // 4


def estimate_image_tokens(width: int, height: int, model_name: str = "gpt-4o") -> int:
    """
    Estimate token count for image based on dimensions.
    
    Args:
        width: Image width in pixels
        height: Image height in pixels  
        model_name: Model name for token estimation
        
    Returns:
        Estimated token count for image
    """
    if "gpt-4" in model_name:
        # GPT-4 Vision token calculation
        # Images are resized to fit within 2048x2048, maintaining aspect ratio
        # Then divided into 512x512 tiles
        
        # Calculate scaled dimensions
        scale = min(2048 / width, 2048 / height, 1.0)
        scaled_width = int(width * scale)
        scaled_height = int(height * scale)
        
        # Calculate number of tiles
        tiles_x = (scaled_width + 511) // 512
        tiles_y = (scaled_height + 511) // 512
        total_tiles = tiles_x * tiles_y
        
        # Each tile costs 170 tokens, plus 85 base tokens
        return total_tiles * 170 + 85
    
    else:
        # For other models, use a simple approximation
        return (width * height) // 1000


def calculate_conversation_tokens(messages: list, model_name: str = "gpt-4") -> int:
    """
    Calculate total tokens in a conversation.
    
    Args:
        messages: List of message dictionaries
        model_name: Model name for tokenization
        
    Returns:
        Total token count
    """
    total_tokens = 0
    
    for message in messages:
        # Count role tokens
        total_tokens += 4  # Role and formatting tokens
        
        content = message.get("content", "")
        
        if isinstance(content, str):
            total_tokens += count_tokens(content, model_name)
        elif isinstance(content, list):
            for item in content:
                if item.get("type") == "text":
                    total_tokens += count_tokens(item.get("text", ""), model_name)
                elif item.get("type") == "image_url":
                    # Estimate image tokens (would need actual dimensions)
                    total_tokens += estimate_image_tokens(512, 512, model_name)
    
    # Add conversation formatting tokens
    total_tokens += 2  # Conversation start/end
    
    return total_tokens
