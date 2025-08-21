"""
LLM-as-judge evaluation system for PhyVPuzzle.
"""

import json
from typing import Tuple, List, Dict, Any
import openai
from tenacity import retry, stop_after_attempt, wait_exponential
from PIL import Image
import base64
from io import BytesIO

from ..core.base import BaseJudge


class LLMJudge(BaseJudge):
    """LLM-based judge for evaluating task completion."""
    
    def __init__(self, model_name: str, config: Dict[str, Any]):
        super().__init__(model_name, config)
        
        # Initialize OpenAI client
        self.client = openai.OpenAI(
            api_key=config.get("api_key"),
            base_url=config.get("base_url", "https://api.openai.com/v1")
        )
        
        self.temperature = config.get("temperature", 0.1)
        self.max_tokens = config.get("max_tokens", 200)
        self.timeout = config.get("timeout", 30.0)
        
    def judge_success(self, final_image: Image.Image, task_description: str, 
                     trajectory: List[str]) -> Tuple[bool, float, str]:
        """
        Judge if task was completed successfully.
        
        Args:
            final_image: Final state image
            task_description: Description of the task
            trajectory: List of action descriptions
            
        Returns:
            Tuple of (success, confidence_score, reasoning)
        """
        # Prepare the evaluation prompt
        evaluation_prompt = self._create_evaluation_prompt(task_description, trajectory)
        
        # Get judgment from LLM
        try:
            response = self._get_judge_response(final_image, evaluation_prompt)
            return self._parse_judge_response(response)
        except Exception as e:
            print(f"Judge evaluation failed: {e}")
            return False, 0.0, f"Evaluation failed: {e}"
            
    def _create_evaluation_prompt(self, task_description: str, trajectory: List[str]) -> str:
        """Create evaluation prompt for the judge."""
        trajectory_text = "\n".join(trajectory) if trajectory else "No actions recorded"
        
        prompt = f"""You are an expert evaluator for physics puzzle tasks. Your job is to determine if the task was completed successfully based on the final state image.

TASK DESCRIPTION:
{task_description}

ACTION TRAJECTORY:
{trajectory_text}

Please analyze the final state image and determine if the puzzle task was completed successfully. 

For your evaluation, consider:
1. Whether the puzzle objective has been achieved (e.g., dominoes fallen, pieces assembled, etc.)
2. The stability and correctness of the final configuration
3. Whether the result matches the expected outcome for this type of puzzle

Respond with a JSON object containing:
- "success": boolean (true if task completed successfully, false otherwise)
- "confidence": float between 0.0 and 1.0 (how confident you are in your judgment)  
- "reasoning": string (detailed explanation of your decision)

Example response format:
{{"success": true, "confidence": 0.95, "reasoning": "All dominoes have fallen in the correct sequence, forming a complete chain reaction."}}

Your response:"""
        
        return prompt
        
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=8),
        reraise=True
    )
    def _get_judge_response(self, image: Image.Image, prompt: str) -> str:
        """Get response from judge LLM."""
        # Convert image to base64
        image_data = self._image_to_base64(image)
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_data}"}
                    }
                ]
            }
        ]
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout=self.timeout
        )
        
        return response.choices[0].message.content
        
    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string."""
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return img_str
        
    def _parse_judge_response(self, response: str) -> Tuple[bool, float, str]:
        """Parse the judge's response into structured format."""
        try:
            # Try to extract JSON from response
            response = response.strip()
            
            # Handle potential markdown formatting
            if "```json" in response:
                json_start = response.find("```json") + 7
                json_end = response.find("```", json_start)
                response = response[json_start:json_end].strip()
            elif "```" in response:
                json_start = response.find("```") + 3
                json_end = response.find("```", json_start)
                response = response[json_start:json_end].strip()
                
            # Parse JSON
            judgment = json.loads(response)
            
            success = judgment.get("success", False)
            confidence = max(0.0, min(1.0, judgment.get("confidence", 0.0)))
            reasoning = judgment.get("reasoning", "No reasoning provided")
            
            return success, confidence, reasoning
            
        except (json.JSONDecodeError, KeyError) as e:
            # Fallback parsing for non-JSON responses
            response_lower = response.lower()
            
            if "success" in response_lower and "true" in response_lower:
                success = True
            elif "success" in response_lower and "false" in response_lower:
                success = False
            else:
                # Try to infer from keywords
                positive_keywords = ["completed", "successful", "solved", "correct", "achieved"]
                negative_keywords = ["failed", "incomplete", "unsuccessful", "wrong", "not solved"]
                
                positive_count = sum(1 for word in positive_keywords if word in response_lower)
                negative_count = sum(1 for word in negative_keywords if word in response_lower)
                
                success = positive_count > negative_count
                
            # Try to extract confidence if mentioned
            confidence = 0.5  # Default moderate confidence
            
            import re
            confidence_match = re.search(r'confidence[^\d]*(\d+(?:\.\d+)?)', response_lower)
            if confidence_match:
                try:
                    conf_value = float(confidence_match.group(1))
                    if conf_value <= 1.0:
                        confidence = conf_value
                    elif conf_value <= 100:
                        confidence = conf_value / 100.0
                except ValueError:
                    pass
                    
            reasoning = f"Parsed from non-JSON response: {response[:200]}..."
            
            return success, confidence, reasoning
            
    def judge_multiple_tasks(self, task_data: List[Dict[str, Any]]) -> List[Tuple[bool, float, str]]:
        """Judge multiple tasks in batch."""
        results = []
        
        for task in task_data:
            final_image = task.get("final_image")
            task_description = task.get("task_description", "")
            trajectory = task.get("trajectory", [])
            
            if final_image:
                result = self.judge_success(final_image, task_description, trajectory)
                results.append(result)
            else:
                results.append((False, 0.0, "No final image provided"))
                
        return results
        
    def get_judge_statistics(self, judgments: List[Tuple[bool, float, str]]) -> Dict[str, Any]:
        """Get statistics about judge evaluations."""
        if not judgments:
            return {}
            
        success_count = sum(1 for success, _, _ in judgments if success)
        avg_confidence = sum(conf for _, conf, _ in judgments) / len(judgments)
        
        confidence_by_success = {
            "successful_tasks": [conf for success, conf, _ in judgments if success],
            "failed_tasks": [conf for success, conf, _ in judgments if not success]
        }
        
        return {
            "total_judgments": len(judgments),
            "success_rate": success_count / len(judgments),
            "average_confidence": avg_confidence,
            "confidence_distribution": {
                "successful_avg": sum(confidence_by_success["successful_tasks"]) / len(confidence_by_success["successful_tasks"]) if confidence_by_success["successful_tasks"] else 0,
                "failed_avg": sum(confidence_by_success["failed_tasks"]) / len(confidence_by_success["failed_tasks"]) if confidence_by_success["failed_tasks"] else 0
            }
        }
