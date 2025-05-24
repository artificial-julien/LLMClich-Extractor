import json
import re
from typing import Dict, Any, List, Optional, Literal

def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract JSON from text using regex pattern matching.
    
    Args:
        text: Text that may contain JSON
        
    Returns:
        Extracted JSON as dictionary or None if not found
    """
    # Look for JSON-like structure in the text
    json_pattern = r'\{[^{}]*\}'
    matches = re.findall(json_pattern, text)
    
    for match in matches:
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue
            
    return None

def generate_constrained_prompt(
    prompt: str, 
    possible_answers: List[str],
    answer_format: Literal["enum", "numbered"] = "numbered"
) -> str:
    """
    Generate a prompt that includes constraints for the model.
    
    Args:
        prompt: Original prompt
        possible_answers: List of possible answers
        answer_format: Format for the answer - "enum" for direct answer selection or "numbered" for numbered selection
        
    Returns:
        Enhanced prompt with constraints
    """
    if answer_format == "enum":
        # Create a clean list of answers without numbers
        answers_list = "\n".join(f"- {answer}" for answer in possible_answers)
        
        enhanced_prompt = f"""{prompt}

{answers_list}

Your response must be a valid JSON object with this exact format:
{{"answer": "<chosen_answer>"}}

Where <chosen_answer> must be one of the exact answers listed above."""
    else:  # numbered format
        numbered_answers = "\n".join(f"{i+1}. {answer}" for i, answer in enumerate(possible_answers))
        
        enhanced_prompt = f"""{prompt}

{numbered_answers}."""
    
    return enhanced_prompt 