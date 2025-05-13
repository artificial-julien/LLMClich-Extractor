import json
import re
from typing import Dict, Any, List, Optional

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

def generate_constrained_prompt(prompt: str, possible_answers: List[str]) -> str:
    """
    Generate a prompt that includes constraints for the model.
    
    Args:
        prompt: Original prompt
        possible_answers: List of possible answers
        
    Returns:
        Enhanced prompt with constraints
    """
    numbered_answers = "\n".join(f"{i+1}. {answer}" for i, answer in enumerate(possible_answers))
    
    enhanced_prompt = f"""{prompt}

Please choose one of the following options by responding with a JSON object containing a single "answer" field with the number of your choice:

{numbered_answers}

Your response must be a valid JSON object with this exact format:
{{"answer": <number>}}

Where <number> is the number (1-{len(possible_answers)}) corresponding to your chosen answer."""
    
    return enhanced_prompt 