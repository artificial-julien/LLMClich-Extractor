from typing import List, Dict, Any

def format_template_variables(template: str, variables: Dict[str, Any]) -> str:
    """
    Format a template string by replacing variables in [variable_name] format.
    
    Args:
        template: Template string with [variable_name] placeholders
        variables: Dictionary of variable names and values
        
    Returns:
        Formatted string with variables replaced
    """
    formatted = template
    for key, value in variables.items():
        formatted = formatted.replace(f"[{key}]", str(value))
    return formatted

def add_possible_answers(prompt: str, possible_answers: List[str]) -> str:
    """
    Add a numbered list of possible answers to a prompt.
    
    Args:
        prompt: Base prompt string
        possible_answers: List of possible answers
        
    Returns:
        Prompt with possible answers appended
    """
    answers_str = "\n".join([f"{i+1}. {ans}" for i, ans in enumerate(possible_answers)])
    return f"{prompt}\n\nPossible answers:\n{answers_str}" 