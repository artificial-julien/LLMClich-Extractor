from typing import Dict, Any

def format_template(template: str, variables: Dict[str, Any]) -> str:
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
