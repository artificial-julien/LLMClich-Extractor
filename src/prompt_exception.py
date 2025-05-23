from dataclasses import dataclass

@dataclass
class LLMException(Exception):
    """Exception raised for errors in LLM interactions.
    
    Attributes:
        message -- explanation of the error
        prompt -- the prompt that caused the error
    """
    message: str
    prompt: str

    def __str__(self) -> str:
        return self.message
