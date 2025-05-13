import os
from typing import Dict, Any, List, Optional, Union, Literal
import aisuite as ai
import json
from .prompt_engineering import extract_json_from_text, generate_constrained_prompt

class LLMClient:
    """
    Client for interacting with language models using AISuite.
    """
    
    def __init__(
        self, 
        api_key: Optional[str] = None, 
        base_url: Optional[str] = None
    ):
        """
        Initialize the LLM client.
        
        Args:
            api_key: API key (defaults to OPENAI_API_KEY environment variable)
            base_url: Optional base URL for the API
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("No API key provided and OPENAI_API_KEY environment variable not set")
        
        # Initialize AISuite client
        self.client = ai.Client()
    
    def generate_constrained_completion(
        self, 
        model: str,
        prompt: str,
        possible_answers: List[str],
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: Optional[int] = None,
        constraint_method: Literal["json_schema", "prompt_engineering"] = "json_schema"
    ) -> Dict[str, Any]:
        """
        Generate a completion with a constrained set of possible answers.
        
        Args:
            model: The model name to use
            prompt: The prompt to send to the model
            possible_answers: List of possible answers to constrain responses
            temperature: Sampling temperature (0-2)
            top_p: Nucleus sampling parameter (0-1)
            seed: Optional seed for deterministic generation
            constraint_method: Method to use for constraining responses
                             - "json_schema": Use JSON schema (default, requires model support)
                             - "prompt_engineering": Use prompt engineering and JSON extraction
            
        Returns:
            Dictionary with results including chosen answer, probabilities, etc.
        """
        # Ensure model name has provider prefix
        if ":" not in model:
            model = f"openai:{model}"

        if constraint_method == "json_schema":
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                top_p=top_p,
                seed=seed,
                logprobs=True,
                top_logprobs=10,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "numerical_answer",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "answer": {
                                    "type": "integer",
                                    "minimum": 1,
                                    "maximum": len(possible_answers)
                                }
                            },
                            "required": ["answer"]
                        }
                    }
                }
            )
            response_content = json.loads(response.choices[0].message.content)
        else:  # prompt_engineering
            enhanced_prompt = generate_constrained_prompt(prompt, possible_answers)
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": enhanced_prompt}],
                temperature=temperature,
                top_p=top_p,
                seed=seed,
                logprobs=True,
                top_logprobs=10
            )
            
            # Extract JSON from the response
            response_content = extract_json_from_text(response.choices[0].message.content)
            if not response_content or "answer" not in response_content:
                raise ValueError("Failed to extract valid answer from response")
        
        answer_index = response_content["answer"] - 1
        
        # Validate the answer index
        if answer_index < 0 or answer_index >= len(possible_answers):
            raise ValueError("Answer index out of range")
        
        # Extract probabilities
        chosen_answer = possible_answers[answer_index] if not error_message else None
        probabilities = {}
        
        if hasattr(response.choices[0], 'logprobs') and response.choices[0].logprobs:
            target_number = str(response_content["answer"])
            for token_logprob in response.choices[0].logprobs.content:
                if target_number in token_logprob.token:
                    if token_logprob.top_logprobs:
                        for top_logprob in token_logprob.top_logprobs:
                            number_str = ''.join(c for c in top_logprob.token if c.isdigit())
                            if number_str and int(number_str) <= len(possible_answers):
                                import math
                                prob = round(math.exp(top_logprob.logprob), 4)
                                probabilities[number_str] = prob
                    break
        
        return {
            "chosen_answer": chosen_answer,
            "answer_index": answer_index,
            "answer_number": answer_index + 1,
            "probabilities": probabilities,
            "answer_probability": probabilities.get(str(answer_index + 1)),
            "raw_response": response
        }