import os
from typing import Dict, Any, List, Optional, Union, Literal
from openai import OpenAI
import json
from .prompt_engineering import extract_json_from_text, generate_constrained_prompt

class LLMClient:
    """
    Client for interacting with OpenAI language models.
    """
    
    def __init__(
        self, 
        api_key: Optional[str] = None, 
        base_url: Optional[str] = None
    ):
        """
        Initialize the LLM client.
        
        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY environment variable)
            base_url: Optional base URL for the API
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("No API key provided and OPENAI_API_KEY environment variable not set")
        
        self.client = OpenAI(api_key=self.api_key, base_url=base_url)

    def _create_json_schema(self, possible_answers: List[str], answer_format: Literal["enum", "numbered"]) -> Dict[str, Any]:
        """Create JSON schema for constrained responses."""
        if answer_format == "enum":
            return {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "enum": possible_answers
                    }
                },
                "required": ["answer"]
            }
        else:  # numbered format
            return {
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

    def _process_response_content(
        self, 
        response_content: Dict[str, Any], 
        possible_answers: List[str],
        answer_format: Literal["enum", "numbered"]
    ) -> Dict[str, Any]:
        """Process the response content and validate the answer."""
        if answer_format == "enum":
            chosen_answer = response_content["answer"]
            if chosen_answer not in possible_answers:
                return {
                    "chosen_answer": None,
                    "answer_index": None,
                    "error": f"Invalid answer: {chosen_answer} not in {possible_answers}"
                }
            return {
                "chosen_answer": chosen_answer,
                "answer_index": possible_answers.index(chosen_answer),
                "error": None
            }
        else:  # numbered format
            answer_index = response_content["answer"] - 1
            if answer_index < 0 or answer_index >= len(possible_answers):
                return {
                    "chosen_answer": None,
                    "answer_index": None,
                    "error": f"Answer index out of range: #{answer_index} for {len(possible_answers)} possible answers {possible_answers}"
                }
            return {
                "chosen_answer": possible_answers[answer_index],
                "answer_index": answer_index,
                "error": None
            }

    def _extract_probabilities(
        self, 
        response: Any,
        answer_format: Literal["enum", "numbered"],
        answer_index: Optional[int],
        chosen_answer: Optional[str]
    ) -> Dict[str, Any]:
        """Extract probabilities from the response."""
        probabilities = {}
        if not hasattr(response.choices[0], 'logprobs') or not response.choices[0].logprobs:
            return {"probabilities": None, "answer_probability": None}

        if answer_format == "enum":
            # TODO: Implement enum probability extraction if needed
            return {"probabilities": None, "answer_probability": None}
        else:  # numbered format
            target_number = str(answer_index + 1)
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
            "probabilities": probabilities,
            "answer_probability": probabilities.get(target_number) if probabilities else None
        }

    def _generate_single_completion(
        self,
        model: str,
        prompt: str,
        possible_answers: List[str],
        temperature: float,
        top_p: float,
        llm_seed: Optional[int],
        constraint_method: Literal["json_schema", "prompt_engineering"],
        answer_format: Literal["enum", "numbered"]
    ) -> Dict[str, Any]:
        if constraint_method == "json_schema":
            schema = self._create_json_schema(possible_answers, answer_format)
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                top_p=top_p,
                seed=llm_seed,
                logprobs=True,
                top_logprobs=10,
                extra_body={
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "answer_schema",
                            "schema": schema
                        }
                    }
                },
            )
            if response.choices[0].message.refusal is not None:
                raise ValueError(f"LLM refused to answer: {response.choices[0].message.refusal}")
            response_content = json.loads(response.choices[0].message.content)
        else:  # prompt_engineering
            enhanced_prompt = generate_constrained_prompt(prompt, possible_answers, answer_format)
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": enhanced_prompt}],
                temperature=temperature,
                top_p=top_p,
                seed=llm_seed,
                logprobs=True,
                top_logprobs=10
            )
            
            response_content = extract_json_from_text(response.choices[0].message.content)
            if not response_content or "answer" not in response_content:
                raise ValueError("Failed to extract valid answer from response")

        # Process response content
        result = self._process_response_content(response_content, possible_answers, answer_format)
        
        # Extract probabilities
        prob_result = self._extract_probabilities(
            response, 
            answer_format, 
            result["answer_index"], 
            result["chosen_answer"]
        )
        
        return {
            "chosen_answer": result["chosen_answer"],
            "answer_index": result["answer_index"],
            "answer_number": result["answer_index"] + 1 if result["chosen_answer"] is not None else None,
            "error": result["error"],
            "probabilities": prob_result["probabilities"],
            "answer_probability": prob_result["answer_probability"],
            "raw_response": response
        }

    def generate_constrained_completion(
        self, 
        model: str,
        prompt: str,
        possible_answers: List[str],
        temperature: float = 0.0,
        top_p: float = 1.0,
        llm_seed: Optional[int] = None,
        constraint_method: Literal["json_schema", "prompt_engineering"] = "json_schema",
        answer_format: Literal["enum", "numbered"] = "enum",
        max_tries: int = 1
    ) -> Dict[str, Any]:
        """
        Generate a completion with a constrained set of possible answers.
        
        Args:
            model: The model name to use
            prompt: The prompt to send to the model
            possible_answers: List of possible answers to constrain responses
            temperature: Sampling temperature (0-2)
            top_p: Nucleus sampling parameter (0-1)
            llm_seed: Optional llm_seed for deterministic generation
            constraint_method: Method to use for constraining responses
                             - "json_schema": Use JSON schema (default, requires model support)
                             - "prompt_engineering": Use prompt engineering and JSON extraction
            answer_format: Format for the answer
                         - "enum": Direct answer selection (default)
                         - "numbered": Numbered answer selection
            max_tries: Maximum number of attempts to generate a valid completion
            
        Returns:
            Dictionary with results including chosen answer, probabilities, etc.
            
        Raises:
            Exception: If all attempts fail to generate a valid completion
        """
        last_error = None
        for attempt in range(max_tries):
            try:
                result = self._generate_single_completion(
                    model=model,
                    prompt=prompt,
                    possible_answers=possible_answers,
                    temperature=temperature,
                    top_p=top_p,
                    llm_seed=llm_seed,
                    constraint_method=constraint_method,
                    answer_format=answer_format
                )
                
                if result["error"] is None:
                    return result
                    
                last_error = result["error"]
            except Exception as e:
                last_error = str(e)
                
        raise Exception(f"Failed to generate valid completion after {max_tries} attempts. Last error: {last_error}") 