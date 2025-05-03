import os
import json
import csv
from openai import OpenAI
from pathlib import Path
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, TypeVar, Optional
from dotenv import load_dotenv
from enum import Enum
from pydantic import BaseModel, create_model
import argparse

# Load environment variables from .env file
load_dotenv()

T = TypeVar('T')

def get_with_default(dictionary: Dict[str, Any], key: str, default_value: T) -> T:
    """Get a value from a dictionary with a default if the value is None, empty string, or NaN."""
    value = dictionary.get(key)
    if value is None or value == "" or (isinstance(value, float) and np.isnan(value)):
        return default_value
    return value

class LLMConstrainedGenerator:
    def __init__(self, api_key: str, model: str, decimal_places: int = 4):
        """Initialize the generator with OpenAI API key and model.
        
        Args:
            api_key: OpenAI API key
            model: Model to use for generation
            decimal_places: Number of decimal places to round probabilities to
        """
        # The OpenAI client will automatically use OPENAI_BASE_URL from environment if set
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.decimal_places = decimal_places
        
    def load_csv_variables(self, csv_path: str) -> List[Dict[str, str]]:
        """Load variables from CSV file."""
        return pd.read_csv(csv_path).to_dict('records')
    
    def load_prompt_template(self, template_path: str) -> str:
        """Load prompt template from file."""
        with open(template_path, 'r') as f:
            return f.read().strip()
    
    def load_possible_answers(self, answers_path: str) -> List[str]:
        """Load list of possible answers from file."""
        with open(answers_path, 'r') as f:
            return [line.strip() for line in f if line.strip()]
    
    def create_response_model(self, possible_answers: List[str]) -> type[BaseModel]:
        """Create a Pydantic model with an integer field for answer index."""
        # Create a Pydantic model with integer field
        ResponseModel = create_model(
            'ResponseModel',
            answer=(int, ...),  # ... means the field is required
        )
        
        return ResponseModel

    def process_directory(self, input_dir: str) -> None:
        """Process all files in the specified directory."""
        input_dir = Path(input_dir)
        
        # Load template and possible answers (these are required)
        template = self.load_prompt_template(str(input_dir / "prompt_template.txt"))
        possible_answers = self.load_possible_answers(str(input_dir / "possible_answers.txt"))
        
        # Create response model
        response_model = self.create_response_model(possible_answers)
        
        # Try to load variables file, but handle case where it's missing or empty
        variables_path = input_dir / "variables.csv"
        if variables_path.exists():
            try:
                variables = self.load_csv_variables(str(variables_path))
                if not variables:  # Empty CSV file
                    variables = [{}]  # Use empty dict for no variables
            except pd.errors.EmptyDataError:
                variables = [{}]  # Handle completely empty CSV
        else:
            variables = [{}]  # No CSV file exists
        
        # Try to load models configuration file
        models_path = input_dir / "models.csv"
        if models_path.exists():
            try:
                models_config = pd.read_csv(str(models_path)).to_dict('records')
                if not models_config:  # Empty CSV file
                    # Use default configuration from instance
                    models_config = [{
                        'endpoint': os.getenv("OPENAI_BASE_URL", None),
                        'api_key_env': "OPENAI_API_KEY",
                        'model': self.model,
                        'temperature': 0.0,
                        'top_p': 0.0,
                        'iterations': 1
                    }]
            except pd.errors.EmptyDataError:
                # Use default configuration
                models_config = [{
                    'endpoint': os.getenv("OPENAI_BASE_URL", None),
                    'api_key_env': "OPENAI_API_KEY",
                    'model': self.model,
                    'temperature': 0.0,
                    'top_p': 0.0,
                    'iterations': 1
                }]
        else:
            # Use default configuration
            models_config = [{
                'endpoint': os.getenv("OPENAI_BASE_URL", None),
                'api_key_env': "OPENAI_API_KEY",
                'model': self.model,
                'temperature': 0.0,
                'top_p': 0.0,
                'iterations': 1
            }]
        
        # Process each model configuration, variable combination, and iteration
        results = []
        for model_config in models_config:
            # Get API key from environment variable
            api_key_env = get_with_default(model_config, 'api_key_env', "OPENAI_API_KEY")
            api_key = os.getenv(api_key_env)
            if not api_key:
                raise ValueError(f"API key not found for environment variable: {api_key_env}")
            
            # Set up client for this model configuration
            endpoint = get_with_default(model_config, 'endpoint', None)
            endpoint = endpoint if endpoint else "https://api.openai.com/v1"
            model_client = OpenAI(api_key=api_key, base_url=endpoint) if endpoint else OpenAI(api_key=api_key)
            
            # Get model parameters
            model_name = get_with_default(model_config, 'model', self.model)
            temperature = float(get_with_default(model_config, 'temperature', 0.0))
            top_p = float(get_with_default(model_config, 'top_p', 0.0))
            iterations = int(get_with_default(model_config, 'iterations', 1))
            
            # Run specified number of iterations
            for iteration in range(iterations):
                # Process each variable combination
                for row in variables:
                    # Format prompt with variables if any exist
                    formatted_prompt = template
                    for key, value in row.items():
                        formatted_prompt = formatted_prompt.replace(f"[{key}]", str(value))
                    # Add numbered possible answers to the prompt
                    answers_str = "\n".join([f"{i+1}. {ans}" for i, ans in enumerate(possible_answers)])
                    formatted_prompt += f"\n\nPossible answers:\n{answers_str}"
                    
                    # Use a deterministic seed based on the iteration
                    seed = iteration
                    
                    # Generate response using the configured model
                    response = model_client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {"role": "user", "content": formatted_prompt}
                        ],
                        temperature=temperature,
                        top_p=top_p,
                        seed=seed,
                        logprobs=True,
                        top_logprobs=10,
                        extra_body={
                            "response_format": {
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
                        },
                    )
                    
                    # Parse the JSON response
                    response_content = json.loads(response.choices[0].message.content)
                    print(f"Response content: {response_content}")
                    answer_index = response_content["answer"] - 1  # Convert to 0-based index
                    
                    # Check if answer_index is in valid range
                    error_message = None
                    if answer_index < 0 or answer_index >= len(possible_answers):
                        error_message = "out of range"
                    
                    # Start with either the existing row data or an empty dict
                    result_row = row.copy()
                    
                    # Add model configuration details to result row
                    result_row['model-name'] = model_name
                    result_row['temperature'] = temperature
                    result_row['top_p'] = top_p
                    result_row['seed'] = seed
                    
                    # Add prompt, response and probabilities to row
                    result_row['prompt'] = formatted_prompt
                    result_row['generated_answer'] = possible_answers[answer_index] if error_message is None else None
                    result_row['error'] = error_message
                    
                    # Extract logprobs from the response
                    logprobs_data = {}
                    if hasattr(response.choices[0], 'logprobs') and response.choices[0].logprobs:
                        # Find the token that represents the numerical choice
                        target_number = str(response_content["answer"])
                        
                        for token_logprob in response.choices[0].logprobs.content:
                            # Check if this token contains our target number
                            if target_number in token_logprob.token:
                                if token_logprob.top_logprobs:
                                    # Process probabilities for this specific token
                                    for top_logprob in token_logprob.top_logprobs:
                                        # Extract the number from the token if it's a number
                                        number_str = ''.join(c for c in top_logprob.token if c.isdigit())
                                        if number_str and int(number_str) <= len(possible_answers):
                                            # Convert logprob to probability and round to specified decimal places
                                            prob = round(np.exp(top_logprob.logprob), self.decimal_places)
                                            logprobs_data[number_str] = prob
                                break  # We found our target token, no need to continue
                    
                    # Add probability for the selected answer
                    answer_num = answer_index + 1
                    result_row['answer_prob'] = logprobs_data.get(str(answer_num), None)
                    
                    # Add top 5 probabilities for each possible answer
                    for i, answer in enumerate(possible_answers):
                        prob_key = str(i+1)
                        if prob_key in logprobs_data:
                            result_row[f'prob_{i+1}_{answer[:20]}'] = logprobs_data[prob_key]
                        else:
                            result_row[f'prob_{i+1}_{answer[:20]}'] = None  # Use None for unknown probabilities
                    
                    results.append(result_row)
        
        # Save results with NA values for empty cells
        output_df = pd.DataFrame(results)
        output_df.to_csv(str(input_dir / "output_results.csv"), index=False, na_rep='')

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process prompts with constrained LLM responses')
    parser.add_argument('input_dir', help='Directory containing input files (variables.csv, prompt_template.txt, possible_answers.txt)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Get API key and optional model from environment variables
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    decimal_places = int(os.getenv("DECIMAL_PLACES", "4"))  # Get decimal places from env var
    
    if not api_key:
        raise ValueError("Please set OPENAI_API_KEY environment variable")
    
    # Initialize generator
    generator = LLMConstrainedGenerator(api_key, model, decimal_places)
    
    # Process directory using command line argument
    generator.process_directory(args.input_dir)

if __name__ == "__main__":
    main()