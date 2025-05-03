import os
import json
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

PROMPT_TEMPLATE_FILE = "prompt_template.txt"
POSSIBLE_ANSWERS_FILE = "possible_answers.txt"
VARIABLES_FILE = "variables.csv"
MODELS_FILE = "models.csv"
OUTPUT_FILE = "output_results.csv"
DEFAULT_API_KEY_ENV = "OPENAI_API_KEY"
DEFAULT_MODEL_NAME = "gpt-4o-mini"
DEFAULT_MODEL_CONFIG = {
    'endpoint': os.getenv("OPENAI_BASE_URL", None),
    'api_key_env': DEFAULT_API_KEY_ENV,
    'model': DEFAULT_MODEL_NAME,
    'temperature': 0.0,
    'top_p': 0.0,
    'iterations': 1
}

class LLMConstrainedGenerator:
    def __init__(self, api_key: str, decimal_places: int = 4):
        """Initialize the generator with OpenAI API key and decimal places for rounding probabilities."""
        self.client = OpenAI(api_key=api_key)
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

    def load_csv_with_default(self, csv_path: str, default: list) -> list:
        """Load a CSV file and return its records as a list of dicts, or a default if missing/empty."""
        if Path(csv_path).exists():
            try:
                data = pd.read_csv(csv_path).to_dict('records')
                if not data:
                    return default
                return data
            except pd.errors.EmptyDataError:
                return default
        return default

    def load_models_config(self, input_dir: Path) -> list:
        """Load model configurations from models.csv, or use the default config if missing/empty."""
        models_path = input_dir / MODELS_FILE
        default_config = [DEFAULT_MODEL_CONFIG.copy()]
        return self.load_csv_with_default(str(models_path), default_config)

    def get_model_client(self, model_config: dict) -> OpenAI:
        """Instantiate and return an OpenAI client for the given model configuration."""
        api_key_env = get_with_default(model_config, 'api_key_env', DEFAULT_API_KEY_ENV)
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise ValueError(f"API key not found for environment variable: {api_key_env}")
        endpoint = get_with_default(model_config, 'endpoint', None)
        endpoint = endpoint if endpoint else "https://api.openai.com/v1"
        return OpenAI(api_key=api_key, base_url=endpoint) if endpoint else OpenAI(api_key=api_key)

    def format_prompt(self, template: str, row: dict, possible_answers: list) -> str:
        """Format the prompt by substituting variables and appending possible answers."""
        formatted_prompt = template
        for key, value in row.items():
            formatted_prompt = formatted_prompt.replace(f"[{key}]", str(value))
        answers_str = "\n".join([f"{i+1}. {ans}" for i, ans in enumerate(possible_answers)])
        formatted_prompt += f"\n\nPossible answers:\n{answers_str}"
        return formatted_prompt

    def parse_response(self, response, possible_answers: list) -> tuple:
        """Parse the LLM response and return the content, answer index, and error message if any."""
        response_content = json.loads(response.choices[0].message.content)
        answer_index = response_content["answer"] - 1
        error_message = None
        if answer_index < 0 or answer_index >= len(possible_answers):
            error_message = "out of range"
        return response_content, answer_index, error_message

    def build_result_row(self, row: dict, model_name: str, temperature: float, top_p: float, seed: int, error_message: str, possible_answers: list, answer_index: int, response, response_content) -> dict:
        """Build a result row with all relevant information and probabilities from the response."""
        result_row = row.copy()
        result_row['model-name'] = model_name
        result_row['temperature'] = temperature
        result_row['top_p'] = top_p
        result_row['seed'] = seed
        result_row['error'] = error_message
        result_row['chosen_answer'] = possible_answers[answer_index] if error_message is None else None
        logprobs_data = {}
        if hasattr(response.choices[0], 'logprobs') and response.choices[0].logprobs:
            target_number = str(response_content["answer"])
            for token_logprob in response.choices[0].logprobs.content:
                if target_number in token_logprob.token:
                    if token_logprob.top_logprobs:
                        for top_logprob in token_logprob.top_logprobs:
                            number_str = ''.join(c for c in top_logprob.token if c.isdigit())
                            if number_str and int(number_str) <= len(possible_answers):
                                prob = round(np.exp(top_logprob.logprob), self.decimal_places)
                                logprobs_data[number_str] = prob
                    break
        answer_num = answer_index + 1
        result_row['answer_prob'] = logprobs_data.get(str(answer_num), None)
        for i, answer in enumerate(possible_answers):
            prob_key = str(i+1)
            if prob_key in logprobs_data:
                result_row[f'prob_{i+1}_{answer[:20]}'] = logprobs_data[prob_key]
            else:
                result_row[f'prob_{i+1}_{answer[:20]}'] = None
        return result_row

    def save_results(self, results: list, input_dir: Path):
        """Save the results to a CSV file in the input directory."""
        output_df = pd.DataFrame(results)
        output_df.to_csv(str(input_dir / OUTPUT_FILE), index=False, na_rep='')

    def process_directory(self, input_dir: str, verbose: bool = False) -> None:
        input_dir = Path(input_dir)
        template = self.load_prompt_template(str(input_dir / PROMPT_TEMPLATE_FILE))
        possible_answers = self.load_possible_answers(str(input_dir / POSSIBLE_ANSWERS_FILE))
        response_model = self.create_response_model(possible_answers)
        variables = self.load_csv_with_default(str(input_dir / VARIABLES_FILE), [{}])
        # Prefix variable columns
        variables = [
            {f"var_{k}": v for k, v in row.items()} for row in variables
        ]
        models_config = self.load_models_config(input_dir)
        output_path = input_dir / OUTPUT_FILE

        # Check for existing output and load if present
        existing_df = None
        if output_path.exists():
            try:
                existing_df = pd.read_csv(output_path)
            except pd.errors.EmptyDataError:
                existing_df = None

        # Check if variables match
        variables_df = pd.DataFrame(variables)
        if existing_df is not None and not existing_df.empty:
            # Only compare columns with 'var_' prefix
            variable_cols = set([col for col in variables_df.columns if col.startswith('var_')])
            existing_variable_cols = set([col for col in existing_df.columns if col.startswith('var_')])
            if variable_cols != existing_variable_cols:
                raise ValueError("Variable columns in output file do not match current variables.csv. Please clear or backup the output file. Existing variable columns: " + str(existing_variable_cols) + " Current variable columns: " + str(variable_cols))

        # Prepare for appending
        write_header = not output_path.exists()

        skipped_rows = 0
        added_rows = 0

        for model_config in models_config:
            model_client = self.get_model_client(model_config)
            model_name = get_with_default(model_config, 'model', DEFAULT_MODEL_NAME)
            temperature = float(get_with_default(model_config, 'temperature', 0.0))
            top_p = float(get_with_default(model_config, 'top_p', 0.0))
            iterations = int(get_with_default(model_config, 'iterations', 1))
            for iteration in range(iterations):
                for row in variables:
                    # Check if this row/model/params/seed already exists in output
                    already_done = False
                    if existing_df is not None and not existing_df.empty:
                        # Build a query mask
                        mask = (
                            (existing_df.get('model-name', None) == model_name) &
                            (existing_df.get('temperature', None) == temperature) &
                            (existing_df.get('top_p', None) == top_p) &
                            (existing_df.get('seed', None) == iteration)
                        )
                        for k, v in row.items():
                            if k in existing_df.columns:
                                mask &= (existing_df[k] == v)
                        if mask.any():
                            already_done = True
                    if already_done:
                        skipped_rows += 1
                        continue

                    if verbose:
                        print(f"[VERBOSE] Request: model={model_name}, temperature={temperature}, top_p={top_p}, iteration={iteration}, variables={row}")

                    # Remove 'var_' prefix for prompt formatting
                    prompt_row = {k[4:]: v for k, v in row.items() if k.startswith('var_')}
                    formatted_prompt = self.format_prompt(template, prompt_row, possible_answers)
                    seed = iteration
                    response = model_client.chat.completions.create(
                        model=model_name,
                        messages=[{"role": "user", "content": formatted_prompt}],
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
                    response_content, answer_index, error_message = self.parse_response(response, possible_answers)
                    
                    # Build result row with prefixed variable columns
                    result_row = row.copy()
                    result_row.update(self.build_result_row({}, model_name, temperature, top_p, seed, error_message, possible_answers, answer_index, response, response_content))
                    # Write result row immediately
                    result_df = pd.DataFrame([result_row])
                    result_df.to_csv(str(output_path), mode='a', header=write_header, index=False, na_rep='')
                    write_header = False  # Only write header for the first row
                    added_rows += 1


        print(f"Skipped rows (already present): {skipped_rows}")
        print(f"Added rows: {added_rows}")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process prompts with constrained LLM responses')
    parser.add_argument('input_dir', help='Directory containing input files (variables.csv, prompt_template.txt, possible_answers.txt)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging of each request')
    # Parse arguments
    args = parser.parse_args()
    # Get API key and optional model from environment variables
    api_key = os.getenv("OPENAI_API_KEY")
    decimal_places = int(os.getenv("DECIMAL_PLACES", "4"))  # Get decimal places from env var
    if not api_key:
        raise ValueError("Please set OPENAI_API_KEY environment variable")
    # Initialize generator
    generator = LLMConstrainedGenerator(api_key, decimal_places)
    # Process directory using command line argument
    generator.process_directory(args.input_dir, verbose=args.verbose)

if __name__ == "__main__":
    main()