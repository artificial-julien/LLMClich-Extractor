import os
import json
import csv
from openai import OpenAI
from pathlib import Path
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv
from enum import Enum
from pydantic import BaseModel, create_model
import argparse

# Load environment variables from .env file
load_dotenv()

class LLMConstrainedGenerator:
    def __init__(self, api_key: str, model: str = "gpt-4o-mini", decimal_places: int = 4):
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
    
    def generate_response(self, prompt: str, possible_answers: List[str], response_model: type[BaseModel]) -> Tuple[str, Dict[str, float]]:
        """Generate constrained response using OpenAI API and return probabilities."""
        try:
            # Add numbered answers to the prompt
            numbered_answers = "\n".join(f"{i+1}. {answer}" for i, answer in enumerate(possible_answers))
            full_prompt = f"{prompt}\n\nPlease respond with the number of your answer from the following options:\n{numbered_answers}"
            
            # Define the JSON schema for the response
            json_schema = {
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
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": full_prompt}
                ],
                temperature=0.0,
                top_p=0.0,
                seed=42,
                logprobs=True,
                top_logprobs=10,
                extra_body={
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "numerical_answer",
                            "schema": json_schema
                        }
                    }
                },
            )
            
            # Parse the JSON response
            response_content = json.loads(response.choices[0].message.content)
            answer_index = response_content["answer"] - 1  # Convert to 0-based index
            
            # Extract logprobs from the response
            logprobs_data = {}
            if hasattr(response.choices[0], 'logprobs') and response.choices[0].logprobs:
                # Find the token that represents the numerical choice
                # We look for the token that contains the actual number in the JSON response
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
                                    prob = round((top_logprob.logprob), self.decimal_places)
                                    logprobs_data[number_str] = prob
                        break  # We found our target token, no need to continue
            
            return possible_answers[answer_index], logprobs_data
        except Exception as e:
            raise Exception(f"Failed to generate response: {str(e)}")

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
        
        # Process each row and generate responses
        results = []
        for row in variables:
            # Format prompt with variables if any exist
            formatted_prompt = template
            for key, value in row.items():
                formatted_prompt = formatted_prompt.replace(f"[{key}]", str(value))
            
            # Generate response and get probabilities
            response, probs = self.generate_response(formatted_prompt, possible_answers, response_model)
            
            # Start with either the existing row data or an empty dict
            result_row = row.copy()
            
            # Add response and probabilities to row
            result_row['generated_answer'] = response
            # Add probability for the selected answer
            answer_index = possible_answers.index(response) + 1
            result_row['answer_prob'] = probs.get(str(answer_index), None)
            
            # Add top 5 probabilities for each possible answer
            for i, answer in enumerate(possible_answers):
                prob_key = str(i+1)
                if prob_key in probs:
                    result_row[f'prob_{i+1}_{answer[:20]}'] = probs[prob_key]
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