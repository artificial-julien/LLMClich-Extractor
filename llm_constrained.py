import os
import json
import csv
from openai import OpenAI
from pathlib import Path
import pandas as pd
from typing import List, Dict, Any
from dotenv import load_dotenv
from enum import Enum
from pydantic import BaseModel, create_model

# Load environment variables from .env file
load_dotenv()

class LLMConstrainedGenerator:
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo-0125"):
        """Initialize the generator with OpenAI API key and model."""
        self.client = OpenAI(api_key=api_key)
        self.model = model
        
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
    
    def generate_response(self, prompt: str, possible_answers: List[str], response_model: type[BaseModel]) -> str:
        """Generate constrained response using OpenAI API."""
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
                    {"role": "system", "content": "You are a helpful assistant that responds with a JSON object containing an 'answer' field with the number of the selected option."},
                    {"role": "user", "content": full_prompt}
                ],
                temperature=0.0,
                seed=42,
                response_format={"type": "json_object", "schema": json_schema}
            )
            
            # Parse the JSON response
            response_content = json.loads(response.choices[0].message.content)
            answer_index = response_content["answer"] - 1  # Convert to 0-based index
            return possible_answers[answer_index]
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return ""

    def process_directory(self, input_dir: str) -> None:
        """Process all files in the specified directory."""
        input_dir = Path(input_dir)
        
        # Load required files
        variables = self.load_csv_variables(str(input_dir / "variables.csv"))
        template = self.load_prompt_template(str(input_dir / "prompt_template.txt"))
        possible_answers = self.load_possible_answers(str(input_dir / "possible_answers.txt"))
        
        # Create response model
        response_model = self.create_response_model(possible_answers)
        
        # Process each row and generate responses
        results = []
        for row in variables:
            # Format prompt with variables
            formatted_prompt = template
            for key, value in row.items():
                formatted_prompt = formatted_prompt.replace(f"[{key}]", str(value))
            
            # Generate response
            response = self.generate_response(formatted_prompt, possible_answers, response_model)
            
            # Add response to row
            row['generated_answer'] = response
            results.append(row)
        
        # Save results
        output_df = pd.DataFrame(results)
        output_df.to_csv(str(input_dir / "output_results.csv"), index=False)

def main():
    # Get API key and optional model from environment variables
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    
    if not api_key:
        raise ValueError("Please set OPENAI_API_KEY environment variable")
    
    # Initialize generator
    generator = LLMConstrainedGenerator(api_key, model)
    
    # Process directory (assuming example directory)
    generator.process_directory("example")

if __name__ == "__main__":
    main() 