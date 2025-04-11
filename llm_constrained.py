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
        """Create a Pydantic model with an enum field for possible answers."""
        # Create an Enum class dynamically
        AnswerEnum = Enum('AnswerEnum', {
            f'ANSWER_{i}': answer 
            for i, answer in enumerate(possible_answers)
        })
        
        # Create a Pydantic model dynamically
        ResponseModel = create_model(
            'ResponseModel',
            answer=(AnswerEnum, ...),  # ... means the field is required
        )
        
        return ResponseModel
    
    def generate_response(self, prompt: str, response_model: type[BaseModel]) -> str:
        """Generate constrained response using OpenAI API."""
        try:
            response = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                seed=42,
                response_format=response_model
            )
            
            return response.choices[0].message.parsed.answer.value
            
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
            response = self.generate_response(formatted_prompt, response_model)
            
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