import os
import argparse
from llm_constrained_generator import LLMConstrainedGenerator
from dotenv import load_dotenv

def main():
    """CLI entrypoint for processing prompts with constrained LLM responses (JSON input)."""
    load_dotenv()
    parser = argparse.ArgumentParser(description='Process prompts with constrained LLM responses (JSON input)')
    parser.add_argument('input_json', help='JSON input file (with template, possible_answers, foreach)')
    parser.add_argument('--output', help='Output CSV file or directory (optional)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging of each request')
    args = parser.parse_args()
    api_key = os.getenv("OPENAI_API_KEY")
    decimal_places = int(os.getenv("DECIMAL_PLACES", "4"))
    if not api_key:
        raise ValueError("Please set OPENAI_API_KEY environment variable")
    generator = LLMConstrainedGenerator(api_key, decimal_places)
    generator.process_json(args.input_json, output_path=args.output, verbose=args.verbose)

if __name__ == "__main__":
    main() 