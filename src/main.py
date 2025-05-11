import os
import argparse
from dotenv import load_dotenv
from pipeline import Pipeline
from llm_prompt_executor import LLMPromptExecutor

def main():
    """CLI entrypoint for processing prompts with constrained LLM responses (JSON input)."""
    load_dotenv()
    parser = argparse.ArgumentParser(description='Process prompts with constrained LLM responses (JSON input)')
    parser.add_argument('input_json', help='JSON input file (with template, possible_answers, foreach)')
    parser.add_argument('--output', help='Output CSV file or directory (optional)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging of each request')
    parser.add_argument('--parallel', type=int, default=2, help='Number of parallel chat completions (default: 2)')
    args = parser.parse_args()

    # Environment variables
    api_key = os.getenv("OPENAI_API_KEY")
    decimal_places = int(os.getenv("DECIMAL_PLACES", "4"))
    if not api_key:
        raise ValueError("Please set OPENAI_API_KEY environment variable")

    llm_handler = LLMPromptExecutor(api_key, decimal_places, parallel=args.parallel)
    pipeline = Pipeline(args.input_json)
    pipeline.run(llm_handler=llm_handler, verbose=args.verbose)

if __name__ == "__main__":
    main() 