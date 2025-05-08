from openai import OpenAI
from pathlib import Path
import pandas as pd
from typing import Union, Optional
from utils import (
    OUTPUT_FILE,
    DEFAULT_API_KEY_ENV,
    format_prompt,
    parse_response,
    build_result_row,
    deduplicate_combinations,
    expand_foreach
)
from tqdm import tqdm
import hashlib

class LLMConstrainedGenerator:
    """
    Generator for processing prompts with constrained LLM responses.
    Handles prompt formatting, LLM calls, and result saving.
    """
    def __init__(self, api_key: str, decimal_places: int = 4):
        """Initialize the generator with OpenAI API key and decimal places for rounding probabilities."""
        self.client = OpenAI(api_key=api_key)
        self.decimal_places = decimal_places

    def save_results(self, results: list, input_dir: Path) -> None:
        """Save the results to a CSV file in the input directory."""
        output_df = pd.DataFrame(results)
        output_df.to_csv(str(input_dir / OUTPUT_FILE), index=False, na_rep='')

    def process_json(self, json_path: Union[str, Path], output_path: Optional[Union[str, Path]] = None, verbose: bool = False) -> None:
        """Process a JSON config file, run LLM prompts, and save results to CSV."""
        import os
        import json
        json_path = Path(json_path)
        with open(json_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        possible_answers = config['possible_answers']
        foreach = config['foreach']

        # No special handling for template node_type; treat all nodes the same

        # Determine output path
        if output_path is None:
            output_path = json_path.parent / OUTPUT_FILE
        else:
            output_path = Path(output_path)
            if output_path.is_dir():
                output_path = output_path / OUTPUT_FILE

        # Prepare for skipping already-done rows
        existing_df = None
        if output_path.exists():
            try:
                existing_df = pd.read_csv(output_path)
            except pd.errors.EmptyDataError:
                existing_df = None

        # Build all combinations from foreach
        all_combinations = expand_foreach(foreach)
        original_count = len(all_combinations)
        all_combinations = deduplicate_combinations(all_combinations)
        removed_count = original_count - len(all_combinations)
        print(f"Removed {removed_count} duplicate combinations.")
        print(f"Unique combinations: {len(all_combinations)}")

        # Calculate total iterations
        total_iterations = sum(int(combo['__model__'].get('iterations', 1)) for combo in all_combinations)
        print(f"Total iterations (all combinations): {total_iterations}")

        skipped_rows = 0
        added_rows = 0
        write_header = not output_path.exists()

        progress = tqdm(total=total_iterations, desc="Processing", unit="iteration")
        completed_iterations = 0

        for combo in all_combinations:
            model_cfg = combo['__model__']
            model_name = model_cfg['name']
            temperature = float(model_cfg.get('temperature', 0.0))
            top_p = float(model_cfg.get('top_p', 0.0))
            iterations = int(model_cfg.get('iterations', 1))
            # Remove model and template from row for prompt
            row = {k: v for k, v in combo.items() if k not in ('__model__', '__template__')}
            template = combo.get('__template__')
            if template is None:
                raise ValueError("No template found in combination. Please include a 'template' node_type in your foreach list.")
            template_hash = hashlib.sha256(template.encode('utf-8')).hexdigest()[:8]
            for iteration in range(iterations):
                # Check if this row/model/params/seed/template_hash already exists in output
                already_done = False
                if existing_df is not None and not existing_df.empty:
                    mask = (
                        (existing_df.get('model-name', None) == model_name) &
                        (existing_df.get('temperature', None) == temperature) &
                        (existing_df.get('top_p', None) == top_p) &
                        (existing_df.get('seed', None) == iteration)
                    )
                    for k, v in row.items():
                        if k in existing_df.columns:
                            mask &= (existing_df[k] == v)
                    # Add template_hash to mask
                    if 'template_hash' in existing_df.columns:
                        mask &= (existing_df['template_hash'] == template_hash)
                    else:
                        # If template_hash column doesn't exist, treat as not done
                        mask &= False
                    if mask.any():
                        already_done = True
                if already_done:
                    skipped_rows += 1
                    progress.update(1)
                    continue

                if verbose:
                    template_preview = template[:100] + "..." if len(template) > 100 else template
                    print(f"[VERBOSE] Request: model={model_name}, temperature={temperature}, top_p={top_p}, iteration={iteration}, variables={row}, template_hash={template_hash}, template={template_preview}")

                formatted_prompt = format_prompt(template, row, possible_answers)
                seed = iteration
                api_key_env = model_cfg.get('api_key_env', DEFAULT_API_KEY_ENV)
                api_key = os.getenv(api_key_env)
                endpoint = model_cfg.get('endpoint', None)
                model_client = OpenAI(api_key=api_key, base_url=endpoint) if endpoint else OpenAI(api_key=api_key)
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
                response_content, answer_index, error_message = parse_response(response, possible_answers)
                result_row = row.copy()
                result_row['template_hash'] = template_hash
                result_row.update(build_result_row({}, model_name, temperature, top_p, seed, error_message, possible_answers, answer_index, response, response_content))
                result_df = pd.DataFrame([result_row])
                result_df.to_csv(str(output_path), mode='a', header=write_header, index=False, na_rep='')
                write_header = False
                added_rows += 1
                progress.update(1)

        progress.close()
        print(f"Skipped rows (already present): {skipped_rows}")
        print(f"Added rows: {added_rows}") 