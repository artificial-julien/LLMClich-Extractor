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
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

class LLMConstrainedGenerator:
    """
    Generator for processing prompts with constrained LLM responses.
    Handles prompt formatting, LLM calls, and result saving.
    """
    def __init__(self, api_key: str, decimal_places: int = 4, parallel: int = 2):
        """Initialize the generator with OpenAI API key, decimal places, and parallelism."""
        self.client = OpenAI(api_key=api_key)
        self.decimal_places = decimal_places
        self.parallel = parallel

    def save_results(self, results: list, input_dir: Path) -> None:
        """Save the results to a CSV file in the input directory."""
        output_df = pd.DataFrame(results)
        output_df.to_csv(str(input_dir / OUTPUT_FILE), index=False, na_rep='')

    def process_json(self, json_path: Union[str, Path], output_path: Optional[Union[str, Path]] = None, verbose: bool = False, parallel: int = None) -> None:
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
        tqdm.write(f"Removed {removed_count} duplicate combinations.")
        tqdm.write(f"Unique combinations: {len(all_combinations)}")

        # Calculate total iterations
        total_iterations = sum(int(combo['__model__'].get('iterations', 1)) for combo in all_combinations)
        tqdm.write(f"Total iterations (all combinations): {total_iterations}")

        skipped_rows = 0
        added_rows = 0
        write_header = not output_path.exists()

        progress = tqdm(total=total_iterations, desc="Processing", unit="iteration")
        completed_iterations = 0

        if parallel is None:
            parallel = getattr(self, 'parallel', 2)

        write_lock = threading.Lock()
        header_written = [False]  # Use a mutable object to allow modification in nested function
        header_lock = threading.Lock()
        def process_one_request(combo, iteration):
            model_cfg = combo['__model__']
            model_name = model_cfg['name']
            temperature = float(model_cfg.get('temperature', 0.0))
            top_p = float(model_cfg.get('top_p', 0.0))
            row = {k: v for k, v in combo.items() if k not in ('__model__', '__template__')}
            template = combo.get('__template__')
            template_hash = hashlib.sha256(template.encode('utf-8')).hexdigest()[:8]
            if verbose:
                template_preview = template[:100] + "..." if len(template) > 100 else template
                tqdm.write(f"[VERBOSE] Request: model={model_name}, temperature={temperature}, top_p={top_p}, iteration={iteration}, variables={row}, template_hash={template_hash}, template={template_preview}")
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
            with write_lock:
                result_df = pd.DataFrame([result_row])
                with header_lock:
                    if not header_written[0]:
                        result_df.to_csv(str(output_path), mode='a', header=True, index=False, na_rep='')
                        header_written[0] = True
                    else:
                        result_df.to_csv(str(output_path), mode='a', header=False, index=False, na_rep='')
                progress.update(1)
            return True

        # Prepare all jobs (skip already-done rows)
        jobs = []
        for combo in all_combinations:
            model_cfg = combo['__model__']
            model_name = model_cfg['name']
            temperature = float(model_cfg.get('temperature', 0.0))
            top_p = float(model_cfg.get('top_p', 0.0))
            iterations = int(model_cfg.get('iterations', 1))
            row = {k: v for k, v in combo.items() if k not in ('__model__', '__template__')}
            template = combo.get('__template__')
            if template is None:
                raise ValueError("No template found in combination. Please include a 'template' node_type in your foreach list.")
            template_hash = hashlib.sha256(template.encode('utf-8')).hexdigest()[:8]
            for iteration in range(iterations):
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
                    if 'template_hash' in existing_df.columns:
                        mask &= (existing_df['template_hash'] == template_hash)
                    else:
                        mask &= False
                    if mask.any():
                        already_done = True
                if already_done:
                    skipped_rows += 1
                    progress.update(1)
                    continue
                jobs.append((combo, iteration))

        # Run jobs in parallel
        with ThreadPoolExecutor(max_workers=parallel) as executor:
            futures = [executor.submit(process_one_request, combo, iteration) for combo, iteration in jobs]
            for future in as_completed(futures):
                try:
                    future.result()
                    added_rows += 1
                except Exception as e:
                    tqdm.write(f"[ERROR] Exception in parallel execution: {e}")

        progress.close()
        tqdm.write(f"Skipped rows (already present): {skipped_rows}")
        tqdm.write(f"Added rows: {added_rows}") 