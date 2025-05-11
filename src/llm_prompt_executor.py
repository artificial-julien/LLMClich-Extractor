from openai import OpenAI
from pathlib import Path
import pandas as pd
from typing import Union, Optional, List, Dict, Any
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

class LLMPromptExecutor:
    """
    Generator for processing prompts with constrained LLM responses.
    Handles prompt formatting, LLM calls, and result saving.
    Now refactored to be used as a handler for the Pipeline class.
    """
    def __init__(self, api_key: str, decimal_places: int = 4, parallel: int = 2):
        self.api_key = api_key
        self.decimal_places = decimal_places
        self.parallel = parallel

    def __call__(self, variables, models, templates, possible_answers, result_var_name, verbose=False):
        """
        Process the LLM step for all combinations of variables and models.
        Returns a list of result dicts.
        """
        results = []
        jobs = []
        # Build all combinations
        all_combinations = []
        for model_cfg in models:
            for template in templates:
                combo = variables.as_dict().copy()
                combo['__model__'] = model_cfg
                combo['__template__'] = template
                all_combinations.append(combo)
        all_combinations = deduplicate_combinations(all_combinations)
        total_iterations = sum(int(combo['__model__'].get('iterations', 1)) for combo in all_combinations)
        tqdm.write(f"Total iterations (all combinations): {total_iterations}")
        progress = tqdm(total=total_iterations, desc="Processing", unit="iteration")
        write_lock = threading.Lock()
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
            api_key = self.api_key
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
            result_row[result_var_name] = possible_answers[answer_index] if error_message is None else None
            result_row.update(build_result_row({}, model_name, temperature, top_p, seed, error_message, possible_answers, answer_index, response, response_content))
            with write_lock:
                results.append(result_row)
                progress.update(1)
            return True
        # Prepare all jobs
        for combo in all_combinations:
            model_cfg = combo['__model__']
            iterations = int(model_cfg.get('iterations', 1))
            for iteration in range(iterations):
                jobs.append((combo, iteration))
        # Run jobs in parallel
        with ThreadPoolExecutor(max_workers=self.parallel) as executor:
            futures = [executor.submit(process_one_request, combo, iteration) for combo, iteration in jobs]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    tqdm.write(f"[ERROR] Exception in parallel execution: {e}")
        progress.close()
        return results 