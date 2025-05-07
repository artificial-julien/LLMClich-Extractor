import os
import json
from pathlib import Path
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, TypeVar, Optional
from dataclasses import dataclass, asdict

T = TypeVar('T')

@dataclass
class ModelConfig:
    endpoint: Optional[str]
    api_key_env: str
    model: str
    temperature: float
    top_p: float
    iterations: int
    name: Optional[str] = None  # For compatibility with input JSON

PROMPT_TEMPLATE_FILE = "prompt_template.txt"
POSSIBLE_ANSWERS_FILE = "possible_answers.txt"
VARIABLES_FILE = "variables.csv"
MODELS_FILE = "models.csv"
OUTPUT_FILE = "output_results.csv"
DEFAULT_API_KEY_ENV = "OPENAI_API_KEY"
DEFAULT_MODEL_NAME = "gpt-4o-mini"
DEFAULT_MODEL_CONFIG = ModelConfig(
    endpoint=os.getenv("OPENAI_BASE_URL", None),
    api_key_env=DEFAULT_API_KEY_ENV,
    model=DEFAULT_MODEL_NAME,
    temperature=0.0,
    top_p=0.0,
    iterations=1,
)

def get_with_default(dictionary: Dict[str, Any], key: str, default_value: T) -> T:
    """Get a value from a dictionary with a default if the value is None, empty string, or NaN."""
    value = dictionary.get(key)
    if value is None or value == "" or (isinstance(value, float) and np.isnan(value)):
        return default_value
    return value

def load_csv_with_default(csv_path: str, default: list) -> list:
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

def load_models_config(input_dir: Path) -> List[Dict[str, Any]]:
    """Load model configurations from models.csv, or use the default config if missing/empty."""
    models_path = input_dir / MODELS_FILE
    default_config = [asdict(DEFAULT_MODEL_CONFIG)]
    return load_csv_with_default(str(models_path), default_config)

def create_response_model(possible_answers: List[str]):
    from pydantic import create_model
    ResponseModel = create_model(
        'ResponseModel',
        answer=(int, ...),
    )
    return ResponseModel

def format_prompt(template: str, row: dict, possible_answers: list) -> str:
    formatted_prompt = template
    for key, value in row.items():
        formatted_prompt = formatted_prompt.replace(f"[{key}]", str(value))
    answers_str = "\n".join([f"{i+1}. {ans}" for i, ans in enumerate(possible_answers)])
    formatted_prompt += f"\n\nPossible answers:\n{answers_str}"
    return formatted_prompt

def parse_response(response, possible_answers: list) -> Tuple[dict, int, Optional[str]]:
    response_content = json.loads(response.choices[0].message.content)
    answer_index = response_content["answer"] - 1
    error_message = None
    if answer_index < 0 or answer_index >= len(possible_answers):
        error_message = "out of range"
    return response_content, answer_index, error_message

def build_result_row(
    row: dict,
    model_name: str,
    temperature: float,
    top_p: float,
    seed: int,
    error_message: str,
    possible_answers: list,
    answer_index: int,
    response,
    response_content
) -> dict:
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
                            prob = round(np.exp(top_logprob.logprob), 4)
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

def expand_foreach(nodes: list, idx: int = 0, current: Optional[dict] = None) -> List[dict]:
    if current is None:
        current = {}
    if idx >= len(nodes):
        return [current]
    node = nodes[idx]
    node_type = node['node_type']
    results = []
    for item in node['list']:
        new_current = current.copy()
        if node_type == 'models':
            new_current['__model__'] = item
        elif node_type == 'variables':
            new_current.update(item)
        else:
            new_current[node_type] = item
        results.extend(expand_foreach(nodes, idx+1, new_current))
    return results

def deduplicate_combinations(all_combinations: List[dict]) -> List[dict]:
    seen = set()
    deduped_combinations = []
    for combo in all_combinations:
        combo_tuple = tuple(sorted((k, json.dumps(v, sort_keys=True)) for k, v in combo.items()))
        if combo_tuple not in seen:
            seen.add(combo_tuple)
            deduped_combinations.append(combo)
    return deduped_combinations 