import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from variables import Variables

class Pipeline:
    """
    Pipeline class for processing the new JSON schema and managing pipeline steps.
    """
    def __init__(self, config_path: Union[str, Path]):
        self.config_path = Path(config_path)
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        self.foreach = self.config.get('foreach', [])
        self.steps = self.foreach
        self.variables = Variables()
        self.results = []
        self.export_config = None

    def run(self, llm_handler=None, verbose=False):
        """
        Run the pipeline, processing each step in order.
        llm_handler: callable for LLM steps (e.g., LLMConstrainedGenerator or similar)
        """
        for step in self.steps:
            node_type = step['node_type']
            if node_type == 'variables':
                for var_dict in step['list']:
                    self.variables.update(var_dict)
            elif node_type == 'models':
                self.models = step['list']
            elif node_type == 'template_prompt_list_of_answers':
                self.template = step['list']
                self.possible_answers = step.get('possible_answers', [])
                self.result_var_name = step.get('result_var_name', 'result')
            elif node_type == 'export_to_csv':
                self.export_config = step
            else:
                # Custom or future node types can be handled here
                pass
        # After processing all steps, run the LLM handler if provided
        if llm_handler:
            self.results = llm_handler(self.variables, self.models, self.template, self.possible_answers, self.result_var_name, verbose=verbose)
        # Export if needed
        if self.export_config:
            self.export_to_csv(self.results, self.export_config)

    def export_to_csv(self, results, export_config):
        import pandas as pd
        output_file = export_config.get('output_file', 'output.csv')
        columns = export_config.get('columns', None)
        df = pd.DataFrame(results)
        if columns:
            df = df[columns]
        df.to_csv(output_file, index=False) 