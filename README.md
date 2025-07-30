# LLM Constrained Output Generator

This program generates responses from OpenAI's GPT models with constrained outputs based on a predefined list of possible answers.

## Setup


1. Create and activate a virtual environment:
```bash
# Create virtual environment
python -m venv venv

# Activate on Linux/Mac
source venv/bin/activate

# Activate on Windows
venv\Scripts\activate.bat
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file from the example:
```bash
cp .env.example .env
```

4. Edit the `.env` file and add your OpenAI API key:
```
OPENAI_API_KEY='your-api-key-here'
```

## Usage

```bash
python <pipeline file> [--parallel <number>] [--batch-seed <number>] [--csv-append] [--llm-max-retries <number>] [--llm-seed <number>] [--output-dir <path>] [--verbose]
```

`--parallel <number>`: Number of parallel requests to run simultaneously (default: 1). Higher values can speed up execution but may be rate-limited by your LLM provider.

`--batch-seed <number>`: Optional seed for batch generation reproducibility.

`--csv-append`: Append to existing CSV files instead of overwriting.

`--llm-max-retries <number>`: Maximum number of retries for failed LLM calls (default: 1).

`--llm-seed <number>`: Optional seed for LLM calls reproducibility.

`--output-dir <path>`: Base directory for output files (default: "output").

`--verbose`: Enable verbose logging.

Example
```bash
python "examples/elo_rating/funniest_fictional_character/funniest_fictional_character.py"
```

Alternatively if you want a faster execution and a complete determinism:
```bash
python "examples/elo_rating/funniest_fictional_character/funniest_fictional_character.py" --parallel 8 --batch-seed 0
```
