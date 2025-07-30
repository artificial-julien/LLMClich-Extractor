# LLMsClichés: Framework for Analyzing Clichés in Large Language Models

LLMsClichés is a toolkit for poking and prodding large language models to see what clichés and stereotypes they fall back on. It lets you set up experiments where the model has to pick from a list, rank things in head-to-head matchups, or show how close different ideas are in its "mind" using embeddings. T this project helps you dig into what LLMs really think is typical or obvious.
Great for anyone curious about the patterns and quirks hiding in AI-generated text.

See [output](output) for some result and [examples](examples).

## Evaluation Techniques

### Elo Rating for Ranking
Pairwise comparisons judged by the LLM to rank items via an Elo rating system. The LLM acts as a judge in simulated "matches," updating scores based on wins/losses/draws to create ordered lists by perceived attributes.

### Embedding Matrix of Distances
Generates semantic embeddings for sets of items and computes distance matrices to measure similarities. This reveals how closely the LLM associates different concepts in its embedding space, highlighting potential clichéd or biased groupings.

## Constraint Techniques
The framework employs multiple strategies to constrain the model's responses to a predefined set of choices.

### Structured Responses
Using OpenAI's JSON schema mode to constrain LLMs to output one of a predefined list of answers. This forces explicit choices, revealing preferences or biases in scenarios like dilemmas or rankings.
*Work 100% of the time but is not supported by many LLMs providers.*

### Prompt Engineering
Using carefully crafted prompts to guide and constrain LLM outputs into specific formats or choices. This provides control over response structure while allowing analysis of the model's preferences and biases.
*Can be used on any LLM but is not completely reliable*

### Function calling
Using function calling to constrain LLM outputs into specific formats. This provides a structured way to get responses in a predefined format while allowing analysis of the model's preferences and biases. Function calling is supported by most LLM providers and is more reliable than prompt engineering but less constrained than JSON schema mode.
*Not supported yet*

## Setup

```bash
# Create and activate a virtual environment:
python -m venv venv

# Activate on Linux/Mac/WSL
source venv/bin/activate

# Install the required dependencies:
pip install -r requirements.txt

# Create a `.env` file from the example:
cp .env.example .env
```

Edit the `.env` file and add your OpenAI API key:
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

