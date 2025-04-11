# LLM Constrained Output Generator

This program generates responses from OpenAI's GPT models with constrained outputs based on a predefined list of possible answers.

## Setup

### Automatic Setup

#### On Linux/Mac:
```bash
# Make the setup script executable
chmod +x setup.sh

# Run the setup script
./setup.sh
```

#### On Windows:
```bash
setup.bat
```

### Manual Setup

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

## Directory Structure

Your input directory should contain the following files:

1. `variables.csv` - Contains input variables
   - Each column represents a variable that will be replaced in the prompt template
   - Each row represents a separate prompt to generate

2. `prompt_template.txt` - Contains the prompt template
   - Use `[variable_name]` syntax for variables that should be replaced
   - Example: "His name is [name]. And his profession is:"

3. `possible_answers.txt` - Contains the list of possible answers
   - One answer per line
   - Example:
     ```
     Engineer
     Doctor
     Factory worker
     Thief
     Politician
     ```

## Output

The program will generate an `output_results.csv` file in the same directory containing:
- All original variables from the input CSV
- A new column `generated_answer` with the LLM's constrained response

## Usage

Make sure your virtual environment is activated, then run:
```bash
python llm_constrained.py
```

## Example Files

### variables.csv
```csv
name,age
John,25
Alice,30
```

### prompt_template.txt
```
His name is [name] and he is [age] years old. His profession is:
```

### possible_answers.txt
```
Engineer
Doctor
Factory worker
Thief
Politician
``` 