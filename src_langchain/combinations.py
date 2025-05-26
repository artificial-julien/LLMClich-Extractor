from langchain.schema.runnable import RunnableMap, RunnableSequence
from langchain.prompts import PromptTemplate
from typing import List

def generate_combinations(origins: List[str], counts: List[str]) -> List[str]:
    """
    Generate all combinations of origins and counts using LangChain.
    
    Args:
        origins: List of origin strings (e.g. ["french", "german"])
        counts: List of count strings (e.g. ["1", "1000"])
        
    Returns:
        List of combined strings in the format "{count} {origin} people"
    """
    # Create a RunnableMap to generate all combinations
    variable_generator = RunnableMap({
        "origin": lambda x: origins,
        "count": lambda x: counts
    })

    # Create the template for combining origin and count
    combination_template = PromptTemplate.from_template("{count} {origin} people")

    # Create the chain
    chain = (
        variable_generator
        | combination_template
    )

    # Generate all combinations
    combinations = []
    for origin in origins:
        for count in counts:
            result = chain.invoke({"origin": origin, "count": count})
            combinations.append(result)

    return combinations

if __name__ == "__main__":
    # Example usage
    origins = ["french", "german", "spanish", "italian"]
    counts = ["1", "1000"]
    
    combinations = generate_combinations(origins, counts)
    for combo in combinations:
        print(combo) 