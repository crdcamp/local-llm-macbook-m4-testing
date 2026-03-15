# %% Imports
import pandas as pd
import matplotlib.pyplot as plt

# %% Open (We'll get this in a better state later for file management)
benchmarks_path = "benchmarks/benchmarks_20260313_155955.csv"
benchmarks = pd.read_csv(benchmarks_path)

# Fix tokens per second calculation
benchmarks['tokens_per_second'] = benchmarks["total_tokens"] / benchmarks['chat_completion_time']

# Create "easy, medium", and "hard" prompt assignments
# We'll move the prompts into a more centralized space another time
# We'll also make sure to add a difficult parameter in the original function to avoid this mess
def classify_difficulty(prompt):
    if prompt.startswith("What is the capital"):
        return "easy"
    elif prompt.startswith("Summarize the main arguments"):
        return "medium"
    elif prompt.startswith("Compare the epistemological foundations of Bayesian"):
        return "hard"
    else:
        return None

benchmarks['prompt_difficulty'] = benchmarks['prompt'].apply(classify_difficulty)

print(benchmarks.head())
