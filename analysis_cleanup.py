# %% Imports
import pandas as pd
import matplotlib.pyplot as plt
import re

# %% Open (We'll get this in a better state later for file management)
benchmarks_path = "benchmarks/benchmarks_20260313_155955.csv"
benchmarks = pd.read_csv(benchmarks_path)

# Fix tokens per second calculation
benchmarks['tokens_per_second'] = benchmarks["total_tokens"] / benchmarks['chat_completion_time']
print(benchmarks.columns)

# Create "easy, medium", and "hard" prompt assignments
# We'll move the prompts into a more centralized space another time
prompts = {
    'easy_prompt': "What is the capital of France?",
    'medium_prompt': "Summarize the main arguments for and against nuclear energy as a solution to climate change.",
    'hard_prompt': "Compare the epistemological foundations of Bayesian and frequentist statistics. Where do they genuinely disagree, and where is the disagreement mostly philosophical?"
}

def classify_difficulty(prompt):
    if prompt.startswith("What is the capital"):
        return "easy"
    elif prompt.startswith("Summarize the main arguments"):
        return "medium"
    else:
        return "hard"

benchmarks['prompt_difficulty'] = benchmarks['prompt'].apply(classify_difficulty)
print(benchmarks['prompt_difficulty'])
