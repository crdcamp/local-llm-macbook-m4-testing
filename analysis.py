# %% Imports
import pandas as pd

# %% Open dat guy
benchmarks_path = "benchmarks/benchmarks_20260313_155955.csv"
benchmarks = pd.read_csv(benchmarks_path)
print(benchmarks.head(2))

# %% Fix the tokens/second calculation
"""
This is just gonna be a rough draft for now.
"""
benchmarks["tokens_per_second"] = benchmarks["total_tokens"] / benchmarks["chat_completion_time"]
#print(benchmarks)
output_path = "benchmarks/benchmarks_test.csv"

# %% Group by model
prompt_group = benchmarks.groupby(["model", "prompt"])
#print(prompt_group.head())
output_path = "testing/test.csv"


# Only saves the "Captial of France" prompt
print(prompt_group.head())
