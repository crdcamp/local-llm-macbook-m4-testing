# %% Imports
import pandas as pd

# %% Open dat guy
benchmarks_path = "benchmarks/benchmarks_20260313_155955.csv"
benchmarks = pd.read_csv(benchmarks_path)
print(len(benchmarks))
print(benchmarks.head())

# %% Fix the tokens/second calculation
"""
This is just gonna be a rough draft for now.
"""
benchmarks["tokens_per_second"] = benchmarks["total_tokens"] / benchmarks["chat_completion_time"]
#print(benchmarks)
output_path = "benchmarks/benchmarks_test.csv"

# %% Group by model and prompt
benchmarks_grouped = benchmarks.groupby(["model", "prompt"])
#print(len(benchmarks_grouped))
if len(benchmarks) == len(benchmarks_grouped):
    print("Lengths match")
else:
    print("Lengths don't match")
