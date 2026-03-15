# %% Imports
import pandas as pd
import matplotlib.pyplot as plt

from analysis import benchmarks_pivot_token

# %% Open
benchmarks_path = "benchmarks/benchmarks_20260313_155955.csv"
benchmarks = pd.read_csv(benchmarks_path)

# Fix tokens per second calculation
benchmarks['tokens_per_second'] = benchmarks["total_tokens"] / benchmarks['chat_completion_time']

# Create "easy, medium", and "hard" prompt assignments

print(benchmarks.head(3))

# %& Pivot Tables
#benchmarks_pivot_token = pd.pivot_table(benchmarks, index='prompt', columns='model', values='total_tokens', sort=True)
