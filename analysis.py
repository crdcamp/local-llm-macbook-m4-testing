# %% Imports
import os
import json
import pandas as pd

# %% Open dat guy
benchmark_dir = os.listdir("benchmarks")
if len(benchmark_dir) == 0:
    print("No benchmarks found.")
else:
    print(benchmark_dir)
