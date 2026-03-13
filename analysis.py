# %% Imports
import pandas as pd
import matplotlib.pyplot as plt

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

# %%
model_names = benchmarks["model"].unique()
print(type(model_names))
print(model_names)

model_ruv_name = model_names[0]
print(model_ruv)
print()

# Absolutely horrenduos way to do this
benchmarks_ruv = benchmarks[benchmarks["model"] == ]
benchmarks_8_qwen = benchmarks[benchmarks["model"] == model_ruv_name]
# benchmarks_9_glm =
# benchmarks_9_gemma =
# benchmarks_deep =
