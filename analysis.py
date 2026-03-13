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
#print(model_names)

i = 0
for model in model_names:
    model_name = model_names[i]
    #print(model_name)

    i += 1

# Absolutely horrendous way to do this. I'll get it right later
benchmarks_ruv = benchmarks[benchmarks["model"] == "0.5B_ruvltra"]
benchmarks_8_qwen = benchmarks[benchmarks["model"] == "8B_qwen_3"]
benchmarks_9_glm = benchmarks[benchmarks["model"] == "9B_glm_4"]
benchmarks_9_gemma = benchmarks[benchmarks["model"] == "9B_gemma_2"]
benchmarks_deep = benchmarks[benchmarks["model"] == "7B_deepseek_chat_second_state"]

output_path = "testing/ruv_results.csv"
benchmarks_ruv.to_csv(output_path, index=False)

# %% Now graph each of the results (gonna be pretty basic to start)
