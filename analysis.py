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
# Just kinda messing around for now... This is just testing

# Bar chart for token usage
# Show token usage for each prompt
fig, ax = plt.subplots()
total_tokens = benchmarks_8_qwen["total_tokens"]
prompts = benchmarks_8_qwen["prompt"].unique() # Already in order from easy to hard. Add logic for this in future use case

ax.bar(prompts, total_tokens)

ax.set_title("Qwen 8B - Prompt Difficulty vs. Token Usage")
ax.set_ylabel("Total Tokens")
ax.set_xlabel("Prompt Difficulty")
ax.set_xticks(range(len(prompts)))
ax.set_xticklabels(["Easy", "Medium", "Hard"])

plt.show();

# Line chart for tokens vs. completion time

# %% Now aggregate the bar charts (ahhhh I was actually looking for a pivot table)
fig, ax = plt.subplots()
benchmarks.pivot_table(index="prompt", columns="model", values="total_tokens").plot(kind="bar", ax=ax)
ax.set_xlabel("")
ax.set_ylabel("Total Tokens")
plt.tight_layout()
plt.show()

# %%
print(prompts)
