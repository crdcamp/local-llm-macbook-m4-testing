# %% Imports
from socket import INADDR_MAX_LOCAL_GROUP
import pandas as pd
import matplotlib.pyplot as plt

"""
OMG THIS CODE IS SUCH A MESS IT DOESN'T MATTER FOR THIS PART THOUGH
"""

# %% Open dat guy and prepare the data

benchmarks_path = "benchmarks/benchmarks_20260313_155955.csv"
benchmarks = pd.read_csv(benchmarks_path)

print(benchmarks.columns)

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

"""
To Do:
    * Order the graph below
    * Time vs. Token usage
    *
"""
benchmarks_pivot_token = pd.pivot_table(benchmarks, index='prompt', columns='model', values='total_tokens', sort=True)

# I'll figure out how to rename the prompt columns (or just rename them up above)
fig, ax = plt.subplots()

benchmarks_pivot_token.plot(kind='bar', ax=ax)
ax.set_xlabel("Prompt Difficulty")
ax.set_xticklabels([])
ax.set_ylabel("Total Tokens")
ax.set_title("Prompt Difficulty vs. Total Tokens")

plt.show();


# %% Mmmmkay... what else can we do? Check out chat completion time?
print(benchmarks.columns)
benchmarks_pivot_time = pd.pivot_table(benchmarks, index='prompt', columns='model', values='chat_completion_time')
fig, ax = plt.subplots()

benchmarks_pivot_time.plot(kind='bar', ax=ax)
ax.set_xlabel("Prompt Difficulty")
ax.set_xticklabels([])
ax.set_ylabel("Completion Time")
ax.set_title('Prompt Difficulty vs. Completion Time')

plt.show();

# %% Now let's find out the min and max values from these 2 pivot tables
# print("Token Pivot Table Columns:\n", benchmarks_pivot_token.columns, "\n\n")
# print("Completion Time Pivot Table Columns:\n", benchmarks_pivot_token.columns, "\n\n")

output_path_token = 'pivot_tables/pivot_token.csv'
output_path_time = 'pivot_tables/pivot_time.csv'

# For loading values into excel
#benchmarks_pivot_token.to_csv(output_path_token)
#benchmarks_pivot_time.to_csv(output_path_time)

# Find max and min for easy, difficult, and hard prompts
pivot_tables = [benchmarks_pivot_token, benchmarks_pivot_time]

# Change the index values so we're not dealing with annoyingly long prompts
# Then, apply max and min stuff
for table in pivot_tables:
    table.index = ['hard', 'medium', 'easy']
    max_values = table.max(axis=1)
    max_model = table.idxmax(axis=1)  # this IS the model name
    print(pd.DataFrame({'max_value': max_values, 'model': max_model}))
