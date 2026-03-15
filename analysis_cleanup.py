# %% Imports
import pandas as pd
import matplotlib.pyplot as plt
from pandas.core.reshape.pivot import pivot_table

# %% Open (We'll get this in a better state later for file management)
benchmarks_path = "benchmarks/benchmarks_20260313_155955.csv"
benchmarks = pd.read_csv(benchmarks_path)
print(benchmarks.columns)

# Fix tokens per second calculation
benchmarks['tokens_per_second'] = benchmarks["total_tokens"] / benchmarks['chat_completion_time']

# %% Create pivot tables
benchmarks_pivot_tokens = pd.pivot_table(benchmarks, index='prompt', columns='model', values='total_tokens', sort=True)
benchmarks_pivot_time = pd.pivot(benchmarks, index='prompt', columns='model', values='chat_completion_time')
benchmarks_pivot_tps = pd.pivot(benchmarks, index='prompt', columns='model', values='tokens_per_second')

# Tokens/second pivot table

# %% Graphing pivot tables
# This is a sketchy way to plot but we'll fix that in the next project
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 6))

# Prompt Difficulty vs. Tokens
benchmarks_pivot_tokens.plot(kind='bar', ax=ax1)
ax1.set_xlabel('Prompt Difficulty')
ax1.set_ylabel('Total Tokens')
ax1.set_xticklabels(['Hard', 'Easy', 'Medium'])
ax1.set_title('Prompt Difficulty vs. Total Tokens')

# Prompt Difficulty vs. Chat Completion Time
benchmarks_pivot_time.plot(kind='bar', ax=ax2)
ax2.set_xlabel('Prompt Difficulty')
ax2.set_ylabel('Chat Completion Time')
ax2.set_xticklabels(['Hard', 'Easy', 'Medium'])
ax2.set_title('Prompt Difficulty vs. Chat Completion Time')

# Prompt Difficulty vs. Tokens/Second
# (This graph might be useless)
benchmarks_pivot_tps.plot(kind='bar', ax=ax3)
ax3.set_title('Prompt Difficulty vs. Tokens/Second')
ax3.set_xlabel('Prompt Difficulty')
ax3.set_ylabel('Tokens/Second')
ax3.set_xticklabels(['Hard', 'Easy', 'Medium'])
ax3.set_title('Prompt Difficulty vs. TPS')

plt.show()

# %% Now we should calculate ratios or something between these....
# Let's look into interpreting pivot tables before continuing
# Maybe calculate the token to chat completion time ratio in some way???? (That's literally tokens/second dummy)

# ......... Let's remove the Qwen model since I didn't even give it enough context to
#... I'm assuming... run properly

# %% Pivot table dict
pivot_tables = {
    "tokens": pd.pivot_table(benchmarks, index='prompt', columns='model', values='total_tokens', sort=True),
    "time": pd.pivot(benchmarks, index='prompt', columns='model', values='chat_completion_time'),
    "tpd": pd.pivot(benchmarks, index='prompt', columns='model', values='tokens_per_second')
}

for key, value in pivot_tables.items():
    pivot_tables[key] = pivot_tables[key].drop('8B_qwen_3', axis=1)
    value = value.drop('8B_qwen_3', axis=1)
