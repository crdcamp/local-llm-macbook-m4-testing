# %% Imports
import pandas as pd
import matplotlib.pyplot as plt

# %% Open (We'll get this in a better state later for file management)
benchmarks_path = "benchmarks/benchmarks_20260313_155955.csv"
benchmarks = pd.read_csv(benchmarks_path)
print(benchmarks.columns)

# Fix tokens per second calculation
benchmarks['tokens_per_second'] = benchmarks["total_tokens"] / benchmarks['chat_completion_time']

# Create "easy, medium", and "hard" prompt assignments
# We'll move the prompts into a more centralized space another time

# %% Pivot Tables
benchmarks_pivot_tokens = pd.pivot_table(benchmarks, index='prompt', columns='model', values='total_tokens', sort=True)
benchmarks_pivot_time = pd.pivot(benchmarks, index='prompt', columns='model', values='chat_completion_time')


# This is a sketchy way to plot but we'll fix that in the next project
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

benchmarks_pivot_tokens.plot(kind='bar', ax=ax1)
ax1.set_xlabel('Prompt Difficulty')
ax1.set_ylabel('Total Tokens')

ax1.set_xticklabels(['Hard', 'Easy', 'Medium'])

ax1.set_title('Prompt Difficulty vs. Total Tokens')

benchmarks_pivot_time.plot(kind='bar', ax=ax2)
ax2.set_xlabel('Prompt Difficulty')
ax2.set_ylabel('Chat Completion Time')

ax2.set_xticklabels(['Hard', 'Easy', 'Medium'])

ax2.set_title('Prompt Difficulty vs. Chat Completion Time')
plt.show()
