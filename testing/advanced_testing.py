# %% Imports
from llama_cpp import Llama
import os
import pandas as pd
import time

# %% Define folder and file structure
model_output_dir = "testing/model_outputs"
os.makedirs(model_output_dir, exist_ok=True)

# %% Define models
verbose_param = False

models = {
    "0.5B_ruvltra": Llama.from_pretrained(
        repo_id="ruv/ruvltra-claude-code",
        filename="ruvltra-claude-code-0.5b-q4_k_m.gguf",
        verbose=verbose_param
    ),
    "9B_gemma_2": Llama.from_pretrained(
        repo_id="bartowski/gemma-2-9b-it-GGUF",
        filename="gemma-2-9b-it-IQ2_M.gguf",
        verbose=verbose_param
    )
}
# Define benchmarks needed
"""
llama-bench is what we need. Here are some basic metrics:
    * tokens per second: generation speed
    * time to first token: milliseconds before output starts
    * time per prompt token: how long each input token takes to process
    * end-to-end time to first token: similar to time to first token but measures from the client side

We should record all these parameters in a dataframe and later find out if it's worth recording the prompt outputs
""";

# %% Define some simple test prompts (we'll just reuse the already created ones)
def chat_completion_benchmark(model, content: str): # -> str:
    benchmarks = pd.DataFrame()

    start_time = time.perf_counter()

    # Define time to completion
    chat_completion = model.create_chat_completion(
        messages=[
            {
                "role": "user",
                "content": content
            }
        ]
    )
    elapsed_time = time.perf_counter() - start_time

    response = chat_completion["choices"][0]["message"]["content"]

    print("Benchmark Data Frame:\n", benchmarks.head(3), "\n")
    print("Elapsed Time: ", elapsed_time, " seconds\n")
    print("Response:\n", response, "\n")

test = chat_completion_benchmark(models["0.5B_ruvltra"], "What is the capital of France?")



# %% Figure out how to use llama-bench and how to record metrics

# Small test model (from `basic_testing.py)
