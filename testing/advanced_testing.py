# %% Imports
from llama_cpp import Llama
import os
import time
import subprocess
import pandas as pd

# Hugging Face Search Parameters: https://huggingface.co/models?pipeline_tag=text-generation&num_parameters=min:9B,max:12B&library=gguf&apps=llama.cpp&sort=trending

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
print()
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
# llama-bench resource: https://github.com/ggml-org/llama.cpp/tree/master/tools/llama-bench#syntax
def chat_completion_benchmark(model: str, content: str): # -> str:
    # Define model parameters for shell commands
    model_object = models[model]
    model_filename = os.path.basename(model_object.model_path)

    start_time = time.perf_counter()
    # Define time to completion
    chat_completion = model_object.create_chat_completion(
        messages=[
            {
                "role": "user",
                "content": content
            }
        ]
    )

    # Temporary testing for storing results
    #benchmarks = pd.DataFrame()
    elapsed_time = time.perf_counter() - start_time
    response = chat_completion["choices"][0]["message"]["content"]

    # Get llama-bench results
    usage = chat_completion["usage"]
    tokens_generated = usage["completion_tokens"]
    tps = tokens_generated / elapsed_time

    print("Model: ", model)
    print("Tokens per second: ", tps, "\n")
    #print("Benchmark Data Frame:\n", benchmarks.head(3), "\n")
    print("Elapsed Time: ", elapsed_time, " seconds\n")
    print("Response:\n", response, "\n")

test = chat_completion_benchmark("0.5B_ruvltra", "What is the capital of France?")
