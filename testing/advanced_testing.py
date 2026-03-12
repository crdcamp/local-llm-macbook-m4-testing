# %% Imports
from llama_cpp import Llama
import os
import time

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

# %% Chat completions benchmarks
benchmarks = []

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
        ],
        stream=False
    )
    elapsed_time = time.perf_counter() - start_time

    response = chat_completion["choices"][0]["message"]["content"]
    usage = chat_completion["usage"]
    tps = usage["completion_tokens"] / elapsed_time # Tokens per second

    results = {
        "model": model,
        "elapsed_time": elapsed_time,
        "tokens_per_second": tps,
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": usage["completion_tokens"],
        "total_tokens": usage["total_tokens"],
        "prompt": content,
        "response": response
    }

    benchmarks.append(results)
    return results


print()
test = chat_completion_benchmark("0.5B_ruvltra", "What is the capital of France?")
