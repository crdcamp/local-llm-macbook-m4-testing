# %% Imports
from llama_cpp import Llama
import os
import time
import json
from datetime import datetime
from numpy.typing import test
import pandas as pd

# Hugging Face Search Parameters: https://huggingface.co/models?pipeline_tag=text-generation&num_parameters=min:9B,max:12B&library=gguf&apps=llama.cpp&sort=downloads
# Models can be deleted in ~/.cache/huggingface/hub/

# %% Define folder and file structure
benchmark_dir = "benchmarks"
os.makedirs(benchmark_dir, exist_ok=True)

# %% Define models
verbose_param = False
context_window = 2048

models = {
    "0.5B_ruvltra": Llama.from_pretrained(
        repo_id="ruv/ruvltra-claude-code",
        filename="ruvltra-claude-code-0.5b-q4_k_m.gguf",
        verbose=verbose_param,
        n_ctx=context_window
    ),
    "9B_glm_4": Llama.from_pretrained(
        repo_id="bartowski/glm-4-9b-chat-1m-GGUF",
        filename="glm-4-9b-chat-1m-IQ2_M.gguf",
        verbose=verbose_param,
        n_ctx=context_window
    ),
    "9B_gemma_2": Llama.from_pretrained(
        repo_id="bartowski/gemma-2-9b-it-GGUF",
        filename="gemma-2-9b-it-IQ2_M.gguf",
        verbose=verbose_param,
        n_ctx=context_window
    ),
    "12B_gemma_3": Llama.from_pretrained(
        repo_id="MaziyarPanahi/gemma-3-12b-it-GGUF",
        filename="gemma-3-12b-it.Q2_K.gguf",
        verbose=verbose_param,
        n_ctx=context_window
    ),
    "9B_qwen_3.5_unsloth": Llama.from_pretrained(
        repo_id="Qwen/Qwen3-8B-GGUF",
        filename="Qwen3-8B-Q4_K_M.gguf",
        verbose=verbose_param,
        n_ctx=context_window
    ),
    # Takes too long. Might try another time
    # "8B_deepseek_unsloth": Llama.from_pretrained(
    #     repo_id="unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF",
    #     filename="DeepSeek-R1-0528-Qwen3-8B-BF16.gguf",
    #     verbose=verbose_param,
    #     n_ctx=context_window
    # ),
    "7B_deepseek_chat_second_state": Llama.from_pretrained(
        repo_id="second-state/Deepseek-LLM-7B-Chat-GGUF",
        filename="deepseek-llm-7b-chat-Q2_K.gguf",
        verbose=verbose_param,
        n_ctx=context_window
    )
}

print()

# %% Chat completions benchmarks function
def chat_completion_benchmark(model: str, content: str):
    # Add check for GGUF format
    print("Current model: ", model)
    print("Prompt: ", content)
    model_object = models[model]

    start_time = time.perf_counter()

    print("Creating chat completion...")
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
    print("Response: ", response)

    usage = chat_completion["usage"]
    tps = usage["completion_tokens"] / elapsed_time # Tokens per second: Double check if you should use `total_tokens` instead
    print("Tokens per second: ", tps)
    print("Total processing time: ", elapsed_time, "\n\n")

    results = pd.DataFrame({"model": [model],
        "elapsed_time": [elapsed_time],
        "tokens_per_second": tps,
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": usage["completion_tokens"],
        "total_tokens": usage["total_tokens"],
        "prompt": content,
        "response": response})


    return results

print()

benchmarks = pd.DataFrame()
test1 = chat_completion_benchmark("0.5B_ruvltra", "What is the capital of France?")
test2 = chat_completion_benchmark("0.5B_ruvltra", "What color is the sky?")
benchmarks = pd.concat([test1, test2])
print(benchmarks)

# %% Call function for easy, medium, and hard prompts
print("Running models and gathering benchmarks... (this will take a while)")
print()
for model in models:
    easy_prompt = chat_completion_benchmark(model, "What is the capital of France?")
    medium_prompt = chat_completion_benchmark(model, "Summarize the main arguments for and against nuclear energy as a solution to climate change.")
    hard_prompt = chat_completion_benchmark(model, "Compare the epistemological foundations of Bayesian and frequentist statistics. Where do they genuinely disagree, and where is the disagreement mostly philosophical?")

benchmarks.sort(key=lambda x: x["tokens_per_second"], reverse=True)
print("Benchmark Results:\n", json.dumps(benchmarks, indent=2))

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = os.path.join(benchmark_dir, f"benchmarks_{timestamp}.json")

with open(output_path, "w") as f:
    json.dump(benchmarks, f, indent=2)
print("Done. Benchmarks have been saved to ", benchmark_dir)
print("Total models tested: ", len(models))
