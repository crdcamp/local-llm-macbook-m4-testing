# %% Imports
from llama_cpp import Llama
import os
import time
import json
from datetime import datetime

# Hugging Face Search Parameters: https://huggingface.co/models?pipeline_tag=text-generation&num_parameters=min:9B,max:12B&library=gguf&apps=llama.cpp&sort=downloads

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
	repo_id="unsloth/Qwen3.5-9B-GGUF",
	filename="Qwen3.5-9B-BF16.gguf",
	verbose=verbose_param,
    n_ctx=context_window
    )
    # Add deepseek
}


print()

# %% Chat completions benchmarks function
benchmarks = []

def chat_completion_benchmark(model: str, content: str):
    # Add check for GGUF format
    model_object = models[model]
    model_filename = os.path.basename(model_object.model_path)

    start_time = time.perf_counter()
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

# %% Call function for easy, medium, and hard prompts
print("Running models and gathering benchmarks...")
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
print("Models are done and benchmarks have been saved")
