# !pip install llama-cpp-python

# %% Imports
from llama_cpp import Llama
import time

# %% Load Model (0.5B)
# Mac needs a gguf format (I think)
# This is only a 0.5b parameter model. He's not the smartest. No reasoning.
# We might mess around with reasoning models, but likely unnecessary (hopefully). Too computationally expensive
llm05B = Llama.from_pretrained(
	repo_id="ruv/ruvltra-claude-code",
	filename="ruvltra-claude-code-0.5b-q4_k_m.gguf",
	verbose=False
)


# %% Test response1
# Simple Question.
start1 = time.perf_counter()
response1 = llm05B.create_chat_completion(
    messages=[
        {
            "role": "user",
            "content": "What is the capital of France?"
        }
    ]
)
elapsed1 = time.perf_counter() - start1
print(response1["choices"][0]["message"]["content"])
print(f"\nresponse1 took {elapsed1:.3f}s")

# %% Test response2
# Slightly more interesting question
start2 = time.perf_counter()
response2 = llm05B.create_chat_completion(
    messages=[
        {
            "role": "user",
            "content": "Tell me some of the most interesting theories on the beginning of the universe"
        }
    ]
)
elapsed2 = time.perf_counter() - start2
print(response2["choices"][0]["message"]["content"])
print(f"\nresponse2 took {elapsed2:.3f}s")

# Kinda dumb response for `response2`, but relatively quick. Let's test a bigger model (still no reasoning)

# %% Loading in a bigger model (9B)

llm9B = Llama.from_pretrained(
	repo_id="bartowski/gemma-2-9b-it-GGUF",
	filename="gemma-2-9b-it-IQ2_M.gguf",
)

# %% Test response1
start1 = time.perf_counter()
response1 = llm9B.create_chat_completion(
    messages=[
        {
            "role": "user",
            "content": "What is the capital of France?"
        }
    ]
)
elapsed1 = time.perf_counter() - start1
print(response1["choices"][0]["message"]["content"])
print(f"\nresponse1 took {elapsed1:.3f}s")

# %% Test response2
# Slightly more interesting question
start2 = time.perf_counter()
response2 = llm9B.create_chat_completion(
    messages=[
        {
            "role": "user",
            "content": "Tell me some of the most interesting theories on the beginning of the universe"
        }
    ]
)
elapsed2 = time.perf_counter() - start2
print(response2["choices"][0]["message"]["content"])
print(f"\nresponse2 took {elapsed2:.3f}s")

# %% Create function for recording execution time
