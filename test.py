# !pip install llama-cpp-python

# %% Imports
from tabnanny import verbose
import llama_cpp
import time
print("llama-cpp-python version:", llama_cpp.__version__)
from llama_cpp import Llama

# %% Load Model
# Mac needs a gguf format (I think)
# This is only a 0.5b parameter model. He's not the smartest.
llm = Llama.from_pretrained(
	repo_id="ruv/ruvltra-claude-code",
	filename="ruvltra-claude-code-0.5b-q4_k_m.gguf",
	stream = True,
	verbose=False
)
# LLM SPEED TEST
# %% Test respone1
# Simple Question.
start1 = time.perf_counter()
response1 = llm.create_chat_completion(
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
response2 = llm.create_chat_completion(
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
# Kinda dumb responses, but pretty darn quick. Let's test a bigger model (no reasoning)
