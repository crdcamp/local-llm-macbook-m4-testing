# !pip install llama-cpp-python

# %% Imports
from tabnanny import verbose
import llama_cpp
print("🎉 llama-cpp-python version:", llama_cpp.__version__)
from llama_cpp import Llama

# %% Load Model
llm = Llama.from_pretrained(
	repo_id="ruv/ruvltra-claude-code",
	filename="ruvltra-claude-code-0.5b-q4_k_m.gguf",
	stream = True,
	verbose=False
)
# LLM SPEED TEST
# %% Test respone1
# Simple Question.
response1 = llm.create_chat_completion(
    messages=[
        {
            "role": "user",
            "content": "What is the capital of France?"
        }
    ]
)
print(response1["choices"][0]["message"]["content"])

# %% Test response2
# Slightly more interesting question
response2 = llm.create_chat_completion(
    messages=[
        {
            "role": "user",
            "content": "Tell me some of the most interesting theories on the beginning of the universe"
        }
    ]
)

print(response2["choices"][0]["message"]["content"])
