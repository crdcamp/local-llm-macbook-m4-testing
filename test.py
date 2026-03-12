# !pip install llama-cpp-python

# Import and test
import llama_cpp
print("🎉 llama-cpp-python version:", llama_cpp.__version__)

from llama_cpp import Llama

llm = Llama.from_pretrained(
	repo_id="ruv/ruvltra-claude-code",
	filename="ruvltra-claude-code-0.5b-q4_k_m.gguf",
)

llm.create_chat_completion(
	messages = [
		{
			"role": "user",
			"content": "What is the capital of France?"
		}
	]
)
