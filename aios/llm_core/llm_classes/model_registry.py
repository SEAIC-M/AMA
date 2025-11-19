# registering all proprietary llm models in a constant

from .gpt_llm import GPTLLM
from .gemini_llm import GeminiLLM
from .claude_llm import ClaudeLLM

#used for closed LLM model registry
MODEL_REGISTRY = {
    # Gemini-1.5
    "gemini-1.5-flash": GeminiLLM,
    "gemini-1.5-pro": GeminiLLM,

    # GPT3.5
    'gpt-3.5-turbo': GPTLLM,
    'gpt-4-turbo': GPTLLM,

    # GPT4o
    'gpt-4o': GPTLLM,
    'gpt-4o-2024-08-06': GPTLLM,
    'gpt-4o-mini': GPTLLM,


    "qwen2.5-instruct-32b": GPTLLM,
    "gemma-3-it": GPTLLM,
    'llama-3.1-instruct':GPTLLM,
    'llama-3.3-instruct':GPTLLM,
    "qwen3": GPTLLM,
    "deepseek-r1-distill-qwen": GPTLLM,


    # claude
    'claude-3-5-sonnet-20240620': ClaudeLLM,
}
