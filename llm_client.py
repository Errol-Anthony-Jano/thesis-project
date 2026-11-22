import requests
import re
from llm_config import (
    CONTEXT_WINDOW,
    MAX_GENERATION_TOKENS,
    OLLAMA_CHAT_API,
    QWEN_3B_MODEL,
    TIMEOUT,
)
from prompts.test_generation_prompt import SYSTEM_PROMPT, build_generation_prompt

# Defines the abstraction that allows the user to interact with the LLM model through a simple interface.


def query_llm(
    fn_code: str,
    fn_name: str,
    fn_desc: str,
    temperature: float = 0.0,
    *,
    messages: list[dict[str, str]] | None = None,
    max_generation_tokens: int | None = MAX_GENERATION_TOKENS,
) -> str:
    """
    Query Ollama's chat endpoint and return the complete assistant response.
    """

    if messages is None:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_generation_prompt(fn_code, fn_desc, fn_name),
            },
        ]

    payload = {
        "model": QWEN_3B_MODEL,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": (
                max_generation_tokens if max_generation_tokens is not None else -1
            ),
            "num_ctx": CONTEXT_WINDOW,
        },
    }

    response = requests.post(OLLAMA_CHAT_API, json=payload, timeout=TIMEOUT)
    response.raise_for_status()
    response_data = response.json()
    if not isinstance(response_data, dict):
        raise ValueError("Unexpected Ollama response shape: expected a JSON object")

    message = response_data.get("message")
    if isinstance(message, dict) and isinstance(message.get("content"), str):
        return message["content"]
    if isinstance(response_data.get("response"), str):
        return response_data["response"]
    raise ValueError("Unexpected Ollama response shape: expected message.content")

def query_llm_fix_code(
    messages,
    temperature=0.0,
    max_generation_tokens: int | None = MAX_GENERATION_TOKENS,
):
    payload = {
        "model": QWEN_3B_MODEL,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": (
                max_generation_tokens if max_generation_tokens is not None else -1
            ),
            "num_ctx": CONTEXT_WINDOW,
        },
    }

    response = requests.post(OLLAMA_CHAT_API, json=payload, timeout=TIMEOUT)
    response.raise_for_status()
    response_data = response.json()
    if not isinstance(response_data, dict):
        raise ValueError("Unexpected Ollama response shape: expected a JSON object")

    message = response_data.get("message")
    if isinstance(message, dict) and isinstance(message.get("content"), str):
        return message["content"]
    if isinstance(response_data.get("response"), str):
        return response_data["response"]
    raise ValueError("Unexpected Ollama response shape: expected message.content")

def extract_code_block(text: str) -> str:
    fenced_code = re.search(
        r"```(?:python|py)?[ \t]*\r?\n(.*?)```",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if fenced_code:
        text = fenced_code.group(1)

    return text.strip()