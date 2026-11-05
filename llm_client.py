import requests
from llm_config import MODEL_NAME, OLLAMA_API_URL, MAX_GENERATION_TOKENS, CONTEXT_WINDOW, TIMEOUT

# Defines the abstraction that allows the user to interact with the LLM model through a simple interface.
def query_llm(prompt: str, temperature: float = 0.0) -> str:
    """
    Queries the LLM model with the given prompt and returns the response.

    Args:
        prompt (str): The input prompt to send to the LLM model.
        model_name (str): The name of the LLM model to use for querying.

    Returns:
        str: The response from the LLM model.
    """
    
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": MAX_GENERATION_TOKENS,
            "num_ctx": CONTEXT_WINDOW
        }
    }

    try:
        res = requests.post(OLLAMA_API_URL, json=payload, timeout=TIMEOUT)
        res.raise_for_status()
        raw_text = res.json().get("response", "")
        return extract_code_block(raw_text)
    except Exception as e:
        print(f"[LLM Client Error] Failed to reach Ollama: {e}")
        return ""

def extract_code_block(text: str) -> str:
    if "```python" in text:
        text = text.split("```python")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]

    return text.strip()