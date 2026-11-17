MODEL_NAME = "deepseek-coder:1.3b"
MAX_REPAIR_TURNS = 3
SANDBOX_TIMEOUT = 30
OUTPUT_DIR = "generated_tests"
OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_CHAT_API = "http://localhost:11434/api/chat"
MAX_GENERATION_TOKENS = 1024
CONTEXT_WINDOW = 4096
TIMEOUT = 300 # seconds
NUM_GENERATION_ROUNDS = 10

QWEN_MODEL = "qwen2.5-coder:1.5b"
QWEN_MAX_GT = 256
QWEN_CTX_WDW = 2048

QWEN_3B_MODEL = "qwen2.5-coder:3b-instruct"

SYNTAX_FIX_MAX_TRIES = 5