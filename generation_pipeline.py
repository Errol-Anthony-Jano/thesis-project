import json
import os
from llm_client import query_llm
from llm_config import OUTPUT_DIR
from prompts.test_generation_prompt import build_generation_prompt

def run_generation_pipeline(list):
    """Accepts a list of Python functions and their names"""
    for function in list:
        gen_prompt = build_generation_prompt(function["code"])
        test_code = query_llm(gen_prompt)
        save_test_file(function["code"], test_code, function["func_name"])


def save_test_file(focal_fn: str, test_code: str, fn_name: str):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    file_path = os.path.join(OUTPUT_DIR, f"test_{fn_name}.py")
    with open(file_path, "w") as f:
        f.write(f"# Auto-generated test suite for {fn_name}\n")
        f.write(f"{focal_fn}\n\n{test_code}\n")