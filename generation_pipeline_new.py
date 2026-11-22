"""
GENERATION PIPELINE

input: 
- function
- explanation
- max-tests = 10 by default

"""

import ast
import os
from tqdm import tqdm
import tempfile

from llm_client import extract_code_block, query_llm, query_llm_fix_code
from llm_config import (
    MAX_GENERATION_TOKENS,
    NUM_GENERATION_ROUNDS,
    OUTPUT_DIR,
    SYNTAX_FIX_MAX_TRIES,
)
from prompts.test_generation_prompt import (
    SINGLE_GENERATION_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    build_follow_up_prompt,
    build_generation_prompt,
    build_single_generation_prompt,
)
from prompts.syntax_fix_prompt import repair_code


def run_generation_pipeline(
    functions,
    num_tests=NUM_GENERATION_ROUNDS,
    *,
    single_generation=False,
):
    if num_tests < 1:
        raise ValueError("num_tests must be at least 1")

    for function in tqdm(functions):
        function_code = function["code"]
        function_name = function["func_name"]
        function_desc = function["prompt"]
        generated_tests = []
        test_names = set()

        messages = [
            {
                "role": "system",
                "content": (
                    SINGLE_GENERATION_SYSTEM_PROMPT
                    if single_generation
                    else SYSTEM_PROMPT
                ),
            },
            {
                "role": "user",
                "content": (
                    build_single_generation_prompt(
                        function_code, function_desc, function_name
                    )
                    if single_generation
                    else build_generation_prompt(
                        function_code, function_desc, function_name
                    )
                ),
            },
        ]

        generation_rounds = 1 if single_generation else num_tests
        for round_number in tqdm(range(generation_rounds)):
            print(f"Generating test #{round_number} for function {function_name}")

            if single_generation:
                llm_response = query_llm(
                    function_code,
                    function_name,
                    function_desc,
                    messages=messages,
                    max_generation_tokens=None,
                )
            else:
                llm_response = query_llm(
                    function_code,
                    function_name,
                    function_desc,
                    messages=messages,
                )
            test_code = extract_code_block(llm_response)
            if not test_code:
                raise ValueError(
                    f"Ollama returned an empty test for {function_name} "
                    f"on generation round {round_number + 1}"
                )

            try:
                parsed_test = ast.parse(test_code)
            except SyntaxError as error:
                print(f"Syntax error by generated test for {function_name}. Entering repair loop.")
                test_code = run_syntax_error_fix_loop(
                    test_code,
                    SYNTAX_FIX_MAX_TRIES,
                    error,
                    max_generation_tokens=(
                        None if single_generation else MAX_GENERATION_TOKENS
                    ),
                )
                parsed_test = ast.parse(test_code)

            test_functions = [
                node.name
                for node in parsed_test.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name.startswith("test_")
            ]
            if single_generation:
                if not test_functions:
                    raise ValueError(
                        f"Ollama must return at least one pytest test function "
                        f"for {function_name} on generation round "
                        f"{round_number + 1}"
                    )
                if len(test_functions) != len(set(test_functions)):
                    raise ValueError(
                        f"Ollama repeated a test function name for "
                        f"{function_name} on generation round "
                        f"{round_number + 1}"
                    )
            elif len(test_functions) != 1:
                raise ValueError(
                    f"Ollama must return exactly one pytest test function for "
                    f"{function_name} on generation round {round_number + 1}"
                )

            test_names.add(test_functions[0])
            generated_tests.append(test_code)
            messages.append({"role": "assistant", "content": test_code})

            if round_number < num_tests - 1:
                messages.append(
                    {
                        "role": "user",
                        "content": build_follow_up_prompt(function_name),
                    }
                )

        function["generated_tests"] = generated_tests

        save_source_file(function_code, function_name)
        save_test_file("\n\n".join(generated_tests), function_name)
        print(f"Successfully wrote tests to test_{function_name}.py")

    return functions


def run_syntax_error_fix_loop(
    function,
    num_tries,
    error,
    max_generation_tokens: int | None = MAX_GENERATION_TOKENS,
):
    fixed_test = function
    fix_error = error
    for _ in tqdm(range(num_tries)):
        messages = [
            {
                "role": "system",
                "content": (
                    "You are repairing a Python pytest test. Return only the "
                    "complete corrected Python code, with no explanation or "
                    "Markdown fences."
                ),
            },
            {
                "role": "user",
                "content": repair_code(fixed_test, fix_error),
            },
        ]

        fixed_test = extract_code_block(
            query_llm_fix_code(
                messages,
                0.0,
                max_generation_tokens=max_generation_tokens,
            )
        )
        if not fixed_test:
            continue

        try:
            ast.parse(fixed_test)
        except SyntaxError as current_error:
            fix_error = current_error
            continue

        return fixed_test

    raise ValueError(
        f"Could not repair Python syntax after {num_tries} attempts"
    ) from fix_error

def save_source_file(focal_fn: str, fn_name: str):
    source_dir = "src"
    os.makedirs(source_dir, exist_ok=True)

    init_path = os.path.join(source_dir, "__init__.py")
    if not os.path.exists(init_path):
        with open(init_path, "w") as f:
            f.write("")

    file_path = os.path.join(source_dir, f"{fn_name}.py")
    with open(file_path, "w") as f:
        f.write(focal_fn.rstrip() + "\n")


def save_test_file(test_code: str, fn_name: str):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    file_path = os.path.join(OUTPUT_DIR, f"test_{fn_name}.py")
    with open(file_path, "w") as f:
        f.write(f"# Auto-generated test suite for {fn_name}\n")
        f.write(f"from src.{fn_name} import {fn_name}\n\n")
        f.write(f"{test_code}\n")

def run_test_fix_loop():
    return True

def stringify_functions(function_list):
    return "\n".join(function_list)