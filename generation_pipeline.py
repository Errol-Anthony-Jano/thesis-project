import ast
import os
from typing import Any

from llm_client import extract_code_block, query_llm
from llm_config import NUM_GENERATION_ROUNDS, OUTPUT_DIR
from prompts.test_generation_prompt import (
    SYSTEM_PROMPT,
    build_follow_up_prompt,
    build_generation_prompt,
)

from tqdm import tqdm

def run_generation_pipeline(
    functions: list[dict[str, Any]],
    num_tests: int = NUM_GENERATION_ROUNDS,
) -> list[dict[str, Any]]:
    """Generate an ordered set of tests for each function over multiple chat turns."""
    if num_tests < 1:
        raise ValueError("num_tests must be at least 1")

    for function in functions:
        fn_code = function["code"]
        fn_name = function["func_name"]
        fn_desc = function["prompt"]
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_generation_prompt(fn_code, fn_desc, fn_name),
            },
        ]
        generated_tests: list[str] = []
        test_names: set[str] = set()

        for round_number in range(num_tests):
            assistant_response = query_llm(
                fn_code,
                fn_name,
                fn_desc,
                messages=messages,
            )
            test_code = extract_code_block(assistant_response)
            if not test_code.strip():
                raise ValueError(
                    f"Ollama returned an empty test for {fn_name} "
                    f"on generation round {round_number + 1}"
                )
            try:
                parsed_test = ast.parse(test_code)
            except SyntaxError as error:
                raise ValueError(
                    f"Ollama returned invalid Python for {fn_name} "
                    f"on generation round {round_number + 1}"
                ) from error

            test_functions = [
                node.name
                for node in parsed_test.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name.startswith("test_")
            ]
            if len(test_functions) != 1:
                raise ValueError(
                    f"Ollama must return exactly one pytest test function for "
                    f"{fn_name} on generation round {round_number + 1}"
                )
            if test_functions[0] in test_names:
                raise ValueError(
                    f"Ollama repeated test function {test_functions[0]!r} "
                    f"for {fn_name} on generation round {round_number + 1}"
                )

            test_names.add(test_functions[0])
            generated_tests.append(test_code)
            messages.append({"role": "assistant", "content": assistant_response})
            if round_number < num_tests - 1:
                messages.append(
                    {
                        "role": "user",
                        "content": build_follow_up_prompt(fn_name),
                    }
                )

        function["generated_tests"] = generated_tests
        save_test_file(
            fn_code,
            "\n\n".join(generated_tests),
            fn_name,
        )

    return functions


def save_test_file(focal_fn: str, test_code: str, fn_name: str):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    file_path = os.path.join(OUTPUT_DIR, f"test_{fn_name}.py")
    with open(file_path, "w") as f:
        f.write(f"# Auto-generated test suite for {fn_name}\n")
        f.write(f"{focal_fn}\n\n{test_code}\n")