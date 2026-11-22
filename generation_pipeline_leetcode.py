import ast
import os
from typing import Any

from tqdm import tqdm

from llm_client import extract_code_block, query_llm
from llm_config import NUM_GENERATION_ROUNDS, OUTPUT_DIR
from prompts.test_generation_prompt import (
    SINGLE_GENERATION_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    build_follow_up_prompt,
    build_generation_prompt,
    build_single_generation_prompt,
)


def run_leetcode_generation_pipeline(
    tasks: list[dict[str, Any]],
    num_tests: int = NUM_GENERATION_ROUNDS,
    *,
    single_generation: bool = False,
) -> list[dict[str, Any]]:
    if num_tests < 1:
        raise ValueError("num_tests must be at least 1")

    output_paths: set[str] = set()
    for task in tqdm(tasks):
        task_num = task.get("task_num")
        function_name = task.get("func_name")
        function_code = task.get("python_solution")
        function_description = task.get("description")

        if not isinstance(task_num, int) or isinstance(task_num, bool):
            raise ValueError("Each LeetCode task must have an integer task_num")
        if not isinstance(function_name, str) or not function_name.isidentifier():
            raise ValueError(f"Task {task_num} must have a valid func_name")
        if not isinstance(function_code, str) or not function_code.strip():
            raise ValueError(f"Task {task_num} must have a non-empty python_solution")
        if not isinstance(function_description, str):
            raise ValueError(f"Task {task_num} must have a description")

        file_name = f"test_{task_num}_{function_name}.py"
        if file_name in output_paths:
            raise ValueError(f"Duplicate LeetCode task output path: {file_name}")
        output_paths.add(file_name)

        prompt_description = (
            f"{function_description}\n\n"
            f"The target method is `{function_name}` on class `Solution`; "
            "instantiate it with `Solution()` in the tests."
        )
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
                        function_code, prompt_description, function_name
                    )
                    if single_generation
                    else build_generation_prompt(
                        function_code, prompt_description, function_name
                    )
                ),
            },
        ]
        generated_tests: list[str] = []
        test_names: set[str] = set()

        generation_rounds = 1 if single_generation else num_tests
        for round_number in range(generation_rounds):
            if single_generation:
                response = query_llm(
                    function_code,
                    function_name,
                    prompt_description,
                    messages=messages,
                    max_generation_tokens=None,
                )
            else:
                response = query_llm(
                    function_code,
                    function_name,
                    prompt_description,
                    messages=messages,
                )
            test_code = extract_code_block(response)
            if not test_code:
                raise ValueError(
                    f"Ollama returned an empty test for task {task_num} "
                    f"on generation round {round_number + 1}"
                )
            try:
                parsed_test = ast.parse(test_code)
            except SyntaxError as error:
                raise ValueError(
                    f"Ollama returned invalid Python for task {task_num} "
                    f"on generation round {round_number + 1}"
                ) from error

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
                        f"for task {task_num} on generation round "
                        f"{round_number + 1}"
                    )
                if len(test_functions) != len(set(test_functions)):
                    raise ValueError(
                        f"Ollama repeated a test function name for task "
                        f"{task_num} on generation round {round_number + 1}"
                    )
            elif len(test_functions) != 1:
                raise ValueError(
                    f"Ollama must return exactly one pytest test function for "
                    f"task {task_num} on generation round {round_number + 1}"
                )
            if test_functions[0] in test_names:
                raise ValueError(
                    f"Ollama repeated test function {test_functions[0]!r} "
                    f"for task {task_num} on generation round {round_number + 1}"
                )

            test_names.add(test_functions[0])
            generated_tests.append(test_code)
            messages.append({"role": "assistant", "content": response})
            if round_number < num_tests - 1:
                messages.append(
                    {
                        "role": "user",
                        "content": build_follow_up_prompt(function_name),
                    }
                )

        task["generated_tests"] = generated_tests
        save_leetcode_test_file(
            function_code,
            "\n\n".join(generated_tests),
            file_name,
        )

    return tasks


def save_leetcode_test_file(
    function_code: str,
    test_code: str,
    file_name: str,
) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    file_path = os.path.join(OUTPUT_DIR, file_name)
    with open(file_path, "w", encoding="utf-8") as output_file:
        output_file.write(f"# Auto-generated tests for LeetCode task {file_name}\n")
        output_file.write(f"{function_code.rstrip()}\n\n{test_code}\n")
