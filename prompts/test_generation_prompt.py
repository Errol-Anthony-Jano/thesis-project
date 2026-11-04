def build_generation_prompt(focal_fn: str) -> str:
    return f"""
    You are an expert Python software testing engineer. Your task is to generate a complete, runnable pytest test suite for the provided Python function.

    ### Testing Guidelines
    
    1. Cover normal execution, boundary conditions, and relevant edge   cases.
    2. Verify both expected return values and observable side effects   where applicable.
    3. Use meaningful assertions that validate actual behavior rather   than merely checking that execution succeeds.
    4. Use `pytest.raises` to test exceptions when they are part of     the function's expected behavior.
    5. Include tests for relevant input types and invalid inputs when   justified by the implementation.
    6. Ensure each test is independent and does not rely on execution   order or shared mutable state.
    7. Do not assume behavior that cannot reasonably be inferred from   the provided implementation.
    8. Do not modify, reimplement, or reproduce the function under  test.
    9. Include all necessary imports, including `pytest`.
    10. Give each test a descriptive name.
    
    ### Output Requirements
    
    - Output ONLY valid, executable Python code.
    - Do not include Markdown code fences.
    - Do not include explanations, comments outside the test code, or   conversational text.
    - Ensure the generated code is syntactically valid and compatible   with pytest.
    
    ### Function Under Test
    
    ```python
    {focal_fn}
    ```
    """