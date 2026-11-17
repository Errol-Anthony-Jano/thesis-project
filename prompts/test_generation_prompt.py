SYSTEM_PROMPT = (
    "You are a professional Python pytest test writer. Respond only with "
    "valid Python test code, without Markdown fences or natural-language "
    "descriptions. Generate exactly one test function per response, and do "
    "not rewrite or repeat the function under test."
)


def build_generation_prompt(
    focal_fn: str, fn_desc: str, fn_name: str
) -> str:
    return f"""Write one pytest test method for the function `{fn_name}`.

Function description:
----
{fn_desc}
----

Program under test:
----
{focal_fn}
----

Cover a meaningful behavior, such as a normal case, boundary, or relevant
edge case. Use assertions that verify the behavior. Include any imports
needed by this test. Name the test `test_{fn_name}_<descriptive_suffix>`.
Return exactly one complete test function and no other text."""


def build_follow_up_prompt(fn_name: str) -> str:
    return f"""Generate one additional pytest test method for `{fn_name}`.

It must test a behavior that is different from every earlier test in this
conversation, covering different statements, branches, or relevant edge
cases. Do not repeat or reproduce earlier tests. Use a unique name beginning
with `test_{fn_name}_` and include any imports needed by this test.

Return exactly one complete test function as valid Python code, with no
Markdown fences or explanation."""