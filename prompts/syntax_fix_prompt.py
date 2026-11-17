def repair_code(source: str, error: SyntaxError):
    prompt = f"""
    Fix the following Python code.

    Source:
    {source}

    Syntax error:
    {error.msg}

    Line: {error.lineno}
    Column: {error.offset}
    """