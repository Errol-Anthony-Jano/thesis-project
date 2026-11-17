def build_syntax_fix_prompt(error_function):
    # Borrowed from ChatUniTest (2024)
    return f"""
        I cannot compile the function that you provided.

        Function
        ----
        ${error_function}
        ----

        Fix this function. Do not modify the logic of the function.
    """