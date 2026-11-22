import json

import generation_pipeline_new
import generation_pipeline_leetcode
import llm_client
import main
from main import load_dataset


def test_load_dataset_accepts_json_and_jsonl(tmp_path):
    json_path = tmp_path / "array.json"
    jsonl_path = tmp_path / "lines.jsonl"
    records = [{"task_num": 1}, {"task_num": 2}]
    json_path.write_text(json.dumps(records), encoding="utf-8")
    jsonl_path.write_text(
        "\n".join(json.dumps(record) for record in records),
        encoding="utf-8",
    )

    assert load_dataset(json_path) == records
    assert load_dataset(jsonl_path) == records


def test_cli_routes_leetcode_dataset_to_its_pipeline(monkeypatch):
    selected_records = []
    selected_modes = []

    def fake_pipeline(tasks, *, single_generation):
        selected_records.extend(tasks)
        selected_modes.append(single_generation)

    monkeypatch.setattr(
        main,
        "run_leetcode_generation_pipeline",
        fake_pipeline,
    )

    main.main(["--dataset", "leetcode-py-all", "--limit", "1"])

    assert len(selected_records) == 1
    assert "python_solution" in selected_records[0]
    assert "description" in selected_records[0]
    assert selected_modes == [False]


def test_cli_forwards_single_generation_flag(monkeypatch):
    selected_modes = []

    def fake_pipeline(tasks, *, single_generation):
        selected_modes.append((len(tasks), single_generation))

    monkeypatch.setattr(
        main,
        "run_leetcode_generation_pipeline",
        fake_pipeline,
    )

    main.main(
        ["--dataset", "leetcode-py-all", "--limit", "1", "--single-generation"]
    )

    assert selected_modes == [(1, True)]


def test_leetcode_pipeline_uses_task_ids_for_output_files(tmp_path, monkeypatch):
    monkeypatch.setattr(generation_pipeline_leetcode, "OUTPUT_DIR", str(tmp_path))

    def fake_query_llm(fn_code, fn_name, fn_description, *, messages):
        assert fn_code and fn_name and fn_description and messages
        return (
            "def test_is_match_case():\n"
            "    assert Solution().isMatch() is True"
        )

    monkeypatch.setattr(
        generation_pipeline_leetcode,
        "query_llm",
        fake_query_llm,
    )
    tasks = [
        {
            "task_num": task_num,
            "func_name": "isMatch",
            "description": "Check a match.",
            "python_solution": (
                "class Solution:\n"
                "    def isMatch(self):\n"
                "        return True"
            ),
        }
        for task_num in (10, 44)
    ]

    generation_pipeline_leetcode.run_leetcode_generation_pipeline(
        tasks, num_tests=1
    )

    assert (tmp_path / "test_10_isMatch.py").is_file()
    assert (tmp_path / "test_44_isMatch.py").is_file()
    assert len(tasks[0]["generated_tests"]) == 1
    assert len(tasks[1]["generated_tests"]) == 1


def test_leetcode_single_generation_accepts_a_suite_in_one_call(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(generation_pipeline_leetcode, "OUTPUT_DIR", str(tmp_path))
    calls = []

    def fake_query_llm(
        fn_code,
        fn_name,
        fn_description,
        *,
        messages,
        max_generation_tokens,
    ):
        assert fn_code and fn_name and fn_description
        assert messages[0]["content"] == (
            generation_pipeline_leetcode.SINGLE_GENERATION_SYSTEM_PROMPT
        )
        calls.append((messages, max_generation_tokens))
        return (
            "def test_is_match_empty():\n"
            "    assert Solution().isMatch() is True\n\n"
            "def test_is_match_another_case():\n"
            "    assert Solution().isMatch() is True"
        )

    monkeypatch.setattr(generation_pipeline_leetcode, "query_llm", fake_query_llm)
    tasks = [
        {
            "task_num": 10,
            "func_name": "isMatch",
            "description": "Check a match.",
            "python_solution": (
                "class Solution:\n"
                "    def isMatch(self):\n"
                "        return True"
            ),
        }
    ]

    generation_pipeline_leetcode.run_leetcode_generation_pipeline(
        tasks, num_tests=10, single_generation=True
    )

    assert len(calls) == 1
    assert calls[0][1] is None
    assert len(tasks[0]["generated_tests"]) == 1
    assert "def test_is_match_another_case" in tasks[0]["generated_tests"][0]


def test_legacy_pipeline_single_generation_uses_one_call(monkeypatch):
    calls = []

    def ignore_source(code, name):
        assert code and name

    def ignore_tests(code, name):
        assert code and name

    monkeypatch.setattr(
        generation_pipeline_new,
        "save_source_file",
        ignore_source,
    )
    monkeypatch.setattr(
        generation_pipeline_new,
        "save_test_file",
        ignore_tests,
    )

    def fake_query_llm(
        fn_code,
        fn_name,
        fn_description,
        *,
        messages,
        max_generation_tokens,
    ):
        assert fn_code and fn_name and fn_description
        assert messages[0]["content"] == (
            generation_pipeline_new.SINGLE_GENERATION_SYSTEM_PROMPT
        )
        calls.append(max_generation_tokens)
        return (
            "def test_example_one():\n"
            "    assert target() == 1\n\n"
            "def test_example_two():\n"
            "    assert target() == 1"
        )

    monkeypatch.setattr(generation_pipeline_new, "query_llm", fake_query_llm)
    functions = [
        {
            "code": "def target():\n    return 1",
            "func_name": "target",
            "prompt": "Returns one.",
        }
    ]

    generation_pipeline_new.run_generation_pipeline(
        functions, num_tests=10, single_generation=True
    )

    assert calls == [None]
    assert len(functions[0]["generated_tests"]) == 1
    assert "def test_example_two" in functions[0]["generated_tests"][0]


def test_query_llm_uses_unlimited_output_when_limit_is_none(monkeypatch):
    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"message": {"content": "response"}}

    def fake_post(url, *, json, timeout):
        captured["url"] = url
        captured["payload"] = json
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(llm_client.requests, "post", fake_post)

    llm_client.query_llm(
        "def target(): pass",
        "target",
        "Returns a value.",
        max_generation_tokens=None,
    )

    assert captured["payload"]["options"]["num_predict"] == -1
