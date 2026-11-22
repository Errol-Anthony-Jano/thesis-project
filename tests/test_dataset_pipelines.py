import json

import generation_pipeline_leetcode
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
    monkeypatch.setattr(
        main,
        "run_leetcode_generation_pipeline",
        lambda tasks: selected_records.extend(tasks),
    )

    main.main(["--dataset", "leetcode-py-all", "--limit", "1"])

    assert len(selected_records) == 1
    assert "python_solution" in selected_records[0]
    assert "description" in selected_records[0]


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
