import argparse
import json
from pathlib import Path
from typing import Any

from generation_pipeline_leetcode import run_leetcode_generation_pipeline
from generation_pipeline_new import run_generation_pipeline


DATASET_FILES = {
    "ULT": "ULT.jsonl",
    "ULT_Lite": "ULT_Lite.jsonl",
    "leetcode-py-all": "leetcode-py-all.jsonl",
}


def load_dataset(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    try:
        records = json.loads(text)
    except json.JSONDecodeError:
        records = [json.loads(line) for line in text.splitlines() if line.strip()]

    if not isinstance(records, list) or not all(
        isinstance(record, dict) for record in records
    ):
        raise ValueError(f"Dataset {path} must contain a list of JSON objects")
    return records


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate tests from a dataset.")
    parser.add_argument(
        "--dataset",
        choices=DATASET_FILES,
        default="ULT",
        help="Dataset name (default: ULT)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum records to process; use 0 to process the entire dataset",
    )
    parser.add_argument(
        "--single-generation",
        action="store_true",
        help="Generate a complete test suite in one model call per record",
    )
    args = parser.parse_args(argv)
    if args.limit < 0:
        parser.error("--limit must be 0 or greater")

    dataset_path = Path(__file__).resolve().parent / "datasets" / DATASET_FILES[
        args.dataset
    ]
    records = load_dataset(dataset_path)
    selected_records = records[: args.limit] if args.limit else records

    if args.dataset == "leetcode-py-all":
        run_leetcode_generation_pipeline(
            selected_records,
            single_generation=args.single_generation,
        )
    else:
        run_generation_pipeline(
            selected_records,
            single_generation=args.single_generation,
        )


if __name__ == "__main__":
    main()
