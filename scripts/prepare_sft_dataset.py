"""Build supervised fine-tuning samples from reviewed campus QA logs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


SYSTEM_PROMPT = (
    "你是高校教务智能助手。回答必须基于学校教务材料，无法确认时应说明限制并建议转人工。"
)


def convert_row(row: dict) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": row["question"]},
            {"role": "assistant", "content": row["approved_answer"]},
        ],
        "metadata": {
            "tenant_id": row.get("tenant_id"),
            "intent": row.get("intent"),
            "source_ids": row.get("source_ids", []),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reviewed-log", default="data/datasets/reviewed_qa.jsonl")
    parser.add_argument("--output", default="data/datasets/sft.jsonl")
    args = parser.parse_args()

    source = Path(args.reviewed_log)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with source.open("r", encoding="utf-8") as src, output.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            row = json.loads(line)
            if not row.get("approved_answer"):
                continue
            dst.write(json.dumps(convert_row(row), ensure_ascii=False) + "\n")
            count += 1

    print(f"Wrote {count} SFT samples to {output}")


if __name__ == "__main__":
    main()

