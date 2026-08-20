"""Parse, clean, chunk, and index campus affairs documents.

This script is intentionally adapter-oriented:
- MinerU parsing should produce structured markdown/text under data/processed.
- Embeddings and Milvus writes can be enabled in the production adapter.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.chunker import chunk_text


def parse_document_metadata(text: str) -> dict:
    metadata: dict[str, str] = {}
    mapping = {
        "文档类型": "doc_type",
        "适用对象": "applicable_to",
        "可见范围": "visibility",
        "状态": "status",
    }
    for line in text.splitlines()[:12]:
        if "：" not in line:
            continue
        key, value = line.split("：", 1)
        normalized_key = key.strip("# ").strip()
        if normalized_key in mapping:
            metadata[mapping[normalized_key]] = value.strip()
    return metadata


def ingest_text_file(path: Path, tenant_id: str) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    metadata = parse_document_metadata(text)
    chunks = chunk_text(text, source=path.name, tenant_id=tenant_id)
    chunks = [
        replace(
            chunk,
            visibility=metadata.get("visibility", chunk.visibility),
            metadata={**chunk.metadata, **metadata, "relative_path": str(path)},
        )
        for chunk in chunks
    ]
    return [
        {
            "chunk_id": chunk.chunk_id,
            "text": chunk.text,
            "source": chunk.source,
            "tenant_id": chunk.tenant_id,
            "department_id": chunk.department_id,
            "visibility": chunk.visibility,
            "metadata": chunk.metadata,
        }
        for chunk in chunks
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="data/processed")
    parser.add_argument("--output", default="data/processed/chunks.jsonl")
    parser.add_argument("--tenant-id", default="demo-university")
    parser.add_argument("--recursive", action="store_true", help="Recursively scan markdown and txt files.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    pattern = "**/*" if args.recursive else "*"
    paths = sorted(input_dir.glob(f"{pattern}.md")) + sorted(input_dir.glob(f"{pattern}.txt"))
    for path in paths:
        rows.extend(ingest_text_file(path, args.tenant_id))

    with output.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows)} chunks to {output}")


if __name__ == "__main__":
    main()
