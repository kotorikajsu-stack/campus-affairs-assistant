"""Search local chunks.jsonl files before wiring a vector database.

This is the first RAG verification step:

    user query -> retrieve relevant chunks -> inspect sources and scores

Example:

    python scripts/search_chunks.py \
      --chunks data/processed/generic_university/all_docs_chunks.jsonl \
      --query "转专业需要什么材料" \
      --tenant-id generic-university
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.security import RequestContext
from app.rag.retriever import InMemoryRetriever
from app.rag.schemas import DocumentChunk


def load_chunks(path: Path) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            chunks.append(
                DocumentChunk(
                    chunk_id=row["chunk_id"],
                    text=row["text"],
                    source=row["source"],
                    tenant_id=row["tenant_id"],
                    department_id=row.get("department_id"),
                    visibility=row.get("visibility", "public"),
                    metadata=row.get("metadata", {"line_number": line_number}),
                )
            )
    return chunks


def preview(text: str, max_chars: int) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= max_chars:
        return normalized
    return normalized[:max_chars].rstrip() + "..."


def render_markdown(
    *,
    query: str,
    chunks_count: int,
    results: list,
    preview_chars: int,
) -> str:
    lines = [
        "# 本地 RAG 检索结果",
        "",
        f"- 查询：{query}",
        f"- 加载 chunks：{chunks_count}",
        f"- 命中结果：{len(results)}",
        "",
    ]
    for index, item in enumerate(results, start=1):
        lines.extend(
            [
                f"## {index}. {item.source}",
                "",
                f"- score: `{item.score:.4f}`",
                f"- chunk_id: `{item.chunk_id}`",
                f"- visibility: `{item.visibility}`",
                f"- doc_type: `{item.metadata.get('doc_type', '')}`",
                "",
                preview(item.text, preview_chars),
                "",
            ]
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Search local RAG chunks.")
    parser.add_argument("--chunks", required=True, type=Path, help="Path to chunks jsonl.")
    parser.add_argument("--query", required=True, help="Question or keyword to search.")
    parser.add_argument("--tenant-id", default="generic-university")
    parser.add_argument("--role", action="append", default=["student"], help="Role, can repeat.")
    parser.add_argument("--department-id", default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--preview-chars", type=int, default=260)
    parser.add_argument("--save-md", type=Path, default=None, help="Save results as UTF-8 markdown.")
    args = parser.parse_args()

    chunks = load_chunks(args.chunks)
    context = RequestContext(
        user_id="local-search",
        tenant_id=args.tenant_id,
        roles=set(args.role),
        department_id=args.department_id,
    )
    retriever = InMemoryRetriever(chunks)
    results = retriever.search(args.query, context, limit=args.top_k)

    print(f"Loaded chunks: {len(chunks)}")
    print(f"Query: {args.query}")
    print(f"Results: {len(results)}")
    print()

    for index, item in enumerate(results, start=1):
        print(f"[{index}] score={item.score:.4f} source={item.source} chunk_id={item.chunk_id}")
        print(f"    visibility={item.visibility} doc_type={item.metadata.get('doc_type', '')}")
        print(f"    {preview(item.text, args.preview_chars)}")
        print()

    if args.save_md:
        args.save_md.parent.mkdir(parents=True, exist_ok=True)
        args.save_md.write_text(
            render_markdown(
                query=args.query,
                chunks_count=len(chunks),
                results=results,
                preview_chars=args.preview_chars,
            ),
            encoding="utf-8",
        )
        print(f"Saved markdown result: {args.save_md}")


if __name__ == "__main__":
    main()
