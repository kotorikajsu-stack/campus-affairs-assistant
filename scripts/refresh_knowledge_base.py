"""One-command knowledge base refresh.

This script is the operational wrapper for the local RAG knowledge base.

Typical workflow:
1. Put new parsed Markdown/TXT files into the configured source folders.
2. Run this script once.
3. Restart the FastAPI service if it was already running.

The script does two things:
- Convert source Markdown/TXT files into one JSONL chunks file.
- Rebuild the Milvus collection from that chunks file.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import sys


# Make imports work whether this file is called from project root or by absolute path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.chunker import chunk_text
from app.rag.embedding import TextEmbeddingModel
from scripts.build_milvus_index import (
    build_records,
    create_collection,
    insert_in_batches,
)
from scripts.ingest_documents import parse_document_metadata


DEFAULT_TENANT_ID = "generic-university"
DEFAULT_OUTPUT = "data/processed/generic_university/all_docs_chunks.jsonl"
DEFAULT_MILVUS_URI = "data/milvus/campus_edu.db"
DEFAULT_COLLECTION = "campus_edu_chunks"
DEFAULT_SOURCE_DIRS = [
    "data/raw/generic_university/synthetic_docs",
    "data/processed/generic_university",
]
DEFAULT_EXCLUDE_NAMES = {
    "README.md",
    "readme.md",
    "search_result_demo.md",
}


def resolve_project_path(path_text: str) -> Path:
    """Resolve a command-line path relative to the project root.

    Users usually run commands from the project root, but scripts may also be
    called with absolute paths. This helper keeps both styles working.
    """

    path = Path(path_text)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def source_label(path: Path) -> str:
    """Create a stable source label for citations and chunk IDs.

    We do not use only path.name here because different folders may contain
    files with the same name. A relative path keeps chunk_id unique enough for
    Milvus primary keys while still being readable in citations.
    """

    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.name


def find_source_files(source_dirs: list[str], exclude_names: set[str]) -> list[Path]:
    """Collect Markdown/TXT files from the configured source directories."""

    files: list[Path] = []

    for source_dir_text in source_dirs:
        source_dir = resolve_project_path(source_dir_text)

        if not source_dir.exists():
            print(f"[跳过] 来源目录不存在：{source_dir}")
            continue

        for path in sorted(source_dir.rglob("*")):
            if not path.is_file():
                continue

            if path.suffix.lower() not in {".md", ".txt"}:
                continue

            if path.name in exclude_names:
                print(f"[排除] {path}")
                continue

            files.append(path)

    # A file may appear twice if source directories overlap. Keep order, remove duplicates.
    unique_files = list(dict.fromkeys(files))
    return unique_files


def build_chunks(
    files: list[Path],
    tenant_id: str,
    chunk_size: int,
    overlap: int,
) -> list[dict]:
    """Parse metadata, split documents into chunks, and return JSONL rows."""

    rows: list[dict] = []

    for path in files:
        text = path.read_text(encoding="utf-8")
        metadata = parse_document_metadata(text)
        label = source_label(path)

        chunks = chunk_text(
            text,
            source=label,
            tenant_id=tenant_id,
            chunk_size=chunk_size,
            overlap=overlap,
        )

        for chunk in chunks:
            chunk = replace(
                chunk,
                visibility=metadata.get("visibility", chunk.visibility),
                metadata={
                    **chunk.metadata,
                    **metadata,
                    "relative_path": label,
                    "file_name": path.name,
                },
            )

            rows.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text,
                    "source": chunk.source,
                    "tenant_id": chunk.tenant_id,
                    "department_id": chunk.department_id,
                    "visibility": chunk.visibility,
                    "metadata": chunk.metadata,
                }
            )

    return rows


def write_chunks(rows: list[dict], output_path: Path) -> None:
    """Write chunks to JSONL."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def rebuild_milvus(
    rows: list[dict],
    uri: str,
    collection_name: str,
    embedding_model_name: str | None,
    batch_size: int,
) -> None:
    """Vectorize chunks and rebuild the Milvus collection."""

    try:
        from pymilvus import MilvusClient
    except ImportError as error:
        active_env = os.getenv("CONDA_DEFAULT_ENV") or "未检测到 Conda 环境"
        raise RuntimeError(
            "缺少 pymilvus 依赖，无法写入 Milvus。\n"
            f"当前环境：{active_env}\n"
            "请先确认已经激活项目环境：conda activate ICSA\n"
            "然后再执行：python scripts\\refresh_knowledge_base.py\n"
            "如果 ICSA 环境里仍然缺依赖，再执行：python -m pip install -r requirements.txt"
        ) from error

    if "://" not in uri:
        resolve_project_path(uri).parent.mkdir(parents=True, exist_ok=True)
        resolved_uri = str(resolve_project_path(uri))
    else:
        resolved_uri = uri

    embedding_model = TextEmbeddingModel(model_name=embedding_model_name)
    client = MilvusClient(uri=resolved_uri)

    create_collection(
        client=client,
        collection_name=collection_name,
        dimension=embedding_model.dimension,
        recreate=True,
    )

    texts = [row["text"] for row in rows]
    print("[向量化] 开始生成 embeddings。首次运行可能需要下载 embedding 模型。")
    embeddings = embedding_model.encode_texts(texts)

    records = build_records(chunks=rows, embeddings=embeddings)
    insert_in_batches(
        client=client,
        collection_name=collection_name,
        records=records,
        batch_size=batch_size,
    )
    client.flush(collection_name=collection_name)
    client.load_collection(collection_name=collection_name)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(
        description="一键重新切分文档并重建 Milvus 知识库。"
    )

    parser.add_argument(
        "--source-dir",
        action="append",
        dest="source_dirs",
        help=(
            "文档来源目录，可重复传入。"
            "默认读取 synthetic_docs 和 processed/generic_university。"
        ),
    )
    parser.add_argument(
        "--tenant-id",
        default=DEFAULT_TENANT_ID,
        help="租户/学校 ID，用于 Milvus 权限过滤。",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="生成的 chunks.jsonl 输出路径。",
    )
    parser.add_argument(
        "--uri",
        default=DEFAULT_MILVUS_URI,
        help="Milvus Lite 文件路径或 Milvus 服务地址。",
    )
    parser.add_argument(
        "--collection",
        default=DEFAULT_COLLECTION,
        help="Milvus collection 名称。",
    )
    parser.add_argument(
        "--embedding-model",
        default=None,
        help="embedding 模型名称或本地路径，默认使用项目配置。",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Milvus 分批写入大小。",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=420,
        help="单个 chunk 的最大字符数。",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=60,
        help="相邻 chunk 的重叠字符数。",
    )
    parser.add_argument(
        "--include-readme",
        action="store_true",
        help="默认会排除 README.md；打开后允许 README.md 入库。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只扫描和切分，不写 Milvus。",
    )

    return parser.parse_args()


def main() -> None:
    """Script entry point."""

    args = parse_args()

    source_dirs = args.source_dirs or DEFAULT_SOURCE_DIRS
    exclude_names = set(DEFAULT_EXCLUDE_NAMES)
    if args.include_readme:
        exclude_names.discard("README.md")
        exclude_names.discard("readme.md")

    print("[开始] 刷新校园教务知识库")
    print(f"[租户] {args.tenant_id}")
    print("[来源目录]")
    for source_dir in source_dirs:
        print(f"  - {resolve_project_path(source_dir)}")

    files = find_source_files(source_dirs=source_dirs, exclude_names=exclude_names)
    if not files:
        raise RuntimeError("没有找到可入库的 Markdown/TXT 文件。")

    print(f"[扫描] 文档数量：{len(files)}")
    for path in files:
        print(f"  - {source_label(path)}")

    rows = build_chunks(
        files=files,
        tenant_id=args.tenant_id,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )
    if not rows:
        raise RuntimeError("文档已扫描到，但切分结果为空，请检查文档内容。")

    output_path = resolve_project_path(args.output)
    write_chunks(rows=rows, output_path=output_path)
    print(f"[切分] chunk 数量：{len(rows)}")
    print(f"[输出] {output_path}")

    if args.dry_run:
        print("[完成] dry-run 模式未写入 Milvus。")
        return

    print("[提醒] 如果 FastAPI 服务正在使用同一个 Milvus Lite 文件，建议先停止服务再重建。")
    rebuild_milvus(
        rows=rows,
        uri=args.uri,
        collection_name=args.collection,
        embedding_model_name=args.embedding_model,
        batch_size=args.batch_size,
    )
    print(f"[完成] Milvus 知识库已重建：{args.collection}")


if __name__ == "__main__":
    main()
