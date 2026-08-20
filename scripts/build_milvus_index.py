import argparse
import json
import sys
from pathlib import Path


# 让脚本无论从项目根目录执行，还是用绝对路径执行，
# 都能正确 import app.xxx。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.embedding import TextEmbeddingModel


DEFAULT_CHUNKS_PATHS = [
    "data/processed/generic_university/all_docs_chunks.jsonl",
]


def load_chunks(chunks_paths: list[str]) -> list[dict]:
    """
    读取 chunks.jsonl 文件。

    chunks.jsonl 的特点：
    - 每一行是一个 JSON；
    - 每一行代表一个文档 chunk；
    - 每个 chunk 至少包含 chunk_id、text、source、tenant_id 等字段。

    这个函数会把多个 jsonl 合并成一个列表。
    """

    chunks = []

    for chunks_path in chunks_paths:
        path = Path(chunks_path)

        if not path.exists():
            print(f"[跳过] chunks 文件不存在：{path}")
            continue

        with path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue

                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"JSONL 解析失败：{path}:{line_number}"
                    ) from error

                chunks.append(row)

    return chunks


def create_collection(
    client,
    collection_name: str,
    dimension: int,
    recreate: bool,
) -> None:
    """
    创建 Milvus collection。

    collection 可以理解成 Milvus 里的“表”。
    我们为教务 chunks 创建一个 collection：

        campus_edu_chunks

    里面包含：
    - chunk_id：主键；
    - embedding：向量字段；
    - text/source/tenant_id/visibility：用于返回和过滤；
    - metadata：保存扩展信息。
    """

    from pymilvus import DataType

    if client.has_collection(collection_name):
        if recreate:
            print(f"[重建] 删除旧 collection：{collection_name}")
            client.drop_collection(collection_name)
        else:
            print(f"[复用] collection 已存在：{collection_name}")
            return

    schema = client.create_schema(
        auto_id=False,
        enable_dynamic_field=False,
    )

    # Milvus 的主键。
    # 这里直接用 chunk_id，方便后续定位原始文档片段。
    schema.add_field(
        field_name="chunk_id",
        datatype=DataType.VARCHAR,
        is_primary=True,
        max_length=512,
    )

    # 向量字段。
    # dimension 必须和 embedding 模型输出维度一致。
    schema.add_field(
        field_name="embedding",
        datatype=DataType.FLOAT_VECTOR,
        dim=dimension,
    )

    # 以下字段用于过滤和返回。
    schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
    schema.add_field(field_name="source", datatype=DataType.VARCHAR, max_length=1024)
    schema.add_field(field_name="tenant_id", datatype=DataType.VARCHAR, max_length=128)
    schema.add_field(field_name="department_id", datatype=DataType.VARCHAR, max_length=128)
    schema.add_field(field_name="visibility", datatype=DataType.VARCHAR, max_length=64)
    schema.add_field(field_name="metadata", datatype=DataType.JSON)

    index_params = client.prepare_index_params()

    # FLAT 索引适合小规模 PoC，结果精确，配置简单。
    # 数据量很大时，可以改成 HNSW / IVF_FLAT 等近似索引。
    index_params.add_index(
        field_name="embedding",
        index_type="FLAT",
        metric_type="COSINE",
    )

    print(f"[创建] collection：{collection_name}, dim={dimension}")
    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=index_params,
    )


def build_records(chunks: list[dict], embeddings: list[list[float]]) -> list[dict]:
    """
    把 chunk 数据和 embedding 合并成 Milvus 可插入记录。

    每条记录对应一个 chunk。
    """

    records = []

    for chunk, embedding in zip(chunks, embeddings, strict=True):
        records.append(
            {
                "chunk_id": chunk["chunk_id"],
                "embedding": embedding,
                "text": chunk["text"],
                "source": chunk["source"],
                "tenant_id": chunk["tenant_id"],
                # Milvus 的 VARCHAR 字段不适合写 None。
                # 没有学院 ID 时，存为空字符串。
                "department_id": chunk.get("department_id") or "",
                "visibility": chunk.get("visibility", "public"),
                "metadata": chunk.get("metadata", {}),
            }
        )

    return records


def insert_in_batches(
    client,
    collection_name: str,
    records: list[dict],
    batch_size: int,
) -> None:
    """
    分批写入 Milvus。

    为什么要分批？
    如果一次插入很多数据，内存和网络压力会比较大。
    你的当前数据量不大，但养成分批写入的习惯更接近生产项目。
    """

    total = len(records)

    for start in range(0, total, batch_size):
        batch = records[start : start + batch_size]
        client.insert(collection_name=collection_name, data=batch)
        print(f"[写入] {min(start + batch_size, total)}/{total}")


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数。
    """

    parser = argparse.ArgumentParser(
        description="把教务 chunks.jsonl 向量化并写入 Milvus。"
    )

    parser.add_argument(
        "--chunks",
        nargs="+",
        default=DEFAULT_CHUNKS_PATHS,
        help="一个或多个 chunks.jsonl 文件路径。",
    )
    parser.add_argument(
        "--uri",
        default="data/milvus/campus_edu.db",
        help=(
            "Milvus 连接地址。"
            "可以是 Milvus Lite 文件路径，也可以是 http://127.0.0.1:19530。"
        ),
    )
    parser.add_argument(
        "--collection",
        default="campus_edu_chunks",
        help="Milvus collection 名称。",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="embedding 模型名称或本地路径，默认读取 EMBEDDING_MODEL_NAME 或使用 bge-small-zh-v1.5。",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Milvus 分批写入大小。",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="如果 collection 已存在，先删除再重建。",
    )

    return parser.parse_args()


def main() -> None:
    """
    脚本入口。

    完整流程：
    1. 读取 chunks.jsonl；
    2. 加载 embedding 模型；
    3. 创建 Milvus collection；
    4. 对 chunk.text 做向量化；
    5. 把 chunk + embedding 写入 Milvus。
    """

    args = parse_args()

    try:
        from pymilvus import MilvusClient
    except ImportError as error:
        raise RuntimeError(
            "缺少 pymilvus 依赖。请先执行：pip install -r requirements.txt"
        ) from error

    # 如果使用 Milvus Lite 文件路径，需要提前创建父目录。
    # 如果 uri 是 http://127.0.0.1:19530，parent 创建不会影响服务。
    if "://" not in args.uri:
        Path(args.uri).parent.mkdir(parents=True, exist_ok=True)

    chunks = load_chunks(args.chunks)
    if not chunks:
        raise RuntimeError("没有读取到任何 chunk，请先检查 chunks.jsonl 路径。")

    print(f"[读取] chunk 数量：{len(chunks)}")

    embedding_model = TextEmbeddingModel(model_name=args.model_name)
    dimension = embedding_model.dimension

    client = MilvusClient(uri=args.uri)

    create_collection(
        client=client,
        collection_name=args.collection,
        dimension=dimension,
        recreate=args.recreate,
    )

    texts = [chunk["text"] for chunk in chunks]
    print("[向量化] 开始生成 embeddings，这一步第一次运行会下载模型，可能较慢。")
    embeddings = embedding_model.encode_texts(texts)

    records = build_records(chunks=chunks, embeddings=embeddings)
    insert_in_batches(
        client=client,
        collection_name=args.collection,
        records=records,
        batch_size=args.batch_size,
    )

    client.flush(collection_name=args.collection)
    print(f"[完成] Milvus 向量库已建立：{args.collection}")


if __name__ == "__main__":
    main()
