import os
from typing import Any

from app.rag.embedding import TextEmbeddingModel
from app.rag.schemas import RetrievedChunk


def build_visibility_filter(tenant_id: str, roles: list[str]) -> str:
    """
    构建 Milvus 标量过滤表达式。

    向量检索不能只看语义相似度，还必须做权限控制。
    例如：
    - A 学校的学生不能搜到 B 学校资料；
    - 学生不能搜到 internal 内部资料；
    - 教师可以看到 teacher 可见资料；
    - 管理员可以看到 internal 资料。

    Milvus 支持在向量搜索时附加 filter。
    这样可以做到：

        先按 tenant_id / visibility 过滤
        再做向量相似度搜索

    这比搜索后再过滤更安全，因为越权数据不会进入召回结果。
    """

    allowed_visibility = ["public"]

    if "student" in roles:
        allowed_visibility.append("student")

    if "teacher" in roles:
        allowed_visibility.append("teacher")

    if "admin" in roles:
        allowed_visibility.extend(["teacher", "internal"])

    # 去重，避免表达式里重复出现同一个 visibility。
    allowed_visibility = list(dict.fromkeys(allowed_visibility))

    visibility_expr = ", ".join(f'"{item}"' for item in allowed_visibility)

    return f'tenant_id == "{tenant_id}" and visibility in [{visibility_expr}]'


class MilvusRetriever:
    """
    基于 Milvus 的向量检索器。

    它和 JsonlRetriever 暴露同样的 search() 方法：

        search(query, tenant_id, roles, limit) -> list[RetrievedChunk]

    这样 LangGraph 不需要知道底层检索方式。
    你可以用环境变量切换：

        RETRIEVER_BACKEND=jsonl
        RETRIEVER_BACKEND=milvus

    当前 MilvusRetriever 做的事情：
    1. 把用户问题转成 query embedding；
    2. 在 Milvus collection 中做向量相似度搜索；
    3. 用 tenant_id 和 visibility 做权限过滤；
    4. 把搜索命中结果转成 RetrievedChunk。
    """

    def __init__(
        self,
        collection_name: str | None = None,
        uri: str | None = None,
        embedding_model: TextEmbeddingModel | None = None,
    ) -> None:
        """
        初始化 Milvus 检索器。

        collection_name:
            Milvus 集合名称。

        uri:
            Milvus 连接地址。

            可以是本地 Milvus Lite 文件：
                data/milvus/campus_edu.db

            也可以是独立 Milvus 服务：
                http://127.0.0.1:19530

        embedding_model:
            文本向量化模型。
            默认使用 TextEmbeddingModel。
        """

        self.collection_name = collection_name or os.getenv(
            "CAMPUS_MILVUS_COLLECTION",
            "campus_edu_chunks",
        )
        self.uri = uri or os.getenv("CAMPUS_MILVUS_URI", "data/milvus/campus_edu.db")
        self.embedding_model = embedding_model or TextEmbeddingModel()

        # pymilvus 自己也会读取 MILVUS_URI 环境变量。
        # 但它的全局 MILVUS_URI 只接受 http://... 这种服务地址，
        # 不接受 Milvus Lite 的本地 .db 文件路径。
        #
        # 我们项目自己的变量改用 CAMPUS_MILVUS_URI。
        # 如果你终端里残留了旧的 MILVUS_URI=data/milvus/campus_edu.db，
        # pymilvus 在 import 阶段就会抛 Illegal uri。
        #
        # 所以这里在导入 pymilvus 前，移除非法的 MILVUS_URI。
        old_milvus_uri = os.environ.get("MILVUS_URI")
        if old_milvus_uri and "://" not in old_milvus_uri:
            os.environ.pop("MILVUS_URI", None)

        try:
            from pymilvus import MilvusClient
        except ImportError as error:
            raise RuntimeError(
                "缺少 pymilvus 依赖。请先执行：pip install -r requirements.txt"
            ) from error

        self.client = MilvusClient(uri=self.uri)
        self._ensure_collection_loaded()

    def _ensure_collection_loaded(self) -> None:
        """
        确保 Milvus collection 已经加载到内存。

        Milvus 的 collection 有两种常见状态：

        1. released：
           只是在磁盘/数据库里存在，但还没有加载到内存。
           这种状态下不能 search。

        2. loaded：
           已经加载到内存，可以执行 search / query。

        你刚才遇到的错误就是：

            Collection 'campus_edu_chunks' is in state 'released';
            call load() before search/get/query

        所以检索器初始化时主动调用 load_collection()。
        搜索前也会再兜底调用一次，避免 collection 被释放后搜索失败。
        """

        if not self.client.has_collection(self.collection_name):
            return

        self.client.load_collection(collection_name=self.collection_name)

    def search(
        self,
        query: str,
        tenant_id: str,
        roles: list[str] | None = None,
        limit: int = 5,
    ) -> list[RetrievedChunk]:
        """
        使用 Milvus 做向量检索。

        参数含义和 JsonlRetriever.search() 保持一致。
        """

        roles = roles or ["student"]

        if limit <= 0:
            return []

        # 如果 collection 不存在，说明还没有执行 build_milvus_index.py。
        # 这里直接抛出清晰错误，方便你定位问题。
        if not self.client.has_collection(self.collection_name):
            raise RuntimeError(
                f"Milvus collection 不存在：{self.collection_name}。"
                "请先运行 scripts/build_milvus_index.py 建立向量库。"
            )

        # Milvus 搜索前必须 load collection。
        # 如果已经 load 过，重复调用通常不会有副作用。
        self._ensure_collection_loaded()

        query_vector = self.embedding_model.encode_query(query)
        filter_expr = build_visibility_filter(tenant_id=tenant_id, roles=roles)

        search_results = self.client.search(
            collection_name=self.collection_name,
            data=[query_vector],
            anns_field="embedding",
            filter=filter_expr,
            limit=limit,
            output_fields=[
                "chunk_id",
                "text",
                "source",
                "tenant_id",
                "department_id",
                "visibility",
                "metadata",
            ],
        )

        # MilvusClient.search 返回的是二维列表：
        # [
        #   [hit1, hit2, hit3]
        # ]
        #
        # 因为我们只传了一个 query_vector，
        # 所以这里只取 search_results[0]。
        hits = search_results[0] if search_results else []

        return [self._hit_to_chunk(hit) for hit in hits]

    def _hit_to_chunk(self, hit: dict[str, Any]) -> RetrievedChunk:
        """
        把 Milvus 的 hit 转成项目内部使用的 RetrievedChunk。

        不同 pymilvus 版本返回的 hit 结构可能略有差异。
        常见结构是：

        {
            "id": "...",
            "distance": 0.83,
            "entity": {
                "chunk_id": "...",
                "text": "...",
                ...
            }
        }

        这里写得稍微兼容一点，避免版本差异导致小问题。
        """

        # pymilvus 版本不同，hit 可能是 dict，也可能是对象。
        # 为了让代码兼容两种情况，这里统一转成字典结构处理。
        if isinstance(hit, dict):
            entity = hit.get("entity", hit)
            score = hit.get("distance", hit.get("score", 0.0))
        else:
            entity = getattr(hit, "entity", None) or {}
            score = getattr(hit, "distance", getattr(hit, "score", 0.0))

        # entity 也可能是 dict 或对象。
        # MilvusClient 新版本通常返回 dict，旧接口可能返回可属性访问对象。
        def read_entity(field_name: str, default: Any = None) -> Any:
            if isinstance(entity, dict):
                return entity.get(field_name, default)

            if hasattr(entity, "get"):
                return entity.get(field_name, default)

            return getattr(entity, field_name, default)

        return RetrievedChunk(
            chunk_id=read_entity("chunk_id"),
            text=read_entity("text"),
            source=read_entity("source"),
            tenant_id=read_entity("tenant_id"),
            department_id=read_entity("department_id") or None,
            visibility=read_entity("visibility", "public"),
            metadata=read_entity("metadata") or {},
            score=float(score),
        )
