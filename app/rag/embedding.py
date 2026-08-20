import os


class TextEmbeddingModel:
    """
    文本向量化模型封装。

    RAG 的向量检索需要先把文本变成向量：

    文本：
        “转专业需要什么材料”

    向量：
        [0.012, -0.034, 0.088, ...]

    Milvus 存储和检索的就是这种向量。

    为什么单独封装一个类？
    1. 后面换 embedding 模型时，只改这里；
    2. 建库脚本和在线检索可以复用同一套向量化逻辑；
    3. 可以统一处理 query 和 document 的编码方式。
    """

    def __init__(self, model_name: str | None = None) -> None:
        """
        初始化 embedding 模型。

        model_name:
            模型名称或本地模型路径。

        默认使用 BAAI/bge-small-zh-v1.5。
        这个模型比较轻量，适合先在本机把流程跑通。

        你也可以通过环境变量修改：
            $env:EMBEDDING_MODEL_NAME="BAAI/bge-base-zh-v1.5"

        或者使用本地模型路径：
            $env:EMBEDDING_MODEL_NAME="F:/models/bge-small-zh-v1.5"
        """

        self.model_name = model_name or os.getenv(
            "EMBEDDING_MODEL_NAME",
            "BAAI/bge-small-zh-v1.5",
        )

        self._model = None

    def _load_model(self):
        """
        懒加载 sentence-transformers 模型。

        为什么懒加载？
        因为 FastAPI 启动、脚本 import 时不一定马上需要 embedding。
        等真正调用 encode_texts() 或 encode_query() 时再加载，
        可以减少启动等待时间。
        """

        if self._model is not None:
            return self._model

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise RuntimeError(
                "缺少 sentence-transformers 依赖。请先执行：pip install -r requirements.txt"
            ) from error

        self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def dimension(self) -> int:
        """
        返回 embedding 向量维度。

        Milvus 创建 collection 时必须提前知道向量维度。
        例如 bge-small-zh-v1.5 的维度通常是 512。
        """

        model = self._load_model()
        return int(model.get_sentence_embedding_dimension())

    def encode_texts(self, texts: list[str]) -> list[list[float]]:
        """
        把一批文档文本转成向量。

        这个方法用于“建库阶段”：
        chunks.jsonl 中每个 chunk 的 text 都会通过这里转成 embedding，
        然后写入 Milvus。

        normalize_embeddings=True 表示把向量归一化。
        使用 COSINE 相似度检索时，归一化后的向量更稳定。
        """

        if not texts:
            return []

        model = self._load_model()
        embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        return embeddings.tolist()

    def encode_query(self, query: str) -> list[float]:
        """
        把用户问题转成向量。

        这个方法用于“在线检索阶段”：
        用户问一句话，我们先把问题转成 query embedding，
        再拿这个向量去 Milvus 中搜索相似 chunks。
        """

        vectors = self.encode_texts([query])
        return vectors[0]
