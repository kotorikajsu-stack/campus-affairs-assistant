from app.core.security import RequestContext, build_row_level_filter
from app.rag.schemas import RetrievedChunk


class MilvusRetriever:
    """Production Milvus adapter placeholder.

    Keep this class thin: embedding, vector search, metadata filtering, and reranking
    should be independently replaceable for different universities.
    """

    def __init__(self, uri: str, collection: str, embedding_model: str):
        self.uri = uri
        self.collection = collection
        self.embedding_model = embedding_model

    def search(self, query: str, context: RequestContext, limit: int = 5) -> list[RetrievedChunk]:
        filters = build_row_level_filter(context)
        raise NotImplementedError(
            "Connect pymilvus and an embedding service here. "
            f"Query={query!r}, filters={filters}, limit={limit}"
        )

