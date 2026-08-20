from collections.abc import AsyncIterator

from app.agent.graph import CampusAffairsGraph
from app.core.config import get_settings
from app.core.security import RequestContext
from app.llm.client import MockLLMClient
from app.rag.retriever import InMemoryRetriever


class ChatService:
    def __init__(self, graph: CampusAffairsGraph):
        self.graph = graph

    async def stream_answer(
        self,
        question: str,
        context: RequestContext,
    ) -> AsyncIterator[dict]:
        state = await self.graph.run(question, context)
        for event in state.events:
            yield event
        yield {
            "event": "message",
            "data": {
                "answer": state.answer,
                "citations": [
                    {
                        "source": item.source,
                        "chunk_id": item.chunk_id,
                        "score": round(item.score, 4),
                    }
                    for item in state.citations
                ],
                "blocked_reason": state.blocked_reason,
            },
        }

    def available_flows(self) -> list[str]:
        return ["学生证补办", "成绩单申请", "缓考申请", "选课退课", "毕业审核"]


def build_default_chat_service() -> ChatService:
    settings = get_settings()
    graph = CampusAffairsGraph(
        retriever=InMemoryRetriever(),
        llm=MockLLMClient(),
        max_context_chunks=settings.max_context_chunks,
    )
    return ChatService(graph)

