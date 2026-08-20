from app.agent.graph import build_chat_graph
from app.rag.schemas import RetrievedChunk


class FakeRetriever:
    """测试用检索器。

    这个测试不依赖真实 Milvus 数据库，只返回一条固定的成绩单办理资料，
    用来验证当前 LangGraph 工作流是否能正常完成：
        意图识别 -> 检索 -> rerank -> 流程节点 -> 引用返回
    """

    def search(self, query: str, tenant_id: str, roles: list[str], limit: int):
        return [
            RetrievedChunk(
                chunk_id="transcript-guide:0",
                text="成绩单可通过教务系统、自助打印设备或学院指定渠道申请办理。",
                source="模拟_成绩单办理指南.md",
                tenant_id=tenant_id,
                visibility="student",
                metadata={"doc_type": "办事指南"},
                score=0.9,
            )
        ][:limit]


class FakeLLM:
    """测试用 LLM。

    成绩单办理属于当前项目里的标准流程节点，一般不会真的调用 LLM。
    这里仍然提供一个假的 generate 方法，避免其他分支需要它时报错。
    """

    def generate(self, *args, **kwargs) -> str:
        return "Fake answer"


def test_agent_routes_transcript_question_with_citation() -> None:
    """当前 Agent 工作流应能回答成绩单办理问题，并返回引用来源。"""

    graph = build_chat_graph(
        retriever=FakeRetriever(),
        llm=FakeLLM(),
    )

    state = graph.invoke(
        {
            "query": "成绩单怎么打印",
            "top_k": 3,
            "tenant_id": "generic-university",
            "role": "student",
        }
    )

    assert state["blocked"] is False
    assert state["intent"] == "transcript"
    assert state["flow_name"] == "transcript_flow"
    assert state["citations"]
    assert state["citations"][0]["source"] == "模拟_成绩单办理指南.md"
