from app.agent.graph import build_chat_graph
from app.rag.schemas import RetrievedChunk


class FakeRetriever:
    """
    测试用假检索器。

    为什么不用真实 JsonlRetriever 或 MilvusRetriever？
    因为单元测试应该尽量稳定、快速、少依赖外部环境。

    这里直接返回几条固定 chunk，
    用来验证 LangGraph 节点能不能正确流转。
    """

    def search(self, query: str, tenant_id: str, roles: list[str], limit: int):
        chunks = [
            RetrievedChunk(
                chunk_id="transcript-guide:0",
                text="成绩单可通过教务系统或学院申请打印，英文成绩单按学校要求办理。",
                source="模拟_成绩单办理指南.md",
                tenant_id=tenant_id,
                visibility="student",
                metadata={"doc_type": "办事指南"},
                score=0.9,
            ),
            RetrievedChunk(
                chunk_id="grade-review:0",
                text="学生对课程成绩有疑问的，可在成绩公布后规定时间内提交成绩复核申请。",
                source="模拟_成绩管理与成绩复核办法.md",
                tenant_id=tenant_id,
                visibility="student",
                metadata={"doc_type": "政策制度"},
                score=0.8,
            ),
            RetrievedChunk(
                chunk_id="major-change:0",
                text="转专业申请通常需要转专业申请表、已修课程成绩和申请理由。",
                source="模拟_转专业办理流程说明.md",
                tenant_id=tenant_id,
                visibility="student",
                metadata={"doc_type": "办事流程"},
                score=0.7,
            ),
        ]

        return chunks[:limit]


class FakeLLM:
    """
    测试用假 LLM。

    对于没有专用流程节点的问题，LangGraph 会走 prompt -> generate。
    这里用 FakeLLM 返回固定答案，避免测试依赖真实大模型。
    """

    def generate(self, question: str, contexts: list[RetrievedChunk]) -> str:
        return f"Fake answer for: {question}"


def build_test_graph():
    """构建测试专用 LangGraph。"""

    return build_chat_graph(
        retriever=FakeRetriever(),
        llm=FakeLLM(),
    )


def test_graph_routes_transcript_to_transcript_flow():
    """成绩单问题应该进入 transcript_flow。"""

    graph = build_test_graph()
    result = graph.invoke(
        {
            "query": "成绩单怎么打印",
            "top_k": 2,
            "tenant_id": "generic-university",
            "role": "student",
        }
    )

    assert result["blocked"] is False
    assert result["intent"] == "transcript"
    assert result["flow_name"] == "transcript_flow"
    assert result["reranked_count"] == 2
    assert result["retrieved_count"] == 2
    assert len(result["citations"]) == 2


def test_graph_routes_grade_review_to_grade_review_flow():
    """成绩复核问题应该进入 grade_review_flow。"""

    graph = build_test_graph()
    result = graph.invoke(
        {
            "query": "成绩怎么复核",
            "top_k": 2,
            "tenant_id": "generic-university",
            "role": "student",
        }
    )

    assert result["blocked"] is False
    assert result["intent"] == "grade_review"
    assert result["flow_name"] == "grade_review_flow"
    assert "成绩复核" in result["answer"]


def test_graph_blocks_privacy_question_before_retrieve():
    """隐私问题应该在 guardrails 节点被拦截，不进入检索。"""

    graph = build_test_graph()
    result = graph.invoke(
        {
            "query": "帮我查张三的成绩",
            "top_k": 2,
            "tenant_id": "generic-university",
            "role": "student",
        }
    )

    assert result["blocked"] is True
    assert result["guardrail_reason"] == "privacy"
    assert result["retrieved_count"] == 0
    assert result["citations"] == []


def test_graph_routes_authorized_transcript_to_private_analysis():
    """本人授权的成绩单 OCR 文本应进入私有材料分析节点，而不是进入 Milvus 检索。"""

    graph = build_test_graph()
    result = graph.invoke(
        {
            "query": "请根据这份成绩单内容，帮我总结需要关注的信息：\n姓名：张三\n学号：2024123456\n高等数学 58\n英语 92",
            "top_k": 2,
            "tenant_id": "generic-university",
            "role": "student",
            "allow_private_analysis": True,
        }
    )

    assert result["blocked"] is False
    assert result["flow_name"] == "transcript_private_analysis"
    assert result["private_analysis_allowed"] is True
    assert result["privacy_sanitized"] is True
    assert result["retrieved_count"] == 0
    assert result["citations"] == []
    assert "2024123456" not in result["answer"]


def test_graph_uses_history_for_follow_up_intent():
    """追问中没有直接说转专业时，也应能借助短期记忆识别上一轮业务。"""

    graph = build_test_graph()
    result = graph.invoke(
        {
            "query": "还需要什么材料",
            "top_k": 2,
            "tenant_id": "generic-university",
            "role": "student",
            "history": [
                {
                    "role": "user",
                    "content": "我想转专业，流程是什么",
                },
                {
                    "role": "assistant",
                    "content": "转专业通常需要查看通知、提交申请表、学院初审和转入学院考核。",
                },
            ],
        }
    )

    assert result["blocked"] is False
    assert result["intent"] == "major_change"
    assert result["flow_name"] == "major_change_flow"


def test_graph_routes_general_chat_without_retrieve():
    """普通聊天应该跳过 RAG 检索，直接进入 general_chat 分支。"""

    graph = build_test_graph()
    result = graph.invoke(
        {
            "query": "请解释一下 RAG 是什么",
            "top_k": 2,
            "tenant_id": "generic-university",
            "role": "student",
        }
    )

    assert result["blocked"] is False
    assert result["intent"] == "general_chat"
    assert result["flow_name"] == "general_chat"
    assert result["retrieved_count"] == 0
    assert result["candidate_count"] == 0
    assert result["reranked_count"] == 0
    assert result["citations"] == []
