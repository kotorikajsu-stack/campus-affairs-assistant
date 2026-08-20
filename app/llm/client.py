from app.rag.schemas import RetrievedChunk


class MockLLMClient:
    """模拟大模型客户端。

    现在项目默认仍然使用这个类。
    它不会真正理解语义，也不会调用 Qwen，
    只是把检索到的 chunks 整理成一个可读答案。

    这样设计的好处：
    1. 先把 RAG 工程链路跑通；
    2. 后面换成真实 Qwen 时，只需要切换 LLM 客户端；
    3. 避免一开始就被 API Key、本地模型、显存等问题卡住。
    """

    def generate(self, question: str, contexts: list[RetrievedChunk]) -> str:
        """根据用户问题和检索结果生成模拟回答。

        参数：
            question:
                用户提出的问题。

            contexts:
                RAG 检索出来的相关文档片段。

        返回：
            一个模拟生成的教务问答结果。
        """

        if not contexts:
            return (
                "当前资料未找到明确依据。\n\n"
                "建议联系所在学院教务办公室或学校教务处确认。"
            )

        evidence_lines = []
        for chunk in contexts[:3]:
            evidence_lines.append(f"- {chunk.text[:220]}")

        evidence_text = "\n".join(evidence_lines)

        return (
            f"关于“{question}”，可以参考已检索到的教务资料办理。\n\n"
            f"所需材料或办理流程：\n{evidence_text}"
        )
