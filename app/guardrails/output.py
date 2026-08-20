from app.guardrails.policies import GuardrailDecision
from app.rag.schemas import RetrievedChunk


def validate_output(answer: str, citations: list[RetrievedChunk]) -> GuardrailDecision:
    if "我保证" in answer or "一定可以" in answer:
        return GuardrailDecision(False, "输出含有绝对化承诺", "overpromise")
    if citations and "依据" not in answer and "参考" not in answer:
        return GuardrailDecision(False, "输出缺少引用提示", "missing_citation_notice")
    return GuardrailDecision(True)

