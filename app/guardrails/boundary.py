from app.guardrails.policies import GuardrailDecision


SUPPORTED_DOMAINS = {
    "学籍",
    "选课",
    "考试",
    "成绩",
    "毕业",
    "证明",
    "奖助",
    "处分",
    "校园卡",
    "宿舍",
}


def validate_business_boundary(intent: str) -> GuardrailDecision:
    if intent in SUPPORTED_DOMAINS or intent == "通用教务咨询":
        return GuardrailDecision(True)
    return GuardrailDecision(False, "该问题超出校园教务助手的业务范围", "out_of_scope")

