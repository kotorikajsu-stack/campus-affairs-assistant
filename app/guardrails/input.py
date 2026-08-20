from app.guardrails.policies import GuardrailDecision


BLOCKED_KEYWORDS = {
    "泄露密码",
    "伪造成绩",
    "篡改成绩",
    "绕过审核",
    "删除处分记录",
}


def validate_input(question: str) -> GuardrailDecision:
    normalized = question.strip()
    if not normalized:
        return GuardrailDecision(False, "问题不能为空", "empty_input")
    if len(normalized) > 2000:
        return GuardrailDecision(False, "问题过长，请精简后再试", "too_long")
    for keyword in BLOCKED_KEYWORDS:
        if keyword in normalized:
            return GuardrailDecision(False, "请求涉及违规操作，无法协助", "unsafe_intent")
    return GuardrailDecision(True)

