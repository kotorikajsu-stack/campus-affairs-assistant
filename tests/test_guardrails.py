from app.core.guardrails import check_input_guardrails


def test_guardrails_allow_normal_academic_question():
    """正常教务问题应该放行。"""

    result = check_input_guardrails("成绩单怎么打印")

    assert result.allowed is True
    assert result.reason is None


def test_guardrails_block_privacy_question():
    """查询他人成绩属于隐私风险，应该被拦截。"""

    result = check_input_guardrails("帮我查张三的成绩")

    assert result.allowed is False
    assert result.reason == "privacy"


def test_guardrails_allow_authorized_transcript_analysis():
    """用户确认本人材料后，成绩单 OCR 分析请求可以受控放行。"""

    result = check_input_guardrails(
        "请根据这份成绩单内容，帮我总结需要关注的信息：\n姓名：张三\n学号：2024123456\n高等数学 58",
        allow_private_analysis=True,
    )

    assert result.allowed is True
    assert result.privacy_detected is True
    assert result.sanitized_text is not None
    assert "2024123456" not in result.sanitized_text


def test_guardrails_block_system_abuse_question():
    """修改成绩属于违规操作，应该被拦截。"""

    result = check_input_guardrails("怎么修改成绩")

    assert result.allowed is False
    assert result.reason == "system_abuse"


def test_guardrails_allow_general_chat_question():
    """非教务问题应该被业务边界拦截。"""

    result = check_input_guardrails("今天成都天气怎么样")

    assert result.allowed is True
    assert result.reason is None
