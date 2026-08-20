from app.agent.intent import classify_intent


def test_classify_major_change_intent():
    """转专业自然表达应该识别为 major_change。"""

    result = classify_intent("我想从现在的专业转到别的专业，流程是什么")

    assert result.intent == "major_change"
    assert result.route == "procedure_qa"
    assert "转专业" in result.retrieval_query


def test_classify_grade_review_intent():
    """成绩复核问题应该识别为 grade_review。"""

    result = classify_intent("成绩怎么复核")

    assert result.intent == "grade_review"
    assert result.route == "procedure_qa"


def test_classify_transcript_intent():
    """成绩单打印问题应该识别为 transcript。"""

    result = classify_intent("成绩单怎么打印")

    assert result.intent == "transcript"
    assert result.route == "procedure_qa"


def test_classify_exam_affairs_intent():
    """缓考问题应该识别为 exam_affairs。"""

    result = classify_intent("缓考怎么申请")

    assert result.intent == "exam_affairs"
    assert result.route == "procedure_qa"


def test_classify_certificate_service_intent():
    """学生证补办问题应该识别为 certificate_service。"""

    result = classify_intent("学生证丢了怎么补办")

    assert result.intent == "certificate_service"
    assert result.route == "procedure_qa"


def test_classify_general_chat_intent():
    """非教务业务问题应该进入通用聊天分支。"""

    result = classify_intent("请解释一下 RAG 是什么")

    assert result.intent == "general_chat"
    assert result.route == "general_chat"
