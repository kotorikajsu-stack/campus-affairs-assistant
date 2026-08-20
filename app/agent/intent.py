from dataclasses import dataclass


@dataclass(frozen=True)
class IntentResult:
    """
    IntentResult 表示一次意图识别的结果。

    用户问一句话，系统不能立刻就检索和生成。
    在真正的教务 Agent 里，通常要先判断：

    1. 用户问的是哪类教务事项；
    2. 这类事项更像政策问答，还是办事流程；
    3. 检索时应该补充哪些关键词；
    4. 后面是否需要进入某个专门流程节点。

    例如：
    用户问：“转专业需要什么材料？”

    识别结果可以是：
    intent = "major_change"
    label = "转专业"
    route = "procedure_qa"
    retrieval_query = "转专业需要什么材料 转专业 申请表 办理流程 材料 条件"
    """

    # 机器可读的意图编码。
    # 后面 LangGraph 可以根据这个字段做条件路由。
    intent: str

    # 给人看的中文意图名称。
    # 前端、日志、调试界面都可以展示这个字段。
    label: str

    # 意图大类。
    # 当前先分成：
    # - procedure_qa：办事流程类
    # - policy_qa：政策解释类
    # - form_qa：表格材料类
    # - status_guide：查询指引类
    # - general_qa：普通教务问答
    route: str

    # 识别原因。
    # 用来解释为什么识别成这个意图。
    reason: str

    # 检索增强后的 query。
    # 本地关键词检索阶段，给 query 补几个业务关键词，
    # 能让召回结果更稳定。
    retrieval_query: str


# 意图规则表。
#
# 这是当前项目的“简化版意图识别器”。
# 它不调用大模型，也不训练分类模型，只用关键词规则。
#
# 为什么先用规则？
# 1. 教务事项范围比较明确；
# 2. 规则便于解释，适合答辩和学习；
# 3. 在 PoC 阶段足够稳定；
# 4. 后面可以替换成 Qwen 分类器或微调分类模型。
#
# 每条规则包含：
# - intent：机器可读编码；
# - label：中文名称；
# - route：后续工作流大类；
# - keywords：命中哪些词就认为属于这个意图；
# - boost_terms：检索时额外拼接的关键词。
INTENT_RULES = [
    {
        "intent": "major_change",
        "label": "转专业",
        "route": "procedure_qa",
        "keywords": ["转专业", "转入专业", "换专业", "别的专业", "转到别的专业", "调整专业"],
        "boost_terms": ["转专业", "申请表", "办理流程", "申请条件", "材料", "考核"],
    },
    {
        "intent": "school_transfer",
        "label": "转学",
        "route": "procedure_qa",
        "keywords": ["转学", "转入学校", "转出学校"],
        "boost_terms": ["转学", "转学申请", "转学材料", "转学程序", "转入", "转出"],
    },
    {
        "intent": "grade_review",
        "label": "成绩复核",
        "route": "procedure_qa",
        "keywords": ["成绩复核", "复核成绩", "成绩有误", "成绩错误", "复核"],
        "boost_terms": ["成绩复核", "复核申请", "成绩管理", "申请表", "办理时间"],
    },
    {
        "intent": "transcript",
        "label": "成绩单办理",
        "route": "procedure_qa",
        "keywords": ["成绩单", "打印成绩", "英文成绩单"],
        "boost_terms": ["成绩单", "办理指南", "打印", "盖章", "证明"],
    },
    {
        "intent": "course_registration",
        "label": "选课退课重修补修",
        "route": "policy_qa",
        "keywords": ["选课", "退课", "重修", "补修", "补选", "课程替代", "学分认定"],
        "boost_terms": ["选课", "退课", "重修", "补修", "课程", "学分认定", "课程替代"],
    },
    {
        "intent": "exam_affairs",
        "label": "考试与缓补考",
        "route": "procedure_qa",
        "keywords": ["缓考", "补考", "考试违纪", "考场规则", "缺考"],
        "boost_terms": ["缓考", "补考", "考试", "申请表", "违纪", "考场规则"],
    },
    {
        "intent": "student_status_change",
        "label": "学籍异动",
        "route": "procedure_qa",
        "keywords": ["休学", "复学", "退学", "保留学籍", "学籍异动", "延长学习年限"],
        "boost_terms": ["学籍异动", "休学", "复学", "退学", "保留学籍", "申请表"],
    },
    {
        "intent": "certificate_service",
        "label": "证明和学生证办理",
        "route": "procedure_qa",
        "keywords": ["学生证", "在读证明", "学籍证明", "证明办理", "补办"],
        "boost_terms": ["学生证", "在读证明", "学籍证明", "补办", "办理指南"],
    },
    {
        "intent": "graduation_degree",
        "label": "毕业与学位",
        "route": "policy_qa",
        "keywords": ["毕业", "学位", "结业", "毕业审核", "学位授予", "结业换毕业"],
        "boost_terms": ["毕业", "学位", "毕业审核", "学位授予", "结业", "换毕业"],
    },
    {
        "intent": "calendar_schedule",
        "label": "校历和教学安排",
        "route": "status_guide",
        "keywords": ["校历", "教学周", "节假日", "调课", "放假"],
        "boost_terms": ["校历", "教学周", "节假日", "调课", "教学安排"],
    },
    {
        "intent": "practice_thesis",
        "label": "实践教学与论文",
        "route": "policy_qa",
        "keywords": ["实习", "实训", "实践教学", "毕业论文", "毕业设计"],
        "boost_terms": ["实践教学", "实习", "实训", "毕业论文", "毕业设计", "管理办法"],
    },
]


def classify_intent(query: str) -> IntentResult:
    """
    根据用户问题识别教务意图。

    当前实现是“规则优先匹配”：
    1. 从上到下遍历 INTENT_RULES；
    2. 只要用户问题包含某条规则的关键词，就返回该意图；
    3. 如果没有任何规则命中，就返回 general_academic。

    为什么规则顺序很重要？
    因为有些词会互相包含或语义接近。
    例如：
    - “成绩复核” 比 “成绩” 更具体；
    - “成绩单” 和 “成绩复核” 都包含“成绩”；
    - “转专业” 和 “转学” 都包含“转”。

    所以更具体、更容易混淆的事项要放在前面。
    """

    normalized_query = query.strip()

    for rule in INTENT_RULES:
        for keyword in rule["keywords"]:
            if keyword in normalized_query:
                boost_text = " ".join(rule["boost_terms"])
                retrieval_query = f"{normalized_query} {boost_text}"

                return IntentResult(
                    intent=rule["intent"],
                    label=rule["label"],
                    route=rule["route"],
                    reason=f"命中关键词：{keyword}",
                    retrieval_query=retrieval_query,
                )

    # 兜底意图。
    #
    # Guardrails 已经做过业务边界检查，
    # 所以能走到这里的问题通常还是教务相关，
    # 只是没有命中我们当前列出的具体事项。
    return IntentResult(
        intent="general_academic",
        label="通用教务咨询",
        route="general_qa",
        reason="未命中具体事项规则，按通用教务咨询处理",
        retrieval_query=normalized_query,
    )
from app.core.guardrails import ACADEMIC_SCOPE_KEYWORDS, contains_any


def classify_intent(query: str) -> IntentResult:
    """识别用户意图。

    新版本多了一个关键分支：general_chat。

    1. 命中具体教务业务：
       例如成绩单、转专业、缓考、学籍异动，走对应业务流程。

    2. 像教务但没有命中具体规则：
       例如“教务系统一般能办理什么”，走 general_academic，
       仍然可以使用 RAG 检索学校资料。

    3. 不像教务业务：
       例如“你好”“帮我解释一下 RAG 是什么”，走 general_chat，
       不检索知识库，直接交给通用大模型回答。
    """

    normalized_query = query.strip()

    for rule in INTENT_RULES:
        for keyword in rule["keywords"]:
            if keyword in normalized_query:
                boost_text = " ".join(rule["boost_terms"])
                retrieval_query = f"{normalized_query} {boost_text}"

                return IntentResult(
                    intent=rule["intent"],
                    label=rule["label"],
                    route=rule["route"],
                    reason=f"命中关键词：{keyword}",
                    retrieval_query=retrieval_query,
                )

    if contains_any(normalized_query, ACADEMIC_SCOPE_KEYWORDS):
        return IntentResult(
            intent="general_academic",
            label="通用教务咨询",
            route="general_qa",
            reason="未命中具体事项规则，但属于教务业务范围，按通用教务 RAG 处理",
            retrieval_query=normalized_query,
        )

    return IntentResult(
        intent="general_chat",
        label="通用聊天",
        route="general_chat",
        reason="未命中教务业务范围，按通用大模型聊天处理",
        retrieval_query=normalized_query,
    )
