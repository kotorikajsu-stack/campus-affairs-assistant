from dataclasses import dataclass
import re

from app.core.privacy import (
    is_authorized_private_analysis_request,
    sanitize_private_text,
)


@dataclass(frozen=True)
class GuardrailResult:
    """
    GuardrailResult 表示一次安全检查的结果。

    为什么要单独定义这个类？
    因为 Guardrails 不应该只返回 True / False。
    如果拦截了用户问题，后端还需要知道：
    1. 为什么拦截；
    2. 属于哪一类风险；
    3. 应该给用户返回什么提示。
    """

    # allowed=True 表示允许进入后续 RAG 流程。
    # allowed=False 表示拦截，不再检索知识库，也不调用大模型。
    allowed: bool

    # reason 是机器可读的原因编码，方便后面做日志统计。
    # 例如：privacy、system_abuse、out_of_scope。
    reason: str | None = None

    # message 是返回给用户看的中文提示。
    message: str | None = None

    # privacy_detected=True 表示输入里包含成绩、学号等个人敏感信息。
    privacy_detected: bool = False

    # 如果用户明确授权本人材料分析，这里保存脱敏后的文本。
    sanitized_text: str | None = None


# 隐私类关键词。
#
# 这类问题通常涉及学生个人信息、成绩、学籍状态等。
# 教务助手可以解释“如何查询”，但不能替用户查询某个人的隐私数据。
PRIVACY_KEYWORDS = [
    "别人",
    "某某",
    "他人",
    "同学的成绩",
    "查成绩",
    "身份证号",
    "手机号",
    "家庭住址",
    "个人信息",
    "学号密码",
]


# 系统攻击或违规操作关键词。
#
# 这类问题不属于教务咨询，且存在明显安全风险。
SYSTEM_ABUSE_KEYWORDS = [
    "破解",
    "绕过",
    "盗号",
    "改成绩",
    "修改成绩",
    "刷分",
    "爬取全部学生",
    "导出所有学生",
    "教务系统漏洞",
    "sql注入",
    "xss",
]


# 校园教务助手允许回答的业务范围。
#
# 如果用户问题完全不包含这些教务相关词，
# 我们就认为它可能偏离业务边界。
ACADEMIC_SCOPE_KEYWORDS = [
    "转专业",
    "换专业",
    "专业",
    "别的专业",
    "转到别的专业",
    "调整专业",
    "转学",
    "学籍",
    "选课",
    "退课",
    "重修",
    "补修",
    "缓考",
    "补考",
    "考试",
    "违纪",
    "成绩",
    "成绩单",
    "成绩复核",
    "学生证",
    "在读证明",
    "学籍证明",
    "毕业",
    "学位",
    "结业",
    "校历",
    "教学周",
    "调课",
    "休学",
    "复学",
    "退学",
    "保留学籍",
    "学分",
    "课程",
    "培养方案",
    "实习",
    "毕业论文",
    "教务",
]


def contains_any(text: str, keywords: list[str]) -> bool:
    """
    判断文本是否包含任意一个关键词。

    这里先用最简单的字符串包含判断。
    对当前项目阶段来说，它足够直观，也方便你调试。
    后面如果规则变复杂，可以换成：
    1. 正则表达式；
    2. 意图分类模型；
    3. LLM 安全分类器。
    """

    return any(keyword in text for keyword in keywords)


def is_privacy_query(text: str) -> bool:
    """
    判断用户问题是否像是在查询他人隐私。

    为什么不只用关键词？
    因为真实用户不会总是说“别人”或“他人”，
    他可能会说：
    - 帮我查张三的成绩
    - 能不能查某个同学的学号
    - 查一下 20240001 的成绩

    所以这里额外加一些简单规则：
    1. 出现明确隐私关键词，直接拦截；
    2. 出现“查 + 某人/某学号 + 成绩”这类句式，拦截；
    3. 出现疑似学号/身份证号，也拦截。
    """

    if contains_any(text, PRIVACY_KEYWORDS):
        return True

    # 常见人名占位词或关系词。
    # 这些词和“成绩/学籍/个人信息”同时出现时，通常是在查他人信息。
    other_person_words = [
        "张三",
        "李四",
        "王五",
        "某个同学",
        "同学",
        "室友",
        "朋友",
        "别人",
        "他人",
    ]

    sensitive_info_words = [
        "成绩",
        "学籍状态",
        "学号",
        "身份证",
        "手机号",
        "个人信息",
    ]

    if contains_any(text, other_person_words) and contains_any(text, sensitive_info_words):
        return True

    # 匹配“查xxx的成绩”这类句式。
    # 例如：“帮我查张三的成绩”。
    if re.search(r"(查|查询|看|获取).{1,12}的成绩", text):
        return True

    # 匹配疑似学号或身份证号。
    # 这里不做精确身份识别，只要用户把长数字和敏感词放在一起，就先拦截。
    has_long_number = re.search(r"\d{8,18}", text) is not None
    if has_long_number and contains_any(text, sensitive_info_words):
        return True

    return False


def check_input_guardrails(query: str, allow_private_analysis: bool = False) -> GuardrailResult:
    """
    输入层 Guardrails：检查用户问题能不能进入 RAG。

    它位于整个链路最前面：

    用户问题
      -> check_input_guardrails()
      -> 通过后才进入 retriever.search()
      -> 再组装 prompt
      -> 再调用 LLM

    这样做的好处：
    1. 风险问题不会进入知识库检索；
    2. 风险问题不会发送给大模型；
    3. 可以减少越权查询、隐私泄露和跑题回答。
    """

    # 去掉首尾空格，避免用户只输入空白字符。
    normalized_query = query.strip()

    if not normalized_query:
        return GuardrailResult(
            allowed=False,
            reason="empty_query",
            message="请输入需要咨询的教务问题。",
        )

    # 第一层：拦截系统攻击、违规操作。
    if contains_any(normalized_query, SYSTEM_ABUSE_KEYWORDS):
        return GuardrailResult(
            allowed=False,
            reason="system_abuse",
            message="该问题涉及系统安全或违规操作，教务助手不能提供相关帮助。",
        )

    # 第二层：拦截明显的隐私查询。
    if is_privacy_query(normalized_query):
        if allow_private_analysis and is_authorized_private_analysis_request(normalized_query):
            return GuardrailResult(
                allowed=True,
                privacy_detected=True,
                sanitized_text=sanitize_private_text(normalized_query),
            )

        return GuardrailResult(
            allowed=False,
            reason="privacy",
            message=(
                "该问题可能涉及学生个人隐私。"
                "如需查询本人信息，请通过学校官方教务系统或联系所在学院办理。"
            ),
        )

    # 第三层：业务边界检查。
    # 如果用户问题完全不像教务问题，就不进入 RAG。
    if not contains_any(normalized_query, ACADEMIC_SCOPE_KEYWORDS):
        return GuardrailResult(
            allowed=False,
            reason="out_of_scope",
            message="我目前主要回答校园教务相关问题，请咨询学籍、选课、考试、成绩、毕业等事项。",
        )

    # 所有检查都通过，允许进入 RAG。
    return GuardrailResult(allowed=True)
def is_academic_scope_query(query: str) -> bool:
    """判断问题是否属于校园教务业务范围。

    这个函数只做“业务分流”判断，不负责拦截。

    项目现在的目标是：
    1. 普通聊天可以正常进入通用大模型；
    2. 教务业务问题才进入 RAG / 权限过滤 / 流程节点；
    3. 隐私查询和违规操作仍然必须拦截。
    """

    return contains_any(query.strip(), ACADEMIC_SCOPE_KEYWORDS)


def check_input_guardrails(query: str, allow_private_analysis: bool = False) -> GuardrailResult:
    """输入安全护栏。

    新版本策略：
        1. 空问题：拦截；
        2. 系统攻击、违规操作：拦截；
        3. 查询他人隐私：拦截；
        4. 非教务普通聊天：不拦截，交给通用大模型回答；
        5. 教务业务问题：不拦截，后续进入 RAG / 权限过滤 / 流程节点。

    这样项目就从“只能答教务问题的机器人”
    升级成“通用大模型聊天底座 + 教务业务增强助手”。
    """

    normalized_query = query.strip()

    if not normalized_query:
        return GuardrailResult(
            allowed=False,
            reason="empty_query",
            message="请输入需要咨询的问题。",
        )

    if contains_any(normalized_query, SYSTEM_ABUSE_KEYWORDS):
        return GuardrailResult(
            allowed=False,
            reason="system_abuse",
            message="该问题涉及系统安全或违规操作，我不能提供相关帮助。",
        )

    if is_privacy_query(normalized_query):
        if allow_private_analysis and is_authorized_private_analysis_request(normalized_query):
            return GuardrailResult(
                allowed=True,
                privacy_detected=True,
                sanitized_text=sanitize_private_text(normalized_query),
            )

        return GuardrailResult(
            allowed=False,
            reason="privacy",
            message=(
                "该问题可能涉及学生个人隐私。"
                "如需查询本人信息，请通过学校官方教务系统或联系所在学院办理。"
            ),
        )

    return GuardrailResult(allowed=True)
