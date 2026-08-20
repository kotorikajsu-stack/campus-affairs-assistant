import re


def is_authorized_private_analysis_request(text: str) -> bool:
    """
    判断这次请求是不是“用户主动上传成绩单后，希望系统分析本人材料”。

    这里不能只看前端有没有传 allow_private_analysis。
    因为用户也可能勾选授权后输入“帮我查张三成绩”，那仍然属于越权查询。

    因此放行条件必须同时满足：
    1. 文本里像成绩单、成绩、课程、学分等材料内容；
    2. 用户是在要求分析、总结、给建议，而不是查询他人隐私。
    """

    normalized_text = text.strip()

    transcript_words = [
        "成绩单",
        "成绩",
        "课程",
        "学分",
        "绩点",
        "GPA",
        "gpa",
    ]
    analysis_words = [
        "这份",
        "上传",
        "根据",
        "总结",
        "分析",
        "关注",
        "建议",
        "风险",
        "帮我看看",
    ]

    has_transcript_signal = any(word in normalized_text for word in transcript_words)
    has_analysis_signal = any(word in normalized_text for word in analysis_words)

    return has_transcript_signal and has_analysis_signal


def sanitize_private_text(text: str) -> str:
    """
    对用户上传材料中的个人敏感信息做脱敏。

    设计原则：
    - 保留课程名、成绩、学分等用于分析的业务信息；
    - 隐藏姓名、学号、身份证号、手机号、邮箱等身份标识；
    - 不追求百分百 OCR 纠错，先覆盖 PoC 阶段最常见的隐私字段。
    """

    sanitized = text

    sanitized = re.sub(
        r"(姓名\s*[:：]?\s*)[\u4e00-\u9fa5·]{2,8}",
        r"\1某同学",
        sanitized,
    )
    sanitized = re.sub(
        r"(学号|考号|身份证号|身份证|证件号)\s*[:：]?\s*[A-Za-z0-9]{6,24}",
        r"\1：[已脱敏]",
        sanitized,
        flags=re.IGNORECASE,
    )
    sanitized = re.sub(r"1[3-9]\d{9}", "[手机号已脱敏]", sanitized)
    sanitized = re.sub(
        r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
        "[邮箱已脱敏]",
        sanitized,
    )

    # 连续 8 到 18 位数字很可能是学号、证件号、编号。
    # 成绩一般是 1 到 3 位数字，所以不会被这条规则误伤。
    sanitized = re.sub(r"\b\d{8,18}\b", "[长数字已脱敏]", sanitized)

    return sanitized

