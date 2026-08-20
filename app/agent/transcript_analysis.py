import re
from dataclasses import dataclass

from app.core.privacy import sanitize_private_text


@dataclass(frozen=True)
class CourseScore:
    """
    从 OCR 文本中粗略识别出来的一条课程成绩。

    OCR 输出经常不规整，所以这里只做轻量规则抽取：
    - course_line 保存整行文本，方便用户回看；
    - score 保存这一行里最像成绩的数字。
    """

    course_line: str
    score: float


def _extract_course_scores(text: str) -> list[CourseScore]:
    """
    从成绩单 OCR 文本中抽取疑似课程成绩。

    注意：
    - 这不是正式成绩解析器，只是项目演示阶段的规则版分析；
    - 真正商业落地时，建议按学校成绩单模板做表格结构化解析。
    """

    course_scores: list[CourseScore] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()

        if not line:
            continue

        numbers = re.findall(r"(?<!\d)(100(?:\.0+)?|[1-9]?\d(?:\.\d+)?)(?!\d)", line)
        if not numbers:
            continue

        score = float(numbers[-1])

        if len(line) < 4:
            continue

        course_scores.append(CourseScore(course_line=line, score=score))

    return course_scores[:30]


def build_transcript_private_analysis_answer(query: str) -> str:
    """
    生成“本人授权成绩单分析”答案。

    这个节点不查询 Milvus，也不引用学校政策文件。
    它只基于用户本次上传并授权分析的成绩单 OCR 文本，输出学习风险提示。
    为了保护隐私，答案中不会完整复述原始成绩单文本。
    """

    sanitized_query = sanitize_private_text(query)
    course_scores = _extract_course_scores(sanitized_query)

    low_scores = [item for item in course_scores if item.score < 60]
    warning_scores = [item for item in course_scores if 60 <= item.score < 70]
    excellent_scores = [item for item in course_scores if item.score >= 90]

    lines = [
        "直接结论：已按本人授权材料进行脱敏分析。以下结果仅基于本次上传文本，不能替代教务系统中的正式成绩记录。",
        "",
        "隐私处理：系统已在分析前隐藏姓名、学号、身份证号、手机号、邮箱和长数字编号。",
        "",
        "成绩概览：",
        f"1. 识别到疑似课程成绩 {len(course_scores)} 条。",
        f"2. 疑似低于 60 分课程 {len(low_scores)} 条。",
        f"3. 60 至 69 分需要关注课程 {len(warning_scores)} 条。",
        f"4. 90 分及以上优势课程 {len(excellent_scores)} 条。",
        "",
        "需要重点关注：",
    ]

    if low_scores:
        for item in low_scores[:5]:
            lines.append(f"- 可能未通过：{item.course_line}")
    else:
        lines.append("- 未从 OCR 文本中稳定识别到低于 60 分的课程。")

    if warning_scores:
        lines.append("")
        lines.append("临界风险课程：")
        for item in warning_scores[:5]:
            lines.append(f"- 建议复盘：{item.course_line}")

    lines.extend(
        [
            "",
            "建议操作：",
            "1. 先登录学校教务系统核对正式成绩、学分和绩点。",
            "2. 如果存在未通过课程，关注补考、重修或补修通知。",
            "3. 如果对成绩有疑问，在学校规定时间内申请成绩复核。",
            "4. 如果涉及毕业审核、奖助评定、转专业或出国材料，以学院和教务处正式要求为准。",
            "",
            "说明：OCR 可能识别错误，尤其是课程名、数字和表格列错位。请以学校官方系统显示结果为准。",
        ]
    )

    return "\n".join(lines)

