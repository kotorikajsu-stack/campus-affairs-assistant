from app.rag.schemas import RetrievedChunk


# 意图感知 rerank 规则表。
#
# 当前还是规则版 reranker，不调用额外模型。
# 它的作用是：
# 1. 根据 intent 给强相关 chunk 加分；
# 2. 给明显偏题 chunk 降分；
# 3. 在必要时过滤掉“有一点关键词重合但业务明显不相关”的 chunk。
#
# 为什么现在先写规则版？
# 因为你的项目当前还处于本地 RAG PoC 阶段。
# 规则版好理解、好调试，也能明显改善 citations 质量。
#
# 后面如果接入真正的 reranker 模型，比如 bge-reranker，
# 可以保留这个文件的函数名，只替换内部实现。
INTENT_RERANK_RULES = {
    "major_change": {
        "strong": ["转专业", "转入专业", "换专业", "申请表", "拟转入", "接收专业"],
        "weak": ["专业", "申请", "材料", "考核", "学院"],
        "negative": ["转学", "转入学校", "转出学校"],
    },
    "grade_review": {
        "strong": ["成绩复核", "复核申请", "复核成绩", "成绩有误", "漏评", "登分错误"],
        "weak": ["成绩", "课程成绩", "任课教师", "成绩管理"],
        "negative": ["结业", "毕业证书", "学籍注册", "休学", "复学", "实践教学", "毕业论文"],
    },
    "transcript": {
        "strong": ["成绩单", "中文成绩单", "英文成绩单", "打印", "盖章", "密封"],
        "weak": ["证明", "办理", "教务系统", "毕业生", "在校生"],
        "negative": ["实践教学", "实习", "实训", "毕业论文", "课程设计"],
    },
    "exam_affairs": {
        "strong": ["缓考", "补考", "考试违纪", "考场规则", "缺考"],
        "weak": ["考试", "申请表", "证明材料", "考核"],
        "negative": ["成绩单", "转专业", "毕业论文", "实习"],
    },
    "student_status_change": {
        "strong": ["学籍异动", "休学", "复学", "退学", "保留学籍", "延长学习年限"],
        "weak": ["学籍", "申请表", "证明材料", "学院审核"],
        "negative": ["成绩单", "缓考", "转专业", "毕业论文"],
    },
    "certificate_service": {
        "strong": ["学生证", "在读证明", "学籍证明", "证明办理", "补办"],
        "weak": ["证明", "盖章", "办理", "身份核验", "份数"],
        "negative": ["成绩复核", "缓考", "重修", "实践教学"],
    },
    "course_registration": {
        "strong": ["选课", "退课", "重修", "补修", "补选", "课程替代", "学分认定"],
        "weak": ["课程", "学分", "培养方案", "教学班", "教务系统"],
        "negative": ["成绩单", "学生证", "在读证明", "毕业论文"],
    },
}


def _count_hits(text: str, keywords: list[str]) -> int:
    """
    统计文本命中了多少个关键词。

    这里不是统计出现次数，而是统计命中了几个不同关键词。
    例如文本里出现很多次“成绩”，也只算一个关键词命中。

    这样可以避免某个词重复出现导致分数虚高。
    """

    return sum(1 for keyword in keywords if keyword in text)


def rerank_contexts(
    intent: str | None,
    contexts: list[RetrievedChunk],
    limit: int,
) -> list[RetrievedChunk]:
    """
    对 RAG 召回结果做意图感知 rerank。

    参数：
    intent:
        意图识别节点得到的意图编码。
        例如 transcript、grade_review、major_change。

    contexts:
        retrieve_node 召回的候选 chunks。

    limit:
        最终保留几个 chunks，也就是用户请求里的 top_k。

    返回：
        排序和过滤后的 RetrievedChunk 列表。

    工作逻辑：
    1. 如果当前 intent 没有配置 rerank 规则，就保持原结果；
    2. 对每个 chunk 计算 strong / weak / negative 命中数；
    3. 强相关关键词加分；
    4. 弱相关关键词小幅加分；
    5. 负向关键词扣分；
    6. 如果一个 chunk 命中了负向词，又完全没命中强相关词，就过滤掉。
    """

    if not contexts:
        return []

    if limit <= 0:
        return []

    rules = INTENT_RERANK_RULES.get(intent or "")

    # 没有规则的通用意图，保持原排序并截断。
    if not rules:
        return contexts[:limit]

    reranked_items = []

    for chunk in contexts:
        # source + text 一起参与判断。
        # 因为很多资料的主题会体现在文件名里，
        # 例如“模拟_成绩单办理指南.md”。
        searchable_text = f"{chunk.source}\n{chunk.text}"

        strong_hits = _count_hits(searchable_text, rules["strong"])
        weak_hits = _count_hits(searchable_text, rules["weak"])
        negative_hits = _count_hits(searchable_text, rules["negative"])

        # 明显偏题过滤：
        # 如果一个 chunk 命中负向词，同时没有任何强相关命中，
        # 说明它大概率只是因为泛词被召回，不适合给用户当引用。
        #
        # 例子：
        # 用户问“成绩单怎么打印”，
        # “实践教学实习实训毕业论文管理办法”里可能有“成绩”二字，
        # 但它命中“实习/毕业论文”等负向词，且不含“成绩单/打印”，
        # 就应该过滤。
        if negative_hits > 0 and strong_hits == 0:
            continue

        rerank_score = (
            chunk.score
            + strong_hits * 0.35
            + weak_hits * 0.12
            - negative_hits * 0.4
        )

        # 如果文件名本身命中强相关关键词，再额外加一点分。
        # 文件名通常比正文更能代表文档主题。
        source_strong_hits = _count_hits(chunk.source, rules["strong"])
        rerank_score += source_strong_hits * 0.25

        reranked_items.append(
            RetrievedChunk(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                source=chunk.source,
                tenant_id=chunk.tenant_id,
                department_id=chunk.department_id,
                visibility=chunk.visibility,
                metadata=chunk.metadata,
                score=rerank_score,
            )
        )

    # 如果过滤太狠导致没有结果，就退回原始结果。
    # 这样可以避免用户明明问了教务问题，却因为规则不完整而完全没有引用。
    if not reranked_items:
        return contexts[:limit]

    return sorted(reranked_items, key=lambda item: item.score, reverse=True)[:limit]
