import json
from collections import Counter
from pathlib import Path

from app.rag.schemas import DocumentChunk, RetrievedChunk


# 这里先定义一组“高频教务意图关键词”。
#
# 为什么要加这个？
# 当前 JsonlRetriever 是本地关键词检索版，中文又是按单字切分。
# 用户问“转专业”时，问题会被切成“转 / 专 / 业”，
# 这会导致包含“转学”的文档也因为有“转”字而被召回。
#
# 所以我们额外加一层轻量规则：
# 1. 先判断用户问题属于哪个教务主题；
# 2. 检索时优先返回同主题资料；
# 3. 遇到明显相近但不是同一业务的资料时，降低或过滤。
#
# 这一步可以理解成“简化版意图识别 + 业务过滤”。
# 后面接 LangGraph 时，真正的意图识别节点也会做类似事情。
TOPIC_KEYWORDS = {
    "major_change": ["转专业"],
    "school_transfer": ["转学"],
    "course_select": ["选课", "退课", "重修", "补修"],
    "exam": ["缓考", "补考", "考试违纪", "考场规则"],
    "grade": ["成绩", "成绩复核", "成绩单"],
    "student_status": ["学籍", "休学", "复学", "退学", "保留学籍"],
    "graduation": ["毕业", "学位", "结业"],
    "calendar": ["校历", "教学周", "节假日", "调课"],
}


# 有些教务词很像，但业务含义完全不同。
# 例如：
# - 转专业：学生在校内从一个专业转到另一个专业；
# - 转学：学生从一个学校转到另一个学校。
#
# 如果用户明确问“转专业”，只包含“转学”的 chunk 就应该被过滤掉。
CONFLICT_TOPICS = {
    "major_change": ["school_transfer"],
    "school_transfer": ["major_change"],
}


def infer_topics(text: str) -> set[str]:
    """根据文本中的关键词，粗略判断它属于哪些教务主题。

    返回值是一个 set，例如：
    {"major_change"} 表示文本和“转专业”相关。

    这里不用模型，而是先用规则实现。
    原因是：
    1. 规则可解释，适合学习阶段；
    2. 不依赖额外模型；
    3. 能解决当前最明显的“转专业 / 转学”混淆问题。
    """
    topics = set()

    for topic, keywords in TOPIC_KEYWORDS.items():
        for keyword in keywords:
            if keyword in text:
                topics.add(topic)
                break

    return topics


def is_conflict_chunk(query_topics: set[str], chunk_topics: set[str]) -> bool:
    """判断当前 chunk 是否和用户问题存在明显业务冲突。

    举例：
    用户问题主题是 major_change（转专业），
    chunk 主题是 school_transfer（转学），
    并且 chunk 本身不包含 major_change，
    那么它就是明显冲突资料，应该过滤掉。
    """
    for query_topic in query_topics:
        conflict_topics = CONFLICT_TOPICS.get(query_topic, [])

        for conflict_topic in conflict_topics:
            if conflict_topic in chunk_topics and query_topic not in chunk_topics:
                return True

    return False


def topic_bonus(query_topics: set[str], chunk_topics: set[str]) -> float:
    """计算主题加分。

    如果用户问题和 chunk 属于同一个教务主题，
    就额外加分，让它排在更前面。
    """
    if not query_topics or not chunk_topics:
        return 0.0

    if query_topics & chunk_topics:
        return 0.35

    return 0.0


def tokenize(text: str) -> Counter[str]:
    """把文本拆成 token，并统计每个 token 出现次数。

    为什么要 tokenize？
    本地检索阶段，我们还没有接 embedding 模型和 Milvus。
    所以先用简单的关键词匹配来判断：
    用户问题和 chunk 文本有没有重合的字/词。

    中文处理：
    当前为了简单，中文按单个字符切分。
    例如：
    “转专业”会被切成 “转”、“专”、“业”。

    英文/数字处理：
    连续的英文和数字会合并成一个 token。
    例如：
    "Qwen2.5" 会保留成类似 qwen2、5 这样的片段。

    返回值 Counter：
    Counter 是一个计数字典。
    例如：
    Counter({"转": 1, "专": 1, "业": 1})
    """
    tokens = []

    # buffer 用来临时保存连续的英文和数字。
    # 比如遇到 q、w、e、n 时，先放进 buffer，
    # 等遇到中文或标点时，再把 qwen 整体加入 tokens。
    buffer = ""

    # 统一转小写，避免英文大小写影响匹配。
    for char in text.lower():
        # 如果是 ASCII 字符，并且是字母或数字，就先拼到 buffer 里。
        # ASCII 可以简单理解成英文字符范围。
        if char.isascii() and char.isalnum():
            buffer += char
            continue

        # 如果当前字符不是英文/数字，并且 buffer 里已有内容，
        # 说明一个英文/数字 token 结束了，需要加入 tokens。
        if buffer:
            tokens.append(buffer)
            buffer = ""

        # 如果当前字符不是空白字符，就作为一个 token。
        # 对中文来说，这里就是按单字切分。
        if char.strip():
            tokens.append(char)

    # 循环结束后，如果 buffer 里还有英文/数字，要补充加入 tokens。
    if buffer:
        tokens.append(buffer)

    # Counter 会统计每个 token 出现了多少次。
    return Counter(tokens)


def lexical_score(query_terms: Counter[str], doc_terms: Counter[str]) -> float:
    """计算用户问题和文档 chunk 的词面重合分数。

    query_terms:
    用户问题切分后的 token 统计。

    doc_terms:
    文档 chunk 切分后的 token 统计。

    计算方式：
    看用户问题里的 token，有多少也出现在文档 chunk 里。

    举例：
    用户问：“转专业材料”
    文档里包含：“转专业需要提交申请表和成绩单”
    那么 “转”、“专”、“业”、“材”、“料” 里，
    有一部分能在文档中找到，分数就会比较高。

    注意：
    这是一个非常简单的本地检索算法。
    它的目的不是替代 Milvus，而是在早期帮我们验证数据质量。
    """
    if not query_terms or not doc_terms:
        return 0.0

    # overlap 表示重合 token 的数量。
    # min(count, doc_terms.get(term, 0)) 是为了避免重复词过度加分。
    overlap = sum(
        min(count, doc_terms.get(term, 0))
        for term, count in query_terms.items()
    )

    # 用重合数量除以 query token 总数，得到 0 到 1 之间的分数。
    return overlap / max(sum(query_terms.values()), 1)


class JsonlRetriever:
    """基于 chunks.jsonl 的本地检索器。

    这个类的作用：
    1. 从 chunks.jsonl 里读取所有 chunk。
    2. 根据用户问题做本地关键词检索。
    3. 根据 tenant_id 和 visibility 做简单权限过滤。
    4. 返回最相关的 top_k 个 chunk。

    为什么先做 JsonlRetriever？
    因为它不依赖 Milvus、不依赖 embedding 模型。
    可以先验证数据和流程是否正确。
    """

    def __init__(self, chunks_paths: list[str]):
        # chunks_paths 是一个列表，因为我们可能同时加载多个 jsonl。
        # 例如：
        # - all_docs_chunks.jsonl 当前统一知识库切分结果
        # - 其他手动指定的 chunks.jsonl 测试文件
        self.chunks_paths = [Path(path) for path in chunks_paths]

        # 初始化时直接把 chunks 加载到内存。
        # 你的 PoC 数据量不大，这样最简单。
        # 生产环境数据量大时，就应该换成 Milvus。
        self.chunks = self._load_chunks()

    def _load_chunks(self) -> list[DocumentChunk]:
        """读取 chunks.jsonl 文件，把每一行转成 DocumentChunk。"""
        chunks = []

        # 遍历所有 chunks 文件。
        for chunks_path in self.chunks_paths:
            # 如果文件不存在，就跳过。
            # 这样可以避免某个 chunks 文件暂时没生成导致服务启动失败。
            if not chunks_path.exists():
                continue

            # chunks.jsonl 是 UTF-8 文本文件。
            # 每一行都是一个 JSON 对象。
            with chunks_path.open("r", encoding="utf-8") as file:
                for line in file:
                    # 跳过空行。
                    if not line.strip():
                        continue

                    # 把一行 JSON 字符串解析成 Python 字典。
                    row = json.loads(line)

                    # 把字典转成 DocumentChunk 对象。
                    chunks.append(
                        DocumentChunk(
                            chunk_id=row["chunk_id"],
                            text=row["text"],
                            source=row["source"],
                            tenant_id=row["tenant_id"],
                            department_id=row.get("department_id"),
                            visibility=row.get("visibility", "public"),
                            metadata=row.get("metadata", {}),
                        )
                    )

        return chunks

    def search(
        self,
        query: str,
        tenant_id: str,
        roles: list[str] | None = None,
        limit: int = 5,
    ) -> list[RetrievedChunk]:
        """搜索和 query 最相关的 chunks。

        参数说明：
        query:
        用户输入的问题。

        tenant_id:
        当前用户所在学校。
        用来避免检索到其他学校的数据。

        roles:
        当前用户角色。
        例如：student、teacher、admin。
        用来控制能看到哪些 visibility 的数据。

        limit:
        返回前几个结果，也就是 top_k。
        """
        # 如果没有传 roles，默认按学生处理。
        roles = roles or ["student"]

        # 默认所有人都可以看 public。
        allowed_visibility = ["public"]

        # 学生可以看 public 和 student。
        if "student" in roles:
            allowed_visibility.append("student")

        # 老师可以看 teacher。
        if "teacher" in roles:
            allowed_visibility.append("teacher")

        # 管理员可以看 teacher 和 internal。
        if "admin" in roles:
            allowed_visibility.extend(["teacher", "internal"])

        # 把用户问题切成 token。
        query_terms = tokenize(query)

        # 识别用户问题所属的教务主题。
        # 例如“转专业需要什么材料”会识别为 major_change。
        query_topics = infer_topics(query)

        results = []

        # 遍历所有 chunk，逐个计算相关性。
        for chunk in self.chunks:
            # 先做学校隔离。
            # 如果 chunk 不属于当前学校，就不能返回。
            if chunk.tenant_id != tenant_id:
                continue

            # 再做权限过滤。
            # 如果当前用户角色不能看这个 visibility，就跳过。
            if chunk.visibility not in allowed_visibility:
                continue

            # 把 chunk 正文和来源文件名拼在一起做主题判断。
            # 有些关键词可能只出现在文件名里，例如“模拟_转专业办理流程说明.md”。
            chunk_search_text = f"{chunk.source}\n{chunk.text}"
            chunk_topics = infer_topics(chunk_search_text)

            # 如果是明显冲突资料，直接过滤。
            # 例如问“转专业”时，过滤只讲“转学”的文档。
            if is_conflict_chunk(query_topics, chunk_topics):
                continue

            # 计算 query 和当前 chunk 的关键词重合分数。
            score = lexical_score(query_terms, tokenize(chunk.text))

            # 如果 chunk 和用户问题属于同一教务主题，额外加分。
            # 这样“转专业”资料会排在“学籍异动”等泛相关资料前面。
            score += topic_bonus(query_topics, chunk_topics)

            # 分数大于 0，说明至少有一点相关。
            if score > 0:
                results.append(
                    RetrievedChunk(
                        **chunk.__dict__,
                        score=score,
                    )
                )

        # 按分数从高到低排序，只返回前 limit 个。
        return sorted(results, key=lambda item: item.score, reverse=True)[:limit]
