from dataclasses import dataclass, field


# dataclass 是 Python 标准库提供的数据类工具。
# 它可以帮我们自动生成 __init__ 方法。
# 这样我们就不用手动写：
# def __init__(self, chunk_id, text, source, ...):
#     self.chunk_id = chunk_id
#     ...
#
# frozen=True 表示这个对象创建后不建议再修改。
# 对于文档 chunk 来说，这样更安全，因为 chunk 是知识库里的固定数据。
@dataclass(frozen=True)
class DocumentChunk:
    """表示知识库中的一段文档切片。

    为什么需要 chunk？
    一篇政策文件可能很长，大模型和检索系统不适合一次处理整篇文档。
    所以我们会先把长文档切成很多小段，每一小段就是一个 chunk。

    一个 chunk 至少要保存：
    - 它的唯一编号
    - 它的正文内容
    - 它来自哪个文件
    - 它属于哪所学校
    - 它的权限范围
    """

    # chunk 的唯一编号。
    # 例如：模拟_转专业办理流程说明.md:0
    # 含义是：这个 chunk 来自该文件，是第 0 个片段。
    chunk_id: str

    # chunk 的正文内容。
    # RAG 检索时真正参与匹配的是这个字段。
    # 后面组装 Prompt 时，也会把这个 text 提供给大模型。
    text: str

    # 来源文件名。
    # 例如：模拟_转专业办理流程说明.md
    # 这个字段非常重要，因为最终回答要给用户返回引用来源。
    source: str

    # 租户 ID，也可以理解成学校 ID。
    # 例如：generic-university
    # 如果以后系统服务多所学校，就靠 tenant_id 区分不同学校的数据。
    tenant_id: str

    # 学院 ID，可选字段。
    # 有些资料只适用于某个学院，比如计算机学院自己的通知。
    # 如果是全校通用资料，可以是 None。
    department_id: str | None = None

    # 可见范围。
    # 常见值：
    # public：公开资料，所有人可见
    # student：学生可见
    # teacher：教师可见
    # internal：教务内部资料
    visibility: str = "public"

    # 其他元数据。
    # 比如文档类型、适用对象、发布时间、原始路径等。
    # field(default_factory=dict) 的作用是：
    # 每个 DocumentChunk 都有自己独立的空字典，避免多个对象共用同一个 dict。
    metadata: dict = field(default_factory=dict)


# RetrievedChunk 继承 DocumentChunk。
# 意思是：被检索命中的结果本质上还是一个文档 chunk，
# 只是它比普通 chunk 多了一个 score 字段。
@dataclass(frozen=True)
class RetrievedChunk(DocumentChunk):
    """表示检索命中的文档切片。"""

    # 相关性分数。
    # 分数越高，说明这个 chunk 和用户问题越相关。
    # 当前本地检索版用的是关键词重合分数。
    # 后面接 Milvus 后，这个分数会变成向量相似度或 reranker 分数。
    score: float = 0.0