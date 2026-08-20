from pydantic import BaseModel, Field


class ChatHistoryItem(BaseModel):
    """
    单条会话历史。

    这里只保存短期会话记忆，不做数据库持久化。
    前端每次请求带最近几轮对话过来，后端用它辅助理解追问。
    """

    role: str = Field(description="消息角色，只建议使用 user 或 assistant")
    content: str = Field(
        min_length=1,
        max_length=2000,
        description="消息内容",
    )


class ChatRequest(BaseModel):
    """
    ChatRequest 表示前端或用户调用问答接口时提交的数据。

    你可以把它理解成“请求体说明书”：
    FastAPI 收到 JSON 后，会自动按照这个类校验字段。

    例如前端传入：
    {
        "query": "转专业需要什么材料",
        "top_k": 5,
        "tenant_id": "generic-university",
        "role": "student"
    }

    FastAPI 会自动检查：
    1. query 是否为空；
    2. top_k 是否在允许范围内；
    3. 没传 tenant_id / role 时是否使用默认值。
    """

    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="用户输入的教务咨询问题",
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=10,
        description="RAG 检索返回的文档片段数量",
    )

    tenant_id: str = Field(
        default="generic-university",
        description="租户 ID，用来区分不同学校或不同业务单位",
    )

    role: str = Field(
        default="student",
        description="用户角色，例如 student、teacher、admin",
    )


    allow_private_analysis: bool = Field(
        default=False,
        description="是否确认上传内容为本人材料，并授权系统仅用于本次脱敏分析",
    )


    history: list[ChatHistoryItem] = Field(
        default_factory=list,
        max_length=10,
        description="当前浏览器会话最近几轮对话，用于短期记忆；后端不持久化保存",
    )


class Citation(BaseModel):
    """
    Citation 表示答案引用的资料来源。

    RAG 项目里一定要返回引用来源。
    因为教务政策类问答不能只给一个“看起来像对的答案”，
    还要告诉用户答案依据来自哪个文档、哪个 chunk。
    """

    chunk_id: str = Field(description="被引用的文档片段 ID")
    source: str = Field(description="来源文件名")
    score: float = Field(description="检索相关性分数")
    text: str = Field(description="引用片段的文本预览")


class ChatResponse(BaseModel):
    """
    ChatResponse 表示问答接口返回给前端的数据结构。

    这里刻意不返回 debug.prompt。
    因为正式接口不应该把完整 Prompt 和全部检索资料暴露给用户。
    """

    query: str = Field(description="用户原始问题")
    answer: str = Field(description="智能助手生成的回答")
    citations: list[Citation] = Field(description="答案引用来源列表")
    retrieved_count: int = Field(description="本次实际检索到的 chunk 数量")
    candidate_count: int | None = Field(
        default=None,
        description="rerank 前的候选 chunk 数量",
    )
    reranked_count: int | None = Field(
        default=None,
        description="rerank 后保留的 chunk 数量",
    )
    intent: str | None = Field(
        default=None,
        description="意图编码，例如 major_change、grade_review",
    )
    intent_label: str | None = Field(
        default=None,
        description="意图中文名称，例如 转专业、成绩复核",
    )
    intent_route: str | None = Field(
        default=None,
        description="意图所属流程大类，例如 procedure_qa、policy_qa",
    )
    intent_reason: str | None = Field(
        default=None,
        description="意图识别原因，方便调试",
    )
    flow_name: str | None = Field(
        default=None,
        description="实际命中的 LangGraph 流程节点名称",
    )
    blocked: bool = Field(
        default=False,
        description="是否被 Guardrails 拦截",
    )
    guardrail_reason: str | None = Field(
        default=None,
        description="Guardrails 拦截原因；未拦截时为 None",
    )

    private_analysis_allowed: bool = Field(
        default=False,
        description="是否进入了用户授权的个人材料分析流程",
    )

    privacy_sanitized: bool = Field(
        default=False,
        description="是否已对用户输入中的个人敏感信息做脱敏处理",
    )

    memory_turns: int = Field(
        default=0,
        description="本次请求使用的历史消息数量",
    )
