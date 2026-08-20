from typing import Any, TypedDict

from app.rag.schemas import RetrievedChunk


class ChatState(TypedDict, total=False):
    """
    ChatState 是 LangGraph 工作流中的“状态对象”。

    你可以把它理解成一张流程表：
    每个节点执行完以后，都会往这张表里写入新的字段；
    下一个节点再从这张表里读取自己需要的数据。

    为什么 LangGraph 需要 State？
    因为 Agent 工作流不是一个普通函数从上到下执行完就结束。
    它通常会经历多个节点：

    guardrails_node
      -> retrieve_node
      -> prompt_node
      -> generate_node
      -> finalize_node

    每个节点都只负责一件事，
    节点之间就通过 ChatState 传递数据。

    total=False 的意思是：
    不是所有字段一开始都必须存在。
    例如刚进入工作流时，只有 query、top_k、tenant_id、role；
    检索节点执行后，才会出现 contexts；
    生成节点执行后，才会出现 answer。
    """

    # 用户原始问题。
    query: str

    # RAG 检索时返回几个 chunk。
    top_k: int

    # 租户 ID，也就是学校 ID。
    # 以后系统服务多所学校时，可以用它隔离不同学校的数据。
    tenant_id: str

    # 用户角色。
    # 当前先用 student / teacher / admin 这种简单字符串。
    role: str

    # 用户是否确认上传内容为本人材料，并授权系统仅用于本次脱敏分析。
    allow_private_analysis: bool

    # 用户授权后，系统是否进入了个人材料分析流程。
    private_analysis_allowed: bool

    # 进入模型或分析节点前是否已经做过脱敏处理。
    privacy_sanitized: bool

    # 脱敏后的用户输入。需要处理成绩单等个人材料时，优先使用这个字段。
    sanitized_query: str

    # 会话级短期记忆。由前端随请求带最近几轮对话，后端不持久化保存。
    history: list[dict[str, str]]

    # 权限过滤时使用的角色列表。
    # retriever.search() 现在接收的是 list[str]。
    roles: list[str]

    # 意图编码。
    # 例如：
    # - major_change：转专业
    # - grade_review：成绩复核
    # - exam_affairs：考试与缓补考
    intent: str

    # 意图中文名称。
    # 例如：“转专业”“成绩复核”。
    intent_label: str

    # 工作流大类。
    # 当前包括：
    # - procedure_qa：办事流程类
    # - policy_qa：政策问答类
    # - form_qa：表格材料类
    # - status_guide：查询指引类
    # - general_qa：通用问答类
    intent_route: str

    # 意图识别原因。
    # 用来调试和解释为什么识别成某个意图。
    intent_reason: str

    # 增强后的检索 query。
    # 例如用户问“转专业需要什么材料”，
    # 意图节点会补充“转专业 申请表 办理流程 申请条件 材料”等词，
    # 让本地关键词检索更容易命中正确资料。
    retrieval_query: str

    # 实际命中的流程节点名称。
    # 例如：
    # - major_change_flow：转专业专用流程
    # - generic_rag：普通 RAG 问答流程
    #
    # 这个字段方便你调试 LangGraph 到底走了哪条边。
    flow_name: str

    # Guardrails 是否拦截。
    blocked: bool

    # 如果被拦截，记录拦截原因。
    # 例如：privacy、system_abuse、out_of_scope。
    guardrail_reason: str | None

    # RAG 检索出来的文档片段。
    contexts: list[RetrievedChunk]

    # rerank 前的候选 chunk 数量。
    # retrieve_node 会先多召回一些候选结果，
    # rerank_node 再从里面筛出最终 top_k。
    candidate_count: int

    # rerank 后实际保留的 chunk 数量。
    reranked_count: int

    # 根据 query + contexts 组装出来的 Prompt。
    # MockLLMClient 暂时不用它，但真实 Qwen 会用它。
    prompt: str

    # 模型最终生成的回答。
    answer: str

    # 引用来源。
    # 这里用 list[dict[str, Any]]，是为了方便直接返回给 FastAPI。
    citations: list[dict[str, Any]]

    # 实际检索到的 chunk 数量。
    retrieved_count: int
