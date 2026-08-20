from app.rag.schemas import RetrievedChunk


def _format_history(history: list | None, max_items: int = 6) -> str:
    """
    把短期会话历史整理成 Prompt 片段。

    history 可能来自 FastAPI 的 Pydantic 对象，也可能来自 LangGraph
    State 中的普通 dict，所以这里兼容两种读取方式。
    """

    if not history:
        return "无"

    formatted_items = []

    for item in history[-max_items:]:
        if isinstance(item, dict):
            role = item.get("role", "user")
            content = item.get("content", "")
        else:
            role = getattr(item, "role", "user")
            content = getattr(item, "content", "")

        role_label = "用户" if role == "user" else "助手"
        content = str(content).strip()

        if content:
            formatted_items.append(f"{role_label}：{content[:800]}")

    return "\n".join(formatted_items) if formatted_items else "无"


def build_rag_prompt(
    question: str,
    contexts: list[RetrievedChunk],
    history: list | None = None,
) -> str:
    """根据用户问题和 RAG 检索结果组装 Prompt。

    Prompt 是 RAG 系统里非常核心的一步。
    它会把下面几类信息合并成一段大模型能理解的输入：

    1. 角色设定：
       告诉模型它是校园教务智能助手。

    2. 回答规则：
       告诉模型必须基于参考资料回答，不能编造政策。

    3. 用户问题：
       用户真正想问的内容。

    4. 参考资料：
       Milvus 或 JSONL 检索出来的 chunks。

    真实接入 Qwen 时，模型看到的就是这个 prompt。
    """

    if contexts:
        reference_text = "\n\n".join(
            f"[{index}] 来源：{chunk.source}\n"
            f"chunk_id：{chunk.chunk_id}\n"
            f"内容：{chunk.text}"
            for index, chunk in enumerate(contexts, start=1)
        )
    else:
        reference_text = "未检索到可引用资料。"

    history_text = _format_history(history)

    prompt = f"""
你是高校通用模拟数据校园教务智能助手。

回答规则：
1. 只能根据【参考资料】回答。
2. 如果参考资料中没有明确依据，请说“当前资料未找到明确依据”。
3. 回答要像正常聊天一样自然，不要固定使用“直接答案：”作为开头。
4. 不要在正文里输出“引用来源：”“来源文件：”“参考资料：”等引用清单；引用资料会由系统在回答下方单独展示。
5. 可以按问题需要使用“申请条件”“所需材料”“办理流程”“注意事项”等小标题。
6. 不要编造政策，不要承诺一定能办理成功。
7. 如果涉及学生个人信息、成绩、学籍状态等隐私信息，需要提示用户通过学校官方系统查询。
8. 如果参考资料标注为模拟资料，只需提醒正式办理以学校教务处或学院正式通知为准，不要列出引用来源。

用户问题：
{question}

历史对话：
{history_text}

参考资料：
{reference_text}

请直接回答用户问题。不要输出“直接答案：”和“引用来源：”这两个栏目。
""".strip()

    return prompt


def build_general_chat_prompt(question: str, history: list | None = None) -> str:
    """构建通用聊天 Prompt。

    这个 Prompt 不携带 RAG 参考资料。

    使用场景：
        用户问的不是教务业务问题，例如：
        - 你好
        - 帮我解释一下 RAG 是什么
        - 帮我写一段学习计划

    设计原则：
        1. 保留通用大模型聊天能力；
        2. 仍然带上基础安全约束；
        3. 如果用户转向教务政策、办事流程、成绩、学籍等问题，
           提醒它应该基于学校资料回答，而不是编造。
    """

    history_text = _format_history(history)

    return f"""
你是一个友好、可靠的校园智能助手。

当前用户问题不属于明确的教务办事业务，因此你可以按通用大模型助手的方式回答。

回答规则：
1. 可以回答通用学习、生活、技术概念、写作和普通聊天问题。
2. 不要编造学校政策、办理条件、办理地点或办理时间。
3. 如果用户询问教务政策、办事流程、成绩、学籍、毕业、选课等学校业务，应提示需要基于学校正式资料或教务系统信息确认。
4. 不要帮助查询、泄露或推断他人的成绩、学号、身份证号、联系方式等隐私信息。
5. 不要提供攻击系统、绕过权限、修改成绩等违规操作帮助。

用户问题：
{question}

历史对话：
{history_text}
""".strip()
