import inspect
from typing import Any

from langgraph.graph import END, START, StateGraph

from app.agent.intent import classify_intent
from app.agent.procedure_flows import (
    build_certificate_service_answer,
    build_course_registration_answer,
    build_exam_affairs_answer,
    build_grade_review_answer,
    build_major_change_answer,
    build_student_status_change_answer,
    build_transcript_answer,
)
from app.agent.state import ChatState
from app.agent.transcript_analysis import build_transcript_private_analysis_answer
from app.core.guardrails import check_input_guardrails
from app.core.privacy import is_authorized_private_analysis_request, sanitize_private_text
from app.llm.prompts import build_general_chat_prompt, build_rag_prompt
from app.rag.reranker import rerank_contexts


FOLLOW_UP_WORDS = [
    "这个",
    "那个",
    "上面",
    "刚才",
    "前面",
    "它",
    "这些",
    "继续",
    "还要",
    "还需要",
    "还有",
    "具体",
    "详细",
    "流程",
    "材料",
    "条件",
    "怎么办",
    "怎么做",
    "呢",
]


def _format_history_for_routing(history: list[dict[str, str]], max_items: int = 6) -> str:
    """
    把短期历史压缩成给意图识别和检索使用的上下文。

    这里不需要完整 Prompt，只需要最近几条消息帮助理解“这个/刚才/还需要什么材料”等追问。
    """

    if not history:
        return ""

    parts = []
    for item in history[-max_items:]:
        role = item.get("role", "user")
        content = str(item.get("content", "")).strip()
        if content:
            parts.append(f"{role}: {content[:300]}")

    return "\n".join(parts)


def _is_follow_up_query(query: str) -> bool:
    """判断当前问题是否像是在追问上一轮上下文。"""

    normalized_query = query.strip()

    return any(word in normalized_query for word in FOLLOW_UP_WORDS)


def _build_contextual_query(state: ChatState) -> str:
    """
    构建“带短期记忆”的当前问题。

    只有当前问题明显像追问时才拼接历史。
    这样可以减少误判：例如用户聊完转专业后又说“你好”，不会被历史强行拉回转专业流程。
    """

    query = state["query"]
    history = state.get("history", [])

    if not history or not _is_follow_up_query(query):
        return query

    history_text = _format_history_for_routing(history)

    if not history_text:
        return query

    return f"{history_text}\n当前追问：{query}"


def _call_llm_generate(llm: Any, state: ChatState) -> str:
    """
    兼容不同 LLM 客户端的 generate() 调用方式。

    你现在项目里用的是 MockLLMClient：

        generate(question: str, contexts: list[RetrievedChunk]) -> str

    后面接真实 Qwen 时，常见写法会变成：

        generate(prompt: str) -> str

    为了让 LangGraph 工作流现在能跑，
    后面也方便换真实大模型，
    这里写一个小适配器：

    1. 如果 llm.generate() 支持 prompt 参数，就传 prompt；
    2. 否则就按 MockLLMClient 的 question + contexts 调用。
    """

    generate_params = inspect.signature(llm.generate).parameters

    if "prompt" in generate_params:
        return llm.generate(prompt=state["prompt"])

    return llm.generate(
        question=state["query"],
        contexts=state.get("contexts", []),
    )


def _call_general_chat_generate(llm: Any, state: ChatState) -> str:
    """调用通用聊天模型。

    通用聊天分支不使用 RAG contexts。

    如果当前 LLM 是 QwenLLMClient：
        它的 generate() 支持 prompt 参数，
        所以直接把通用聊天 Prompt 传给模型。

    如果当前 LLM 是 MockLLMClient：
        它只支持 question + contexts，
        这里就返回一段模拟通用聊天回答，
        保证不接真实模型时项目也能跑通。
    """

    generate_params = inspect.signature(llm.generate).parameters

    if "prompt" in generate_params:
        return llm.generate(prompt=state["prompt"])

    return (
        f"通用回答：你问的是“{state['query']}”。\n\n"
        "当前后端使用 MockLLMClient，所以这里只返回模拟回答。"
        "如果切换到 Qwen，本分支会保留通用大模型聊天能力；"
        "当你询问转专业、成绩单、缓考、学籍等教务业务时，系统会自动进入 RAG 和业务流程。"
    )


def build_chat_graph(retriever: Any, llm: Any):
    """
    构建校园教务助手的 LangGraph 工作流。

    这个函数接收两个外部组件：

    retriever:
        RAG 检索器。
        当前传入的是 JsonlRetriever。
        后面换 MilvusRetriever 时，工作流不用大改。

    llm:
        大模型客户端。
        当前传入的是 MockLLMClient。
        后面可以换成 QwenLLMClient。

    工作流结构：

        START
          |
          v
    guardrails_node
          |
          |-- blocked --> END
          |
          |-- passed --> intent_node
                          |
                          v
                     retrieve_node
                          |
                          v
                      rerank_node
                          |
                          |-- major_change -----------> major_change_flow_node
                          |-- grade_review -----------> grade_review_flow_node
                          |-- transcript -------------> transcript_flow_node
                          |-- exam_affairs -----------> exam_affairs_flow_node
                          |-- student_status_change --> student_status_change_flow_node
                          |-- certificate_service ----> certificate_service_flow_node
                          |-- course_registration ----> course_registration_flow_node
                          |
                          |-- generic --> prompt_node
                                          |
                                          v
                                     generate_node
                                          |
                                          v
                                     finalize_node

    也就是说：
    - 转专业、成绩复核、成绩单、考试、学籍异动、证明办理、选课类问题会进入专用流程节点；
    - 其他问题继续走普通 RAG 生成链路。

    每个节点只做一件事：
    - guardrails_node：判断问题能不能回答；
    - intent_node：识别用户咨询的教务事项；
    - retrieve_node：检索相关资料；
    - rerank_node：根据意图重排和过滤检索结果；
    - major_change_flow_node：生成转专业标准流程答案；
    - grade_review_flow_node：生成成绩复核标准流程答案；
    - transcript_flow_node：生成成绩单办理标准流程答案；
    - exam_affairs_flow_node：生成考试缓补考标准流程答案；
    - student_status_change_flow_node：生成学籍异动标准流程答案；
    - certificate_service_flow_node：生成证明和学生证办理标准流程答案；
    - course_registration_flow_node：生成选课退课重修补修标准流程答案；
    - prompt_node：组装 Prompt；
    - generate_node：调用 LLM 生成答案；
    - finalize_node：整理 citations 和返回字段。
    """

    def guardrails_node(state: ChatState) -> ChatState:
        """
        输入安全检查节点。

        这是整个工作流的第一站。
        如果用户问题涉及隐私、系统攻击、越界话题，
        就在这里直接拦截，不进入 RAG。
        """

        allow_private_analysis = state.get("allow_private_analysis", False)
        guardrail = check_input_guardrails(
            state["query"],
            allow_private_analysis=allow_private_analysis,
        )

        if not guardrail.allowed:
            return {
                "blocked": True,
                "guardrail_reason": guardrail.reason,
                "answer": guardrail.message or "该问题暂时无法回答。",
                "contexts": [],
                "citations": [],
                "retrieved_count": 0,
                "private_analysis_allowed": False,
                "privacy_sanitized": False,
            }

        return {
            "blocked": False,
            "guardrail_reason": None,
            "private_analysis_allowed": guardrail.privacy_detected and allow_private_analysis,
            "privacy_sanitized": guardrail.sanitized_text is not None,
            "sanitized_query": guardrail.sanitized_text or sanitize_private_text(state["query"]),
        }

    def route_after_guardrails(state: ChatState) -> str:
        """
        Guardrails 后面的条件路由函数。

        LangGraph 的条件边需要一个函数来决定下一步走哪里。

        如果 blocked=True：
            说明问题被拦截，直接结束。

        如果 blocked=False：
            说明问题可以回答，继续进入检索节点。
        """

        if state.get("blocked"):
            return "blocked"

        return "passed"

    def intent_node(state: ChatState) -> ChatState:
        """
        意图识别节点。

        这个节点是教务 Agent 和普通 RAG 问答的关键区别之一。

        普通 RAG：
            用户问什么，就直接拿原问题去检索。

        教务 Agent：
            先判断用户问的是哪一类教务事项，
            再决定后续怎么检索、怎么组织流程答案。

        举例：
        - “转专业需要什么材料” -> major_change
        - “成绩怎么复核” -> grade_review
        - “学生证丢了怎么办” -> certificate_service

        当前这个节点先只做“识别 + 检索增强”。
        后面我们可以继续扩展成：
        - major_change_flow_node：转专业流程节点；
        - grade_review_flow_node：成绩复核流程节点；
        - exam_affairs_flow_node：缓考补考流程节点。
        """

        contextual_query = _build_contextual_query(state)
        intent = classify_intent(contextual_query)

        return {
            "intent": intent.intent,
            "intent_label": intent.label,
            "intent_route": intent.route,
            "intent_reason": intent.reason,
            "retrieval_query": intent.retrieval_query,
            "contextual_query": contextual_query,
        }

    def route_after_intent(state: ChatState) -> str:
        """意图识别后的分流函数。

        general_chat：
            普通聊天，不查知识库，不做 RAG，
            直接进入通用大模型聊天节点。

        academic：
            教务业务或教务泛问，
            继续进入 retrieve -> rerank -> 业务流程 / RAG。
        """

        if (
            state.get("private_analysis_allowed")
            and is_authorized_private_analysis_request(state.get("query", ""))
        ):
            return "transcript_private_analysis"

        if state.get("intent") == "general_chat":
            return "general_chat"

        return "academic"

    def retrieve_node(state: ChatState) -> ChatState:
        """
        RAG 检索节点。

        这个节点只负责一件事：
        根据用户问题，从知识库 chunks 中找出最相关的资料片段。

        注意：
        这里仍然会传 tenant_id 和 roles。
        这表示检索阶段就会做学校隔离和权限过滤。
        """

        roles = [state.get("role", "student")]

        # 优先使用意图节点生成的 retrieval_query。
        #
        # 例如原问题是：
        # “转专业需要什么材料”
        #
        # 意图节点会增强成：
        # “转专业需要什么材料 转专业 申请表 办理流程 申请条件 材料 考核”
        #
        # 这样本地关键词检索更容易召回“转专业申请表”和“转专业办理流程”，
        # 不容易被“转学”等相近但不同业务污染。
        retrieval_query = state.get("retrieval_query") or state["query"]

        # retrieve 阶段先“多召回”。
        #
        # 为什么不直接只召回 top_k？
        # 因为 rerank 需要有候选空间。
        # 如果用户 top_k=3，而我们检索也只拿 3 条，
        # 那 rerank 最多只能在这 3 条里排序，没法把第 4、5、6 条中更相关的结果提上来。
        #
        # 所以这里先取 max(top_k * 4, 10) 条候选，
        # 然后交给 rerank_node 过滤回 top_k。
        top_k = state.get("top_k", 5)
        candidate_limit = max(top_k * 4, 10)

        contexts = retriever.search(
            query=retrieval_query,
            tenant_id=state.get("tenant_id", "generic-university"),
            roles=roles,
            limit=candidate_limit,
        )

        return {
            "roles": roles,
            "contexts": contexts,
            "candidate_count": len(contexts),
            "retrieved_count": len(contexts),
        }

    def rerank_node(state: ChatState) -> ChatState:
        """
        意图感知 rerank 节点。

        这个节点位于：

            retrieve -> rerank -> flow_node

        它解决的问题是：
        本地关键词检索会把一些“有少量关键词重合但业务不相关”的资料召回。

        例如用户问“成绩单怎么打印”：
        - “成绩单办理指南”应该排前面；
        - “实践教学实习实训毕业论文管理办法”虽然可能有“成绩”二字，但不应该作为核心引用。

        rerank_node 会根据 intent 调用 rerank_contexts()，
        对候选 chunks 重新打分和过滤。
        """

        top_k = state.get("top_k", 5)

        reranked_contexts = rerank_contexts(
            intent=state.get("intent"),
            contexts=state.get("contexts", []),
            limit=top_k,
        )

        return {
            "contexts": reranked_contexts,
            "reranked_count": len(reranked_contexts),
            "retrieved_count": len(reranked_contexts),
        }

    def route_after_retrieve(state: ChatState) -> str:
        """
        Rerank 后的条件路由函数。

        为什么在 rerank 后面分流？

        因为流程专用节点通常也需要参考 RAG 检索结果。
        例如“转专业需要什么材料”：
        1. 先检索转专业相关 chunks；
        2. 再根据意图 rerank；
        3. 再由 major_change_flow_node 按固定栏目组织答案。

        如果以后你要做“成绩复核流程节点”，
        也可以在这里继续增加判断。
        """

        flow_intents = {
            "major_change",
            "grade_review",
            "transcript",
            "exam_affairs",
            "student_status_change",
            "certificate_service",
            "course_registration",
        }

        intent = state.get("intent")

        if intent in flow_intents:
            return intent

        return "generic"

    def major_change_flow_node(state: ChatState) -> ChatState:
        """
        转专业专用流程节点。

        这个节点不再让 MockLLM 自由拼接 chunks，
        而是调用 build_major_change_answer() 生成稳定结构：

        1. 直接答案；
        2. 申请条件；
        3. 所需材料；
        4. 办理流程；
        5. 注意事项；
        6. 引用来源。

        这就是“有限状态机 Agent”的价值：
        高频、标准化业务不用每次都交给大模型自由发挥，
        而是进入可控流程节点。
        """

        answer = build_major_change_answer(
            query=state["query"],
            contexts=state.get("contexts", []),
        )

        return {
            "answer": answer,
            "flow_name": "major_change_flow",
        }

    def grade_review_flow_node(state: ChatState) -> ChatState:
        """
        成绩复核专用流程节点。

        它会把“成绩怎么复核”这类问题稳定整理成：
        适用情形、所需材料、办理流程、注意事项。
        """

        answer = build_grade_review_answer(
            query=state["query"],
            contexts=state.get("contexts", []),
        )

        return {
            "answer": answer,
            "flow_name": "grade_review_flow",
        }

    def transcript_flow_node(state: ChatState) -> ChatState:
        """
        成绩单办理专用流程节点。

        它负责回答成绩单打印、盖章、英文成绩单等办理类问题。
        """

        answer = build_transcript_answer(
            query=state["query"],
            contexts=state.get("contexts", []),
        )

        return {
            "answer": answer,
            "flow_name": "transcript_flow",
        }

    def transcript_private_analysis_node(state: ChatState) -> ChatState:
        """
        本人授权成绩单分析节点。

        这个节点处理用户上传 OCR 后产生的成绩单文本。
        它和“成绩单怎么办理/怎么打印”的 transcript_flow 不一样：
        - transcript_flow 回答办事流程，需要 RAG 引用学校材料；
        - transcript_private_analysis 只分析用户本人授权上传的材料，不查 Milvus，不返回引用。
        """

        answer = build_transcript_private_analysis_answer(
            query=state.get("sanitized_query") or state["query"],
        )

        return {
            "answer": answer,
            "flow_name": "transcript_private_analysis",
            "contexts": [],
            "candidate_count": 0,
            "reranked_count": 0,
            "retrieved_count": 0,
            "private_analysis_allowed": True,
            "privacy_sanitized": True,
        }

    def exam_affairs_flow_node(state: ChatState) -> ChatState:
        """
        考试与缓补考专用流程节点。

        它负责回答缓考、补考、考试违纪、考场规则等问题。
        """

        answer = build_exam_affairs_answer(
            query=state["query"],
            contexts=state.get("contexts", []),
        )

        return {
            "answer": answer,
            "flow_name": "exam_affairs_flow",
        }

    def student_status_change_flow_node(state: ChatState) -> ChatState:
        """
        学籍异动专用流程节点。

        它负责回答休学、复学、退学、保留学籍等问题。
        """

        answer = build_student_status_change_answer(
            query=state["query"],
            contexts=state.get("contexts", []),
        )

        return {
            "answer": answer,
            "flow_name": "student_status_change_flow",
        }

    def certificate_service_flow_node(state: ChatState) -> ChatState:
        """
        证明和学生证办理专用流程节点。

        它负责回答学生证补办、在读证明、学籍证明等问题。
        """

        answer = build_certificate_service_answer(
            query=state["query"],
            contexts=state.get("contexts", []),
        )

        return {
            "answer": answer,
            "flow_name": "certificate_service_flow",
        }

    def course_registration_flow_node(state: ChatState) -> ChatState:
        """
        选课退课重修补修专用流程节点。

        它负责回答选课、退课、重修、补修、课程替代、学分认定等问题。
        """

        answer = build_course_registration_answer(
            query=state["query"],
            contexts=state.get("contexts", []),
        )

        return {
            "answer": answer,
            "flow_name": "course_registration_flow",
        }

    def general_chat_prompt_node(state: ChatState) -> ChatState:
        """通用聊天 Prompt 节点。

        这个节点不使用 RAG 检索资料。
        它只根据用户原始问题构建通用聊天 Prompt。
        """

        prompt = build_general_chat_prompt(
            question=state["query"],
            history=state.get("history", []),
        )

        return {
            "prompt": prompt,
            "contexts": [],
            "candidate_count": 0,
            "reranked_count": 0,
            "retrieved_count": 0,
        }

    def general_chat_generate_node(state: ChatState) -> ChatState:
        """通用聊天生成节点。

        Qwen 模式：
            调用真实 Qwen 进行普通聊天。

        Mock 模式：
            返回一段模拟回答，方便不加载大模型时测试链路。
        """

        answer = _call_general_chat_generate(llm=llm, state=state)

        return {
            "answer": answer,
            "flow_name": "general_chat",
        }

    def prompt_node(state: ChatState) -> ChatState:
        """
        Prompt 组装节点。

        这个节点把：
        1. 用户问题；
        2. RAG 检索出来的参考资料；
        3. 回答规则；

        组合成一个完整 Prompt。

        后面接真实 Qwen 时，模型看到的就是这里生成的 prompt。
        """

        prompt = build_rag_prompt(
            question=state["query"],
            contexts=state.get("contexts", []),
            history=state.get("history", []),
        )

        return {
            "prompt": prompt,
        }

    def generate_node(state: ChatState) -> ChatState:
        """
        答案生成节点。

        这个节点负责调用 LLM。

        当前 llm 是 MockLLMClient，所以它只是整理 chunks。
        后面换成 QwenLLMClient 后，这个节点就会变成真正的大模型生成。
        """

        answer = _call_llm_generate(llm=llm, state=state)

        return {
            "answer": answer,
            "flow_name": "generic_rag",
        }

    def finalize_node(state: ChatState) -> ChatState:
        """
        结果整理节点。

        这个节点负责把 RetrievedChunk 转成接口更容易返回的 citations。

        为什么不直接把 contexts 返回给前端？
        因为 contexts 是 Python 数据对象，里面可能包含后端内部字段。
        citations 是整理后的引用信息，更适合接口返回。
        """

        citations = [
            {
                "chunk_id": chunk.chunk_id,
                "source": chunk.source,
                "score": round(chunk.score, 4),
                "text": chunk.text[:300],
            }
            for chunk in state.get("contexts", [])
        ]

        return {
            "citations": citations,
            "retrieved_count": len(state.get("contexts", [])),
            "blocked": False,
            "guardrail_reason": None,
        }

    # 创建一个 StateGraph。
    # ChatState 告诉 LangGraph：工作流状态大概有哪些字段。
    workflow = StateGraph(ChatState)

    # 注册节点。
    # 左边是节点名称，右边是节点函数。
    workflow.add_node("guardrails", guardrails_node)
    workflow.add_node("intent", intent_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("rerank", rerank_node)
    workflow.add_node("major_change_flow", major_change_flow_node)
    workflow.add_node("grade_review_flow", grade_review_flow_node)
    workflow.add_node("transcript_flow", transcript_flow_node)
    workflow.add_node("transcript_private_analysis", transcript_private_analysis_node)
    workflow.add_node("exam_affairs_flow", exam_affairs_flow_node)
    workflow.add_node("student_status_change_flow", student_status_change_flow_node)
    workflow.add_node("certificate_service_flow", certificate_service_flow_node)
    workflow.add_node("course_registration_flow", course_registration_flow_node)
    workflow.add_node("general_chat_prompt", general_chat_prompt_node)
    workflow.add_node("general_chat_generate", general_chat_generate_node)
    workflow.add_node("prompt", prompt_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("finalize", finalize_node)

    # 设置入口节点。
    # START 是 LangGraph 内置的开始标记。
    workflow.add_edge(START, "guardrails")

    # 设置条件边。
    # guardrails 节点执行完以后，根据 route_after_guardrails 的返回值决定下一步。
    workflow.add_conditional_edges(
        "guardrails",
        route_after_guardrails,
        {
            "blocked": END,
            "passed": "intent",
        },
    )

    # 设置正常问答链路。
    workflow.add_conditional_edges(
        "intent",
        route_after_intent,
        {
            "general_chat": "general_chat_prompt",
            "transcript_private_analysis": "transcript_private_analysis",
            "academic": "retrieve",
        },
    )
    workflow.add_edge("retrieve", "rerank")
    workflow.add_conditional_edges(
        "rerank",
        route_after_retrieve,
        {
            "major_change": "major_change_flow",
            "grade_review": "grade_review_flow",
            "transcript": "transcript_flow",
            "exam_affairs": "exam_affairs_flow",
            "student_status_change": "student_status_change_flow",
            "certificate_service": "certificate_service_flow",
            "course_registration": "course_registration_flow",
            "generic": "prompt",
        },
    )
    workflow.add_edge("major_change_flow", "finalize")
    workflow.add_edge("grade_review_flow", "finalize")
    workflow.add_edge("transcript_flow", "finalize")
    workflow.add_edge("transcript_private_analysis", "finalize")
    workflow.add_edge("exam_affairs_flow", "finalize")
    workflow.add_edge("student_status_change_flow", "finalize")
    workflow.add_edge("certificate_service_flow", "finalize")
    workflow.add_edge("course_registration_flow", "finalize")
    workflow.add_edge("general_chat_prompt", "general_chat_generate")
    workflow.add_edge("general_chat_generate", "finalize")
    workflow.add_edge("prompt", "generate")
    workflow.add_edge("generate", "finalize")
    workflow.add_edge("finalize", END)

    # compile() 会把工作流编译成可执行对象。
    # 后面在 routes.py 里调用 chat_graph.invoke(initial_state) 即可运行。
    return workflow.compile()
