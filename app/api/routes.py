import json
import os
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from app.agent.intent import classify_intent
from app.agent.graph import build_chat_graph
from app.api.schemas import ChatRequest, ChatResponse, Citation
from app.core.guardrails import check_input_guardrails
from app.llm.client import MockLLMClient
from app.llm.prompts import build_general_chat_prompt, build_rag_prompt
from app.rag.reranker import rerank_contexts
from app.rag.retriever import JsonlRetriever
from app.services.ocr import TranscriptOcrService


# APIRouter 用来管理一组接口。
#
# prefix="/api/v1" 表示这个文件里的接口都会以 /api/v1 开头。
# 例如下面的 /chat 实际访问路径就是：
#     /api/v1/chat
router = APIRouter(prefix="/api/v1")


# 本地 JSONL 检索器默认读取这两个 chunk 文件。
# synthetic_docs_chunks.jsonl 是模拟教务资料。
# official_docs_chunks.jsonl 是你用 PDF / MinerU 等方式解析后的正式资料。
CHUNKS_PATHS = [
    "data/processed/generic_university/synthetic_docs_chunks.jsonl",
    "data/processed/generic_university/official_docs_chunks.jsonl",
]


# 这些意图已经有专用 LangGraph 流程节点。
# 命中这些意图时，答案由规则化流程节点生成，
# 不需要让大模型自由发挥。
FLOW_INTENTS = {
    "major_change",
    "grade_review",
    "transcript",
    "exam_affairs",
    "student_status_change",
    "certificate_service",
    "course_registration",
}


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


def build_contextual_query_for_request(request: ChatRequest) -> str:
    """
    给路由层的快速意图预览构建上下文问题。

    LangGraph 内部也有一份类似逻辑。
    这里主要服务于 Qwen SSE 分支，因为该分支会先判断是否可以直接 token stream。
    """

    if not request.history:
        return request.query

    if not any(word in request.query for word in FOLLOW_UP_WORDS):
        return request.query

    history_text = "\n".join(
        f"{item.role}: {item.content[:300]}"
        for item in request.history[-6:]
    )

    return f"{history_text}\n当前追问：{request.query}"


def build_retriever():
    """根据环境变量创建检索器。

    当前项目支持两种检索后端：

    1. jsonl：
       读取本地 chunks.jsonl，用关键词和规则做检索。
       优点是简单、稳定、方便调试。

    2. milvus：
       使用 embedding + Milvus 向量库做语义检索。
       这是更接近商业落地的 RAG 检索方式。

    切换方式：
        $env:RETRIEVER_BACKEND="jsonl"
        $env:RETRIEVER_BACKEND="milvus"
    """

    backend = os.getenv("RETRIEVER_BACKEND", "jsonl").lower()

    if backend == "milvus":
        # 只有真正启用 milvus 时才导入 pymilvus 相关代码。
        # 这样即使某台电脑没装 Milvus，也不会影响 jsonl 模式启动。
        from app.rag.milvus_retriever import MilvusRetriever

        return MilvusRetriever()

    return JsonlRetriever(chunks_paths=CHUNKS_PATHS)


def build_llm_client():
    """根据环境变量创建大模型客户端。

    当前支持：

    1. mock：
       默认值，不调用真实大模型。
       适合学习 RAG 工程链路、跑测试、做前端演示。

    2. qwen：
       调用本地 transformers 版 Qwen。
       需要你已经安装 torch / transformers，
       并且本机能下载或访问 Qwen 模型。

    切换方式：
        $env:LLM_BACKEND="mock"
        $env:LLM_BACKEND="qwen"
    """

    backend = os.getenv("LLM_BACKEND", "mock").lower()

    if backend == "qwen":
        from app.llm.qwen_client import QwenLLMClient

        return QwenLLMClient()

    return MockLLMClient()


# 这些对象在服务启动时创建一次，后续请求复用。
retriever = build_retriever()
llm = build_llm_client()
chat_graph = build_chat_graph(
    retriever=retriever,
    llm=llm,
)
ocr_service = TranscriptOcrService()


def sse_event(event: str, data: dict[str, Any]) -> str:
    """把 Python 字典包装成 SSE 消息格式。

    SSE 全称是 Server-Sent Events。
    它要求服务端返回的文本大致长这样：

        event: message
        data: {"content": "你好"}

    注意最后必须有两个换行：
        \n\n

    浏览器或 curl 收到两个换行后，
    才会认为这一条事件结束。
    """

    json_data = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {json_data}\n\n"


def split_text_for_stream(text: str, chunk_size: int = 10):
    """把完整答案切成多个小片段，用于模拟流式输出。

    MockLLMClient 一次性返回完整 answer。
    为了让前端仍然能看到“流式效果”，
    我们把完整 answer 按固定长度切片后逐段推送。

    当 LLM_BACKEND=qwen 且客户端支持 stream_generate() 时，
    SSE 接口会优先使用真实 token 流。
    """

    for start in range(0, len(text), chunk_size):
        yield text[start : start + chunk_size]


def stream_qwen_answer(prompt: str):
    """根据 QWEN_STREAM_MODE 输出 Qwen 答案。

    为什么要有这个函数？
        如果电脑没有 CUDA 显卡，Qwen3-4B 在 CPU 上通常只能很慢地生成 token。
        真实 token 流会变成“一秒蹦一个字”，体验反而很差。

    三种模式：
        token:
            真实 token 流。适合有显卡或模型速度足够快的环境。

        smooth:
            半缓冲流式。边生成边返回，但会把零碎 token 合并成更稳定的小片段。
            这是本项目在本地显卡演示时的推荐模式。

        buffered:
            默认模式。先让 Qwen 在后端完整生成答案，再把完整答案切成小段快速推给前端。
            总耗时不一定更短，但视觉上不会卡成一个字一个字跳。
    """

    stream_mode = os.getenv("QWEN_STREAM_MODE", "buffered").lower()

    if stream_mode in {"token", "smooth"}:
        generated_parts = []
        token_buffer = ""
        last_flush_time = time.monotonic()
        default_flush_chars = "8" if stream_mode == "token" else "18"
        default_flush_interval = "0.06" if stream_mode == "token" else "0.16"
        flush_chars = int(os.getenv("QWEN_STREAM_FLUSH_CHARS", default_flush_chars))
        flush_interval = float(os.getenv("QWEN_STREAM_FLUSH_INTERVAL", default_flush_interval))
        punctuation_marks = "。！？；，、.!?;\n"

        for piece in llm.stream_generate(prompt):
            generated_parts.append(piece)
            token_buffer += piece

            now = time.monotonic()
            should_flush = (
                len(token_buffer) >= flush_chars
                or now - last_flush_time >= flush_interval
                or any(token_buffer.endswith(mark) for mark in punctuation_marks)
            )

            if should_flush:
                yield token_buffer
                token_buffer = ""
                last_flush_time = now

        if token_buffer:
            yield token_buffer

        return "".join(generated_parts)

    answer = llm.generate(prompt=prompt)

    for piece in split_text_for_stream(answer, chunk_size=16):
        yield piece
        time.sleep(0.01)

    return answer


def build_initial_state(request: ChatRequest) -> dict[str, Any]:
    """把 HTTP 请求转换成 LangGraph 初始状态。"""

    history = [item.model_dump() for item in request.history]

    return {
        "query": request.query,
        "top_k": request.top_k,
        "tenant_id": request.tenant_id,
        "role": request.role,
        "allow_private_analysis": request.allow_private_analysis,
        "history": history,
        "memory_turns": len(history),
    }


def response_from_state(request: ChatRequest, final_state: dict[str, Any]) -> ChatResponse:
    """把 LangGraph 最终状态转换成 FastAPI 响应模型。"""

    return ChatResponse(
        query=request.query,
        answer=final_state.get("answer", ""),
        citations=[
            Citation(**citation)
            for citation in final_state.get("citations", [])
        ],
        retrieved_count=final_state.get("retrieved_count", 0),
        candidate_count=final_state.get("candidate_count"),
        reranked_count=final_state.get("reranked_count"),
        intent=final_state.get("intent"),
        intent_label=final_state.get("intent_label"),
        intent_route=final_state.get("intent_route"),
        intent_reason=final_state.get("intent_reason"),
        flow_name=final_state.get("flow_name"),
        blocked=final_state.get("blocked", False),
        guardrail_reason=final_state.get("guardrail_reason"),
        private_analysis_allowed=final_state.get("private_analysis_allowed", False),
        privacy_sanitized=final_state.get("privacy_sanitized", False),
        memory_turns=final_state.get("memory_turns", len(request.history)),
    )


def build_citations(contexts) -> list[dict[str, Any]]:
    """把 RetrievedChunk 列表转换成接口可返回的引用来源。"""

    return [
        {
            "chunk_id": chunk.chunk_id,
            "source": chunk.source,
            "score": round(chunk.score, 4),
            "text": chunk.text[:300],
        }
        for chunk in contexts
    ]


def build_streaming_generic_state(request: ChatRequest) -> dict[str, Any]:
    """为 Qwen token 流式生成准备 RAG 状态。

    为什么需要这个函数？
        普通 chat_graph.invoke() 是同步执行的。
        如果某个 generic_rag 问题要调用 Qwen，
        invoke() 会等 Qwen 完整生成结束后才返回 final_state。

        这样前端虽然走 SSE，但看到的只是“完整答案切片”，
        不是真正的模型 token 流。

    这里手动执行通用 RAG 的前半段：
        guardrails -> intent -> retrieve -> rerank -> prompt

    然后 SSE 接口可以直接调用：
        llm.stream_generate(prompt)

    从而做到真正边生成边返回。
    """

    guardrail = check_input_guardrails(
        request.query,
        allow_private_analysis=request.allow_private_analysis,
    )

    if not guardrail.allowed:
        return {
            "blocked": True,
            "guardrail_reason": guardrail.reason,
            "answer": guardrail.message or "该问题暂时无法回答。",
            "contexts": [],
            "citations": [],
            "retrieved_count": 0,
            "candidate_count": 0,
            "reranked_count": 0,
            "flow_name": None,
            "private_analysis_allowed": False,
            "privacy_sanitized": False,
            "memory_turns": len(request.history),
        }

    intent = classify_intent(build_contextual_query_for_request(request))
    roles = [request.role]
    candidate_limit = max(request.top_k * 4, 10)

    candidates = retriever.search(
        query=intent.retrieval_query,
        tenant_id=request.tenant_id,
        roles=roles,
        limit=candidate_limit,
    )

    contexts = rerank_contexts(
        intent=intent.intent,
        contexts=candidates,
        limit=request.top_k,
    )

    prompt = build_rag_prompt(
        question=request.query,
        contexts=contexts,
        history=request.history,
    )

    return {
        "blocked": False,
        "guardrail_reason": None,
        "intent": intent.intent,
        "intent_label": intent.label,
        "intent_route": intent.route,
        "intent_reason": intent.reason,
        "retrieval_query": intent.retrieval_query,
        "contexts": contexts,
        "citations": build_citations(contexts),
        "retrieved_count": len(contexts),
        "candidate_count": len(candidates),
        "reranked_count": len(contexts),
        "prompt": prompt,
        "flow_name": "generic_qwen_stream",
        "private_analysis_allowed": False,
        "privacy_sanitized": False,
        "memory_turns": len(request.history),
    }


def build_stream_metadata(final_state: dict[str, Any]) -> dict[str, Any]:
    """整理 SSE metadata 事件中需要返回的字段。"""

    return {
        "stage": "graph_finished",
        "intent": final_state.get("intent"),
        "intent_label": final_state.get("intent_label"),
        "intent_route": final_state.get("intent_route"),
        "flow_name": final_state.get("flow_name"),
        "blocked": final_state.get("blocked", False),
        "guardrail_reason": final_state.get("guardrail_reason"),
        "retrieved_count": final_state.get("retrieved_count", 0),
        "candidate_count": final_state.get("candidate_count"),
        "reranked_count": final_state.get("reranked_count"),
        "llm_backend": os.getenv("LLM_BACKEND", "mock").lower(),
        "private_analysis_allowed": final_state.get("private_analysis_allowed", False),
        "privacy_sanitized": final_state.get("privacy_sanitized", False),
        "memory_turns": final_state.get("memory_turns", 0),
    }


@router.get("/retriever/status")
def retriever_status():
    """查看当前检索后端和模型后端状态。"""

    backend = os.getenv("RETRIEVER_BACKEND", "jsonl").lower()
    llm_backend = os.getenv("LLM_BACKEND", "mock").lower()
    retriever_class = type(retriever).__name__

    status = {
        "backend": backend,
        "retriever_class": retriever_class,
        "llm_backend": llm_backend,
        "llm_class": type(llm).__name__,
    }

    if hasattr(retriever, "chunks"):
        status["loaded_chunks"] = len(retriever.chunks)

    if hasattr(retriever, "chunks_paths"):
        status["chunks_paths"] = [str(path) for path in retriever.chunks_paths]

    if hasattr(llm, "model_name"):
        status["llm_model_name"] = getattr(llm, "model_name")

    if retriever_class == "MilvusRetriever":
        uri = getattr(retriever, "uri", None)
        collection_name = getattr(retriever, "collection_name", None)
        embedding_model = getattr(retriever, "embedding_model", None)

        status["collection_name"] = collection_name
        status["uri"] = uri
        status["embedding_model"] = getattr(embedding_model, "model_name", None)

        if uri and "://" not in uri:
            status["local_db_path"] = str(Path(uri).resolve())
            status["local_db_exists"] = Path(uri).exists()

        try:
            client = getattr(retriever, "client", None)
            if client is not None and collection_name:
                status["collection_exists"] = client.has_collection(collection_name)
        except Exception as error:
            status["collection_status_error"] = f"{type(error).__name__}: {error}"

    return status


@router.get("/search")
def search(query: str, top_k: int = 5):
    """RAG 检索调试接口。

    访问示例：
        /api/v1/search?query=转专业需要什么材料

    这个接口只做检索，不生成最终答案。
    主要用于检查当前知识库能不能召回正确 chunk。
    """

    results = retriever.search(
        query=query,
        tenant_id="generic-university",
        roles=["student"],
        limit=top_k,
    )

    return {
        "query": query,
        "top_k": top_k,
        "loaded_chunks": len(getattr(retriever, "chunks", [])),
        "results": [
            {
                "chunk_id": item.chunk_id,
                "source": item.source,
                "score": round(item.score, 4),
                "visibility": item.visibility,
                "doc_type": item.metadata.get("doc_type"),
                "text": item.text[:500],
            }
            for item in results
        ],
    }


@router.get("/chat")
def chat(query: str, top_k: int = 5):
    """浏览器调试版问答接口。

    这个 GET 接口方便直接在地址栏测试。
    正式前端建议使用 POST /api/v1/chat。
    """

    guardrail = check_input_guardrails(query)
    if not guardrail.allowed:
        return {
            "query": query,
            "answer": guardrail.message,
            "citations": [],
            "debug": {
                "retrieved_count": 0,
                "blocked": True,
                "guardrail_reason": guardrail.reason,
            },
        }

    contexts = retriever.search(
        query=query,
        tenant_id="generic-university",
        roles=["student"],
        limit=top_k,
    )

    prompt = build_rag_prompt(question=query, contexts=contexts)
    if hasattr(llm, "model_name"):
        answer = llm.generate(prompt=prompt)
    else:
        answer = llm.generate(question=query, contexts=contexts)

    citations = [
        {
            "chunk_id": chunk.chunk_id,
            "source": chunk.source,
            "score": round(chunk.score, 4),
            "text": chunk.text[:300],
        }
        for chunk in contexts
    ]

    return {
        "query": query,
        "answer": answer,
        "citations": citations,
        "debug": {
            "retrieved_count": len(contexts),
            "prompt": prompt,
        },
    }


@router.post("/chat", response_model=ChatResponse)
def chat_post(request: ChatRequest):
    """正式版 RAG 问答接口。

    这个接口把请求交给 LangGraph 工作流：
        guardrails -> intent -> retrieve -> rerank -> flow/generate -> finalize
    """

    try:
        final_state = chat_graph.invoke(build_initial_state(request))
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"{type(error).__name__}: {error}",
        ) from error

    return response_from_state(request, final_state)


@router.post("/chat/stream")
def chat_stream(request: ChatRequest):
    """SSE 流式问答接口。

    当前有两种流式模式：

    1. Mock 或流程节点答案：
       LangGraph 先生成完整 final_state，
       再把 answer 切成小块推送给前端。

    2. Qwen generic_rag：
       如果后端使用 Qwen，并且当前问题走通用 RAG 生成节点，
       可以使用 llm.stream_generate(prompt) 做真实 token 级流式输出。

    注意：
        转专业、成绩单、成绩复核等专用流程节点本身是规则化答案，
        它们不依赖大模型自由生成，所以仍然使用切片流式输出。
    """

    def event_generator():
        try:
            yield sse_event(
                "metadata",
                {
                    "stage": "started",
                    "query": request.query,
                },
            )

            llm_backend = os.getenv("LLM_BACKEND", "mock").lower()
            can_token_stream = hasattr(llm, "stream_generate")
            intent_preview = classify_intent(build_contextual_query_for_request(request))

            guardrail = check_input_guardrails(
                request.query,
                allow_private_analysis=request.allow_private_analysis,
            )
            if not guardrail.allowed:
                final_state = {
                    "blocked": True,
                    "guardrail_reason": guardrail.reason,
                    "answer": guardrail.message or "该问题暂时无法回答。",
                    "citations": [],
                    "retrieved_count": 0,
                    "candidate_count": 0,
                    "reranked_count": 0,
                    "private_analysis_allowed": False,
                    "privacy_sanitized": False,
                    "memory_turns": len(request.history),
                }

                yield sse_event("metadata", build_stream_metadata(final_state))
                for piece in split_text_for_stream(final_state["answer"]):
                    yield sse_event("message", {"content": piece})
                    time.sleep(0.015)
                yield sse_event("citations", {"citations": []})
                yield sse_event("done", {"finish_reason": "stop"})
                return

            # 非教务普通聊天：
            #     直接走 Qwen 通用聊天，不查 Milvus，不拼接 RAG 引用。
            if (
                llm_backend == "qwen"
                and can_token_stream
                and intent_preview.intent == "general_chat"
            ):
                final_state = {
                    "intent": intent_preview.intent,
                    "intent_label": intent_preview.label,
                    "intent_route": intent_preview.route,
                    "intent_reason": intent_preview.reason,
                    "flow_name": "general_chat",
                    "blocked": False,
                    "guardrail_reason": None,
                    "retrieved_count": 0,
                    "candidate_count": 0,
                    "reranked_count": 0,
                    "citations": [],
                    "prompt": build_general_chat_prompt(
                        request.query,
                        history=request.history,
                    ),
                    "memory_turns": len(request.history),
                }

                yield sse_event(
                    "metadata",
                    build_stream_metadata(final_state),
                )

                generated_parts = []
                for piece in stream_qwen_answer(final_state["prompt"]):
                    generated_parts.append(piece)
                    yield sse_event("message", {"content": piece})

                final_state["answer"] = "".join(generated_parts)

                yield sse_event(
                    "citations",
                    {
                        "citations": [],
                    },
                )

            # 教务泛问：
            #     走 RAG 检索 + Qwen token 级流式。
            # 高频教务流程问题已经有固定流程节点，继续走 LangGraph 规则化答案。
            elif (
                llm_backend == "qwen"
                and can_token_stream
                and intent_preview.intent not in FLOW_INTENTS
            ):
                final_state = build_streaming_generic_state(request)

                yield sse_event(
                    "metadata",
                    build_stream_metadata(final_state),
                )

                if final_state.get("blocked"):
                    for piece in split_text_for_stream(final_state.get("answer", "")):
                        yield sse_event("message", {"content": piece})
                        time.sleep(0.015)
                else:
                    generated_parts = []
                    for piece in stream_qwen_answer(final_state["prompt"]):
                        generated_parts.append(piece)
                        yield sse_event("message", {"content": piece})

                    final_state["answer"] = "".join(generated_parts)

                yield sse_event(
                    "citations",
                    {
                        "citations": final_state.get("citations", []),
                    },
                )
            else:
                final_state = chat_graph.invoke(build_initial_state(request))

                yield sse_event(
                    "metadata",
                    build_stream_metadata(final_state),
                )

                answer = final_state.get("answer", "")

                for piece in split_text_for_stream(answer):
                    yield sse_event(
                        "message",
                        {
                            "content": piece,
                        },
                    )
                    time.sleep(0.015)

                yield sse_event(
                    "citations",
                    {
                        "citations": final_state.get("citations", []),
                    },
                )

            yield sse_event(
                "done",
                {
                    "finish_reason": "stop",
                },
            )

        except Exception as error:
            yield sse_event(
                "error",
                {
                    "error_type": type(error).__name__,
                    "message": str(error),
                },
            )

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/ocr/transcript")
async def parse_transcript(file: UploadFile = File(...)):
    """成绩单 OCR 上传接口。

    请求方式：
        POST /api/v1/ocr/transcript

    表单字段：
        file: 上传的图片、txt 或 PDF 文件

    返回：
        文件名、解析引擎、识别文本、提示信息。
    """

    file_bytes = await file.read()

    if not file_bytes:
        raise HTTPException(status_code=400, detail="上传文件为空。")

    # 给学习项目加一个简单大小限制，避免误传超大文件把服务拖慢。
    max_size = 10 * 1024 * 1024
    if len(file_bytes) > max_size:
        raise HTTPException(status_code=413, detail="文件过大，请上传 10MB 以内的文件。")

    result = ocr_service.parse_bytes(
        file_bytes=file_bytes,
        filename=file.filename or "uploaded-file",
        content_type=file.content_type,
    )

    return {
        "filename": result.filename,
        "content_type": result.content_type,
        "engine": result.engine,
        "text": result.text,
        "text_length": len(result.text),
        "warning": result.warning,
    }
