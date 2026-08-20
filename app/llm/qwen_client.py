import os
import threading
from collections.abc import Iterator


def env_bool(name: str, default: bool = False) -> bool:
    """读取布尔类型环境变量。

    环境变量本质上都是字符串，所以这里统一把常见写法转成 True/False。
    例如：
        QWEN_LOCAL_FILES_ONLY=1
        QWEN_LOCAL_FILES_ONLY=true
        QWEN_LOCAL_FILES_ONLY=yes

    都会被理解成 True。
    """

    value = os.getenv(name)

    if value is None:
        return default

    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


class QwenLLMClient:
    """本地 Qwen 大模型客户端。

    这个类负责把 LangGraph 组装好的 Prompt 发送给本地 Qwen 模型，
    再把模型生成的文本返回给 RAG 工作流。

    为什么要单独写成一个类？
        因为业务代码不应该直接到处 import transformers。
        路由层和 LangGraph 只需要知道：
            llm.generate(prompt=...) 可以返回答案。
        至于底层用 Mock、Qwen、本地模型还是云端 API，
        都应该封装在 LLMClient 里面。

    默认模型：
        Qwen/Qwen2.5-1.5B-Instruct

    你也可以通过环境变量覆盖：
        $env:QWEN_MODEL_NAME="你的本地模型路径或 HuggingFace 模型名"
        $env:QWEN_LOCAL_FILES_ONLY="1"

    QWEN_LOCAL_FILES_ONLY 的作用：
        第一次下载模型时不要打开它。
        等模型已经完整缓存到本机之后，再打开它。
        这样 transformers 只会读取本地缓存，不会每次启动都访问 HuggingFace 检查配置。

    注意：
        Qwen2.5-1.5B 比 4B 模型更轻量，
        更适合 8GB 显存的本地学习和项目演示环境。
    """

    def __init__(
        self,
        model_name: str | None = None,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
    ):
        # 模型名可以是 HuggingFace 模型名，也可以是本地模型目录。
        # 例如：
        #   Qwen/Qwen2.5-1.5B-Instruct
        #   F:\models\Qwen2.5-1.5B-Instruct
        self.model_name = model_name or os.getenv(
            "QWEN_MODEL_NAME",
            "Qwen/Qwen2.5-1.5B-Instruct",
        )

        # max_new_tokens 控制模型最多生成多少个新 token。
        # 值太小，答案可能被截断。
        # 值太大，本地推理会更慢。
        self.max_new_tokens = max_new_tokens or int(
            os.getenv("QWEN_MAX_NEW_TOKENS", "256")
        )

        # temperature 越低，回答越稳定。
        # 教务政策问答不适合太发散，所以默认 0.2。
        self.temperature = (
            temperature
            if temperature is not None
            else float(os.getenv("QWEN_TEMPERATURE", "0.2"))
        )

        # top_p 也是控制随机性的参数。
        # 和 temperature 一样，政策问答里不要设得太激进。
        self.top_p = (
            top_p
            if top_p is not None
            else float(os.getenv("QWEN_TOP_P", "0.9"))
        )

        # 是否只使用本地缓存文件。
        # 这个开关适合“模型已经下载完成”之后使用。
        # 如果第一次运行就设置为 True，而本地缓存里没有完整模型，
        # transformers 会直接报错，因为它被禁止联网补文件。
        self.local_files_only = env_bool("QWEN_LOCAL_FILES_ONLY", False)

        # tokenizer 和 model 先不加载。
        # 这样 FastAPI 启动会很快，只有第一次真正调用 Qwen 时才加载模型。
        self._tokenizer = None
        self._model = None
        # FastAPI 可能同时收到多个聊天请求。
        # 如果两个请求在模型尚未加载完成时一起进入 _load_model()，
        # transformers 可能会被触发多次下载/加载同一个模型。
        # 这里用线程锁保证同一个 QwenLLMClient 实例里，同一时间只会有一个请求执行模型加载。
        self._load_lock = threading.Lock()

    def _load_model(self):
        """懒加载 tokenizer 和 model。

        transformers 模型通常加载较慢，而且会占用显存或内存。
        所以我们把加载动作放在第一次 generate() 时执行。
        """

        if self._tokenizer is not None and self._model is not None:
            return self._tokenizer, self._model

        with self._load_lock:
            # 双重检查：
            # 第一个请求拿到锁后会真正加载模型；
            # 其他请求在锁外等待，等它们拿到锁时模型可能已经加载好了，
            # 这时直接复用内存/显存中的对象，不再重复下载或重复加载。
            if self._tokenizer is not None and self._model is not None:
                return self._tokenizer, self._model

            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            # Windows + 普通学习环境下，最稳的策略是自动判断是否有 CUDA。
            # 有显卡就用 device_map="auto"，没有显卡就放到 CPU。
            has_cuda = torch.cuda.is_available()
            dtype = torch.float16 if has_cuda else torch.float32

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                local_files_only=self.local_files_only,
            )

            model_kwargs = {
                "trust_remote_code": True,
                "dtype": dtype,
                "local_files_only": self.local_files_only,
            }

            if has_cuda:
                model_kwargs["device_map"] = os.getenv("QWEN_DEVICE_MAP", "auto")

            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs,
            )

            if not has_cuda:
                self._model.to("cpu")

            self._model.eval()
        return self._tokenizer, self._model

    def _build_chat_text(self, prompt: str) -> str:
        """把普通 Prompt 包装成 Qwen Chat Template。

        Qwen 这类对话模型通常不是直接吃一段纯文本，
        而是希望输入类似：
            system: 你是谁
            user: 用户问题
            assistant: ...

        tokenizer.apply_chat_template() 会把这些消息转成模型训练时熟悉的格式。
        """

        tokenizer, _ = self._load_model()

        messages = [
            {
                "role": "system",
                "content": (
                    "你是校园教务智能助手。"
                    "必须严格依据用户提供的参考资料回答，"
                    "没有依据时要说明当前资料未找到明确依据。"
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]

        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def generate(self, prompt: str) -> str:
        """一次性生成完整答案。

        LangGraph 的普通 invoke() 会调用这个方法。
        返回值是一个完整字符串。
        """

        tokenizer, model = self._load_model()
        chat_text = self._build_chat_text(prompt)

        inputs = tokenizer(
            [chat_text],
            return_tensors="pt",
        ).to(model.device)

        do_sample = self.temperature > 0

        import torch

        generation_kwargs = {
            **inputs,
            "max_new_tokens": self.max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": tokenizer.eos_token_id,
            "use_cache": True,
        }

        if do_sample:
            generation_kwargs["temperature"] = self.temperature
            generation_kwargs["top_p"] = self.top_p

        with torch.inference_mode():
            output_ids = model.generate(**generation_kwargs)

        # output_ids 里包含“输入 token + 新生成 token”。
        # 我们只截取新生成部分，避免把 Prompt 也返回给用户。
        generated_ids = output_ids[0][inputs.input_ids.shape[-1] :]
        answer = tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        ).strip()

        return answer

    def stream_generate(self, prompt: str) -> Iterator[str]:
        """按 token 流式生成答案。

        当前项目的 SSE 接口已经跑通。
        后面如果要把“模拟切片流式”升级为“模型 token 级流式”，
        就可以调用这个方法。

        注意：
            这个方法返回的是一个迭代器。
            每次 yield 都是模型新吐出来的一小段文本。
        """

        import torch
        from transformers import TextIteratorStreamer

        tokenizer, model = self._load_model()
        chat_text = self._build_chat_text(prompt)

        inputs = tokenizer(
            [chat_text],
            return_tensors="pt",
        ).to(model.device)

        streamer = TextIteratorStreamer(
            tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        do_sample = self.temperature > 0

        generation_kwargs = {
            **inputs,
            "streamer": streamer,
            "max_new_tokens": self.max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": tokenizer.eos_token_id,
            "use_cache": True,
        }

        if do_sample:
            generation_kwargs["temperature"] = self.temperature
            generation_kwargs["top_p"] = self.top_p

        def run_generate():
            with torch.inference_mode():
                model.generate(**generation_kwargs)

        thread = threading.Thread(
            target=run_generate,
            daemon=True,
        )
        thread.start()

        for token_text in streamer:
            yield token_text
