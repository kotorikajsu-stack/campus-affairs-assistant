from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from app.api.routes import router


# 创建 FastAPI 应用对象。
#
# 启动命令示例：
# uvicorn app.main:app --reload --port 8001
#
# 这里的 app.main 表示：
# app 文件夹里的 main.py 文件。
#
# 最后的 :app 表示：
# main.py 里的 app 变量。
#
# 所以这一行变量名必须叫 app，uvicorn 才能正确找到应用。
app = FastAPI(title="校园教务智能助手")


# 把 app/api/routes.py 里定义的接口注册到 FastAPI 应用中。
#
# routes.py 里的 router 设置了 prefix="/api/v1"，
# 所以里面的 /chat 实际访问地址就是 /api/v1/chat。
app.include_router(router)


@app.get("/")
def root():
    """首页接口。

    访问：
        http://127.0.0.1:8001/

    作用：
        确认后端服务已经启动。
    """

    return {
        "message": "校园教务智能助手 API 已启动",
        "ui": "http://127.0.0.1:8001/ui",
        "docs": "http://127.0.0.1:8001/docs",
    }


@app.get("/health")
def health():
    """健康检查接口。

    访问：
        http://127.0.0.1:8001/health

    作用：
        判断后端服务是否正常运行。
    """

    return {"status": "ok"}


@app.get("/ui", response_class=HTMLResponse)
def chat_ui():
    """前端聊天页面。

    访问：
        http://127.0.0.1:8001/ui

    为什么这里不用单独启动前端项目？
        当前阶段我们只是验证 FastAPI + SSE 的流式交互链路，
        一个静态 HTML 页面就够了。

    这个页面会在浏览器里调用：
        POST /api/v1/chat/stream

    也就是你刚刚已经用 curl 跑通的 SSE 接口。
    """

    html_path = Path(__file__).resolve().parent / "web" / "index.html"
    return html_path.read_text(encoding="utf-8")
