# 校园教务智能助手

这是一个面向高校教务咨询场景的 RAG + LangGraph 智能助手项目。项目围绕“教务政策问答、办事流程引导、材料核验、权限隔离、隐私保护、流式对话和本地知识库更新”搭建。

本仓库示例数据统一使用“高校通用模拟数据”，不绑定具体学校；真实原始资料、模型权重、Milvus 本地数据库、缓存文件不会上传到仓库。

## 目录

- [0. 快速体验路径](#0-快速体验路径)
- [1. 项目能力模块](#1-项目能力模块)
- [2. 工程目录模块](#2-工程目录模块)
- [3. 环境配置模块](#3-环境配置模块)
- [4. 启动运行模块](#4-启动运行模块)
- [5. 网页使用模块](#5-网页使用模块)
- [6. API 测试模块](#6-api-测试模块)
- [7. 新增数据模块](#7-新增数据模块)
- [8. MinerU 原始文件解析模块](#8-mineru-原始文件解析模块)
- [9. Milvus 向量库模块](#9-milvus-向量库模块)
- [10. Qwen 大模型模块](#10-qwen-大模型模块)
- [11. OCR 模块](#11-ocr-模块)
- [12. LangGraph 工作流模块](#12-langgraph-工作流模块)
- [13. 测试与验证模块](#13-测试与验证模块)
- [14. 常见问题模块](#14-常见问题模块)
- [15. 后续扩展模块](#15-后续扩展模块)

## 0. 快速体验路径

如果你是第一次打开这个项目，推荐先按下面这条最短路径跑通：

```powershell
cd 校园教务助手
conda create -n ICSA python=3.12 -y
conda activate ICSA
pip install -r requirements.txt

$env:LLM_BACKEND="mock"
$env:RETRIEVER_BACKEND="local"

uvicorn app.main:app --reload --port 8001
```

然后打开：

```text
http://127.0.0.1:8001/ui
```

先用 mock 模型和本地检索跑通网页、接口、工作流和 RAG 链路。确认项目能正常运行后，再切换到 Qwen + Milvus 模式。

## 1. 项目能力模块

### 1.1 业务痛点

- 高校教务规则分散在 PDF、通知、表格、网页和内部制度中，学生很难快速找到准确答案。
- 人工咨询重复问题多，响应慢，教务老师和客服人员负担重。
- 通用大模型容易对政策条款进行臆测，存在幻觉和误导风险。
- 转专业、缓考、成绩复核、学籍异动、毕业审核等长流程事项容易遗漏材料和审批节点。
- 教务数据涉及学生个人信息，需要做业务边界、隐私拦截和权限过滤。

### 1.2 核心功能

- RAG 问答：从教务知识库中检索相关条款，再生成回答。
- Milvus 检索：使用本地 Milvus Lite 保存向量数据。
- Rerank 重排：对召回片段重新排序，提高引用相关性。
- LangGraph 工作流：按意图进入不同教务流程节点。
- Guardrails 护栏：输入拦截、业务范围控制、隐私信息保护。
- Qwen 本地大模型：通过 HuggingFace Transformers 加载 Qwen。
- SSE 流式输出：网页端支持流式聊天体验。
- OCR 上传：支持上传图片/PDF 做基础识别演示。
- 新增文档一键入库：支持新增 Markdown/TXT，也支持 PDF/图片/Office 经 MinerU 解析后重建本地 Milvus。

## 2. 工程目录模块

```text
app/
  agent/          LangGraph 风格工作流、意图识别和节点编排
  api/            FastAPI 路由、SSE 流式接口、网页入口
  core/           配置、日志、运行上下文
  guardrails/     输入、业务边界、隐私与输出护栏
  llm/            Mock / Qwen 大模型客户端
  rag/            文档切分、检索、rerank、Milvus 适配
  services/       OCR、对话服务等业务封装
  web/            简洁聊天式前端页面

configs/
  flows.yaml      教务事项流程配置
  guardrails.yaml 护栏策略配置
  rag.yaml        RAG 检索参数配置

data/
  raw/            原始资料投放目录和通用模拟 Markdown 数据
  processed/      切分后的 chunks 文件、MinerU 输出文本
  datasets/       预留 SFT / DPO 数据集目录

docs/
  ARCHITECTURE.md              架构说明

scripts/
  ingest_documents.py           文档切分脚本
  refresh_knowledge_base.py     Markdown/TXT 一键重建 Milvus
  rebuild_from_raw_documents.py PDF/图片/Office 解析并重建 Milvus
  parse_with_mineru.py          单独调用 MinerU 解析文件
  build_milvus_index.py         根据 chunks 创建 Milvus 索引
  prepare_sft_dataset.py        微调数据集生成模板
  train_lora.py                 LoRA 训练入口模板

tests/
  单元测试
```

## 3. 环境配置模块

### 3.1 推荐 Python 版本

推荐使用 Python 3.12。项目也在 Python 3.13 环境下通过过单元测试，但大模型、OCR、MinerU 等第三方依赖通常对 Python 3.12 更稳。

### 3.2 Conda 环境

```powershell
conda create -n ICSA python=3.12 -y
conda activate ICSA
pip install -r requirements.txt
```

### 3.3 venv 环境

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 3.4 可选依赖

MinerU 依赖较重，不默认写入 `requirements.txt` 强制安装。需要解析 PDF、图片、Word、PPT、Excel 原始文件时再安装：

```powershell
pip install uv
uv pip install -U "mineru[all]"
```

如果要使用图片 OCR，Windows 需要额外安装系统级 Tesseract-OCR，并确保 `tesseract` 命令能在终端中识别：

```powershell
tesseract --version
```

## 4. 启动运行模块

### 4.1 Qwen + Milvus 完整启动

第一次运行 Qwen 模型时，允许联网下载模型：

```powershell
cd 校园教务助手
conda activate ICSA

$env:LLM_BACKEND="qwen"
$env:RETRIEVER_BACKEND="milvus"
$env:QWEN_MODEL_NAME="Qwen/Qwen2.5-1.5B-Instruct"
$env:QWEN_MAX_NEW_TOKENS="256"
$env:QWEN_STREAM_MODE="smooth"
$env:QWEN_STREAM_FLUSH_CHARS="18"
$env:QWEN_STREAM_FLUSH_INTERVAL="0.16"
$env:QWEN_LOCAL_FILES_ONLY="0"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING="1"

uvicorn app.main:app --port 8001
```

模型下载完成后，后续可以改成本地缓存优先：

```powershell
$env:QWEN_LOCAL_FILES_ONLY="1"
```

### 4.2 开发调试快速启动

如果只是调前端、接口、RAG 或 LangGraph 流程，不想加载 Qwen，可以使用 mock 模型：

```powershell
cd 校园教务助手
conda activate ICSA

$env:LLM_BACKEND="mock"
$env:RETRIEVER_BACKEND="milvus"

uvicorn app.main:app --reload --port 8001
```

### 4.3 访问地址

```text
网页界面：http://127.0.0.1:8001/ui
接口文档：http://127.0.0.1:8001/docs
健康检查：http://127.0.0.1:8001/health
检索器状态：http://127.0.0.1:8001/api/v1/retriever/status
```

## 5. 网页使用模块

启动服务后打开：

```text
http://127.0.0.1:8001/ui
```

网页支持：

- 类主流 AI 助手的聊天界面。
- 用户消息和 AI 消息气泡展示。
- SSE 流式输出。
- 回车发送消息。
- `Shift + Enter` 换行。
- 输入框旁边的加号功能入口。
- OCR 文件上传入口。
- 回答下方展示引用资料，不在回答正文里重复写引用来源。
- 显示意图、流程、候选片段数、rerank 结果等调试信息。

可以测试的问题：

```text
转专业需要什么材料
成绩单怎么打印
缓考怎么申请
成绩怎么复核
毕业审核需要注意什么
```

## 6. API 测试模块

### 6.1 普通问答接口

```powershell
$body = @{
  query = "转专业需要什么材料"
  top_k = 3
  tenant_id = "generic-university"
  role = "student"
} | ConvertTo-Json -Compress

Invoke-RestMethod `
  -Uri "http://127.0.0.1:8001/api/v1/chat" `
  -Method Post `
  -ContentType "application/json; charset=utf-8" `
  -Body $body
```

### 6.2 SSE 流式问答接口

PowerShell 中建议先写入临时 JSON 文件，再用 `curl.exe --data-binary`，避免中文和引号转义问题：

```powershell
$bodyFile = "$env:TEMP\campus_chat_body.json"
@{
  query = "成绩单怎么打印"
  top_k = 3
  tenant_id = "generic-university"
  role = "student"
} | ConvertTo-Json -Compress | Set-Content -Path $bodyFile -Encoding UTF8

curl.exe -N -X POST "http://127.0.0.1:8001/api/v1/chat/stream" `
  -H "accept: text/event-stream" `
  -H "Content-Type: application/json; charset=utf-8" `
  --data-binary "@$bodyFile"
```

### 6.3 检索器状态接口

```text
http://127.0.0.1:8001/api/v1/retriever/status
```

正常使用 Milvus 时，会看到类似：

```json
{
  "backend": "milvus",
  "retriever_class": "MilvusRetriever",
  "collection_name": "campus_edu_chunks",
  "collection_exists": true
}
```

## 7. 新增数据模块

新增数据分两类：已经解析好的文本数据，以及 PDF/图片/Office 原始资料。

### 7.1 已经解析好的 Markdown/TXT

如果文件已经是：

```text
.md
.txt
```

放入：

```text
data/processed/generic_university/
```

然后停止正在运行的网站服务，执行：

```powershell
python scripts\refresh_knowledge_base.py
```

该脚本会自动：

```text
扫描 Markdown/TXT
-> 文档切分 chunk
-> 生成 data/processed/generic_university/all_docs_chunks.jsonl
-> embedding 向量化
-> 重建 Milvus collection
```

### 7.2 PDF/图片/Office 原始资料

如果文件是：

```text
.pdf
.png
.jpg
.jpeg
.bmp
.webp
.docx
.pptx
.xlsx
```

放入以下任意目录：

```text
data/raw/generic_university/policies/
data/raw/generic_university/guides/
data/raw/generic_university/forms/
data/raw/generic_university/notices/
data/raw/generic_university/reports/
data/raw/generic_university/inbox/
```

目录含义：

| 目录 | 用途 |
|---|---|
| `policies` | 政策制度、管理办法、学籍规定、考试规定 |
| `guides` | 办事指南、流程说明、窗口办理说明 |
| `forms` | 申请表、登记表、证明办理表格 |
| `notices` | 选课通知、考试通知、校历、调停课通知 |
| `reports` | 教学质量报告、培养方案说明、教学管理材料 |
| `inbox` | 临时投放目录，不确定类型的文件可以先放这里 |

然后执行：

```powershell
python scripts\rebuild_from_raw_documents.py
```

该脚本会自动：

```text
扫描 raw 目录中的 PDF/图片/Office 文件
-> 调用 MinerU 解析为 Markdown
-> 输出到 data/processed/generic_university/mineru/
-> 扫描 Markdown/TXT
-> 文档切分 chunk
-> embedding 向量化
-> 重建 data/milvus/campus_edu.db
```

### 7.3 只检查不写入

如果只想检查文档能不能被扫描，不想重建 Milvus：

```powershell
python scripts\refresh_knowledge_base.py --dry-run
```

如果想检查原始文件处理流程：

```powershell
python scripts\rebuild_from_raw_documents.py --dry-run
```


## 8. MinerU 原始文件解析模块

MinerU 在本项目中属于“文档解析工具”，用于把 PDF、扫描件、Office 文件等转成可进入 RAG 的文本。

### 8.1 安装 MinerU

```powershell
pip install uv
uv pip install -U "mineru[all]"
```

### 8.2 单独解析一个文件

```powershell
python scripts\parse_with_mineru.py `
  --input data\raw\generic_university\inbox\example.pdf `
  --output data\processed\generic_university\mineru `
  --backend pipeline
```

### 8.3 一键解析并入库

日常更推荐直接用：

```powershell
python scripts\rebuild_from_raw_documents.py
```

这个命令会先调用 MinerU，再自动刷新本地 Milvus。

## 9. Milvus 向量库模块

本项目默认使用 Milvus Lite，本地数据库文件位置：

```text
data/milvus/campus_edu.db
```

默认 collection：

```text
campus_edu_chunks
```

### 9.1 从 chunks 重建 Milvus

如果已经有：

```text
data/processed/generic_university/all_docs_chunks.jsonl
```

可以执行：

```powershell
python scripts\build_milvus_index.py --recreate
```

### 9.2 新增文档后一键重建

日常使用更推荐：

```powershell
python scripts\refresh_knowledge_base.py
```

或针对原始 PDF/图片/Office：

```powershell
python scripts\rebuild_from_raw_documents.py
```

### 9.3 为什么重建前建议停止服务

因为 Milvus Lite 是本地文件。如果 FastAPI 服务正在使用：

```text
data/milvus/campus_edu.db
```

同时又重建向量库，可能出现文件占用或 collection 状态冲突。

### 9.4 缺少 Milvus Lite 时怎么办

如果重建知识库时报错：

```text
ModuleNotFoundError: No module named 'milvus_lite'
```

说明当前环境没有安装本地 Milvus Lite 依赖。可以执行：

```powershell
python -m pip install milvus-lite
```

然后重新运行：

```powershell
python scripts\refresh_knowledge_base.py
```

## 10. Qwen 大模型模块

### 10.1 当前推荐模型

本地演示推荐：

```text
Qwen/Qwen2.5-1.5B-Instruct
```

这个模型比 4B 更轻，更适合普通笔记本显卡。

### 10.2 重要环境变量

| 配置项 | 作用 |
|---|---|
| `LLM_BACKEND="qwen"` | 使用 Qwen 本地模型 |
| `RETRIEVER_BACKEND="milvus"` | 使用 Milvus 检索 |
| `QWEN_MODEL_NAME` | 指定 HuggingFace 模型名称 |
| `QWEN_MAX_NEW_TOKENS` | 限制回答长度 |
| `QWEN_STREAM_MODE` | 控制流式输出模式 |
| `QWEN_LOCAL_FILES_ONLY` | 是否只读取本地缓存模型 |

### 10.3 流式模式

推荐：

```powershell
$env:QWEN_STREAM_MODE="smooth"
```

如果想首字更快，可以用：

```powershell
$env:QWEN_STREAM_MODE="token"
$env:QWEN_STREAM_FLUSH_CHARS="8"
$env:QWEN_STREAM_FLUSH_INTERVAL="0.06"
```

如果想显示最稳定，可以用：

```powershell
$env:QWEN_STREAM_MODE="buffered"
```

### 10.4 关于模型下载

第一次运行会下载模型到 HuggingFace 缓存目录。后续看到：

```text
Loading checkpoint shards: 100%
```

这不是重新下载，而是在把本地缓存里的模型加载到内存或显存。

## 11. OCR 模块

网页中的 OCR 功能用于演示“上传材料并提取文字”的能力，例如成绩单、申请表、证明材料截图等。

当前 OCR 更适合做功能演示，复杂表格、盖章成绩单、低清晰度截图识别效果可能不稳定。真实落地时建议替换为更强的 OCR 服务，例如 PaddleOCR、云厂商 OCR 或专门的成绩单结构化识别模型。

OCR 入口：

```text
网页 /ui 输入框旁边的加号功能区域
```

后端接口：

```text
POST /api/v1/ocr/transcript
```

如果未安装 Tesseract-OCR，接口会返回提示，不会影响主聊天功能。

## 12. LangGraph 工作流模块

项目中的工作流链路可以理解为：

```text
用户输入
-> Guardrails 输入拦截
-> 隐私信息检测
-> 意图识别
-> LangGraph 节点路由
-> RAG 检索
-> rerank 重排
-> Prompt 组装
-> Qwen / Mock LLM 生成
-> 输出合规检查
-> 返回答案和引用片段
```

当前支持的典型意图包括：

| 意图 | 示例问题 |
|---|---|
| 转专业 | 转专业需要什么材料 |
| 成绩复核 | 成绩怎么复核 |
| 成绩单办理 | 成绩单怎么打印 |
| 缓考 | 缓考怎么申请 |
| 学籍异动 | 休学复学怎么办 |
| 毕业审核 | 毕业审核需要注意什么 |
| 通用教务咨询 | 学校教务咨询有哪些常见事项 |

工作流配置文件：

```text
configs/flows.yaml
```

代码入口：

```text
app/agent/
```

## 13. 测试与验证模块

### 13.1 运行单元测试

```powershell
python -m pytest
```

### 13.2 编译检查

```powershell
python -m compileall app scripts tests
```

### 13.3 手动验证流程

1. 启动服务。
2. 打开 `/api/v1/retriever/status` 确认 Milvus 正常。
3. 打开 `/ui`。
4. 提问：

```text
转专业需要什么材料
成绩单怎么打印
缓考怎么申请
```

5. 检查回答下方是否出现引用片段。


## 14. 常见问题模块

### 14.1 为什么放了 PDF 后问答没有变化

PDF 不能直接被 RAG 使用，需要先解析成 Markdown/TXT。推荐执行：

```powershell
python scripts\rebuild_from_raw_documents.py
```

如果没有安装 MinerU，先执行：

```powershell
pip install uv
uv pip install -U "mineru[all]"
```

### 14.2 为什么启动很慢

如果使用 Qwen，本地服务第一次回答时需要加载模型。看到：

```text
Loading checkpoint shards
```

说明正在加载本地模型权重，不是重新下载。

开发时可以切换 mock：

```powershell
$env:LLM_BACKEND="mock"
```

### 14.3 为什么中文 curl 请求报 JSON 错误

PowerShell 里中文 JSON 容易被转义影响。推荐先写临时 JSON 文件，再使用：

```powershell
curl.exe --data-binary "@$bodyFile"
```

### 14.4 为什么 Milvus 报 collection released

通常是 collection 没有 load，或者服务运行时重建了本地 Milvus。建议：

1. 停止 FastAPI 服务。
2. 重新执行知识库刷新脚本。
3. 再启动服务。

### 14.5 为什么 OCR 识别成绩单不准

成绩单通常是密集表格，且可能有盖章、水印、模糊压缩。Tesseract 对复杂中文表格不稳定。这个模块主要用于演示材料识别入口，生产环境建议换更强 OCR。

### 14.6 为什么 `QWEN_LOCAL_FILES_ONLY=1` 时报找不到模型

说明该模型还没有完整下载到本地缓存。第一次使用时设置：

```powershell
$env:QWEN_LOCAL_FILES_ONLY="0"
```

下载完成后再改为：

```powershell
$env:QWEN_LOCAL_FILES_ONLY="1"
```

## 15. 后续扩展模块

### 15.1 数据层

- 接入真实高校教务处公开政策、通知和办事指南。
- 增加来源 URL、生效时间、废止时间、学院范围等元数据。
- 增加文档版本管理和增量入库。

### 15.2 检索层

- 增加混合检索：BM25 + embedding。
- 使用更强 reranker 模型。
- 增加引用覆盖率和召回命中率评估。

### 15.3 模型层

- 收集人工审核问答，构建 SFT 数据集。
- 使用 Qwen 基座进行 LoRA 微调。
- 使用 DPO 做偏好对齐。

### 15.4 服务层

- 接入登录系统。
- 增加用户画像和多轮记忆。
- 增加后台管理页面。
- 支持 Docker 部署、云服务器部署或企业内网部署。

### 15.5 合规模块

- 增加更严格的学生个人信息脱敏。
- 增加行级权限过滤。
- 增加审计日志和人工转接机制。
