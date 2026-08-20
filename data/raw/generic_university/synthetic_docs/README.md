# 高校通用模拟数据教务智能助手模拟资料包

重要说明：本文件夹内资料均为“项目开发用模拟资料”，不是高校通用模拟数据正式文件，不可作为真实教务办理依据。

这批资料用于在真实 PDF、通知、表格尚未下载齐全前，先跑通：

- 文档清洗
- 文本切分
- RAG 检索
- 引用返回
- LangGraph 流程问答
- Guardrails 和权限测试

正式上线前必须替换为学校官网、教务处官网、学院官网或校内系统发布的正式资料。

## 目录说明

```text
policies/  政策制度、管理办法
guides/    办事指南、流程说明
forms/     表格模板、申请表字段说明
notices/   通知公告、校历、安排类资料
reports/   教学质量、教学管理、统计报告类资料
```

## 建议使用方式

先把这些 Markdown 文件复制或直接作为输入目录执行切分：

```powershell
python scripts/ingest_documents.py --input-dir data/raw/generic_university/synthetic_docs/policies --output data/processed/generic_university/policies_chunks.jsonl --tenant-id generic-university
```

如果要一次处理全部子目录，后续可以把 `ingest_documents.py` 改成递归扫描。

