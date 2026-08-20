"""Parse raw documents and rebuild the local Milvus knowledge base.

This script is the "one command from source files to Milvus" entry point.

Typical workflow:
1. Put PDF/image/Office files into data/raw/generic_university/{policies,guides,...}.
2. Install MinerU if you need to parse those source files.
3. Run this script.

The script will:
- Parse raw files with MinerU into Markdown under data/processed/generic_university/mineru.
- Reuse refresh_knowledge_base.py to chunk Markdown/TXT files.
- Vectorize chunks and rebuild the local Milvus Lite database.

MinerU is intentionally optional because it is a heavy dependency. The GitHub
project keeps the integration script, but does not vendor MinerU models or
generated parsing outputs.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_TENANT_ID = "generic-university"
DEFAULT_PROCESSED_DIR = "data/processed/generic_university"
DEFAULT_MINERU_OUTPUT_DIR = "data/processed/generic_university/mineru"
DEFAULT_RAW_DIRS = [
    "data/raw/generic_university/policies",
    "data/raw/generic_university/guides",
    "data/raw/generic_university/forms",
    "data/raw/generic_university/notices",
    "data/raw/generic_university/reports",
    "data/raw/generic_university/inbox",
]
SUPPORTED_RAW_SUFFIXES = {
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".webp",
    ".docx",
    ".pptx",
    ".xlsx",
}


def resolve_project_path(path_text: str) -> Path:
    """Resolve a path relative to the project root."""

    path = Path(path_text)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def find_raw_files(raw_dirs: list[str]) -> list[Path]:
    """Find raw files that MinerU can parse."""

    files: list[Path] = []

    for raw_dir_text in raw_dirs:
        raw_dir = resolve_project_path(raw_dir_text)
        if not raw_dir.exists():
            continue

        for path in sorted(raw_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in SUPPORTED_RAW_SUFFIXES:
                files.append(path)

    return list(dict.fromkeys(files))


def run_mineru(
    *,
    input_path: Path,
    output_dir: Path,
    method: str,
    backend: str,
    model_source: str,
    dry_run: bool,
) -> None:
    """Run MinerU CLI for one source file."""

    command = [
        "mineru",
        "-p",
        str(input_path),
        "-o",
        str(output_dir),
        "-m",
        method,
        "-b",
        backend,
    ]

    print("[MinerU]", " ".join(command))

    if dry_run:
        return

    env = os.environ.copy()
    env["MINERU_MODEL_SOURCE"] = model_source
    result = subprocess.run(command, env=env, check=False)

    if result.returncode != 0:
        raise RuntimeError(f"MinerU 解析失败：{input_path}，returncode={result.returncode}")


def run_refresh_knowledge_base(
    *,
    tenant_id: str,
    processed_dir: str,
    dry_run: bool,
) -> None:
    """Run the existing chunk + Milvus rebuild script."""

    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "refresh_knowledge_base.py"),
        "--tenant-id",
        tenant_id,
        "--source-dir",
        "data/raw/generic_university/synthetic_docs",
        "--source-dir",
        processed_dir,
    ]

    if dry_run:
        command.append("--dry-run")

    print("[刷新知识库]", " ".join(command))
    result = subprocess.run(command, cwd=PROJECT_ROOT, check=False)

    if result.returncode != 0:
        raise RuntimeError(f"知识库刷新失败，returncode={result.returncode}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="从 PDF/图片/Office 原始文件解析并一键重建本地 Milvus 知识库。"
    )
    parser.add_argument(
        "--raw-dir",
        action="append",
        dest="raw_dirs",
        help="原始文件目录，可重复传入。默认扫描 generic_university 下的分类目录。",
    )
    parser.add_argument(
        "--processed-dir",
        default=DEFAULT_PROCESSED_DIR,
        help="Markdown/TXT 处理后目录，也是知识库刷新脚本扫描的目录。",
    )
    parser.add_argument(
        "--mineru-output-dir",
        default=DEFAULT_MINERU_OUTPUT_DIR,
        help="MinerU 解析输出目录。",
    )
    parser.add_argument(
        "--tenant-id",
        default=DEFAULT_TENANT_ID,
        help="租户/学校 ID。",
    )
    parser.add_argument(
        "--method",
        choices=["auto", "txt", "ocr"],
        default="auto",
        help="MinerU 解析方法。",
    )
    parser.add_argument(
        "--backend",
        choices=[
            "pipeline",
            "vlm-engine",
            "hybrid-engine",
            "vlm-http-client",
            "hybrid-http-client",
        ],
        default="pipeline",
        help="MinerU 后端。Windows 本地优先使用 pipeline。",
    )
    parser.add_argument(
        "--model-source",
        choices=["huggingface", "modelscope"],
        default="modelscope",
        help="MinerU 模型下载源。国内环境通常优先 modelscope。",
    )
    parser.add_argument(
        "--skip-mineru",
        action="store_true",
        help="跳过 MinerU，只基于 processed 目录里已有 Markdown/TXT 重建 Milvus。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印会执行的命令，并运行知识库 dry-run，不真正写入 Milvus。",
    )
    return parser.parse_args()


def main() -> None:
    """Script entry point."""

    args = parse_args()
    raw_dirs = args.raw_dirs or DEFAULT_RAW_DIRS
    mineru_output_dir = resolve_project_path(args.mineru_output_dir)

    print("[开始] 从原始文档重建高校教务知识库")
    print(f"[租户] {args.tenant_id}")

    raw_files = find_raw_files(raw_dirs)
    print(f"[原始文件] 可解析文件数量：{len(raw_files)}")
    for raw_file in raw_files:
        print(f"  - {raw_file.relative_to(PROJECT_ROOT)}")

    if raw_files and not args.skip_mineru:
        if shutil.which("mineru") is None:
            raise RuntimeError(
                "当前环境未找到 MinerU CLI，无法解析 PDF/图片/Office 原始文件。\n"
                "请先安装：pip install uv && uv pip install -U \"mineru[all]\"\n"
                "如果你已经手动把文件解析成 Markdown/TXT，"
                "可以加 --skip-mineru 只重建 Milvus。"
            )

        if not args.dry_run:
            mineru_output_dir.mkdir(parents=True, exist_ok=True)

        for raw_file in raw_files:
            run_mineru(
                input_path=raw_file,
                output_dir=mineru_output_dir,
                method=args.method,
                backend=args.backend,
                model_source=args.model_source,
                dry_run=args.dry_run,
            )
    elif raw_files and args.skip_mineru:
        print("[跳过] 已按参数要求跳过 MinerU 解析。")
    else:
        print("[提示] 未发现 PDF/图片/Office 原始文件，将直接基于现有 Markdown/TXT 重建。")

    run_refresh_knowledge_base(
        tenant_id=args.tenant_id,
        processed_dir=args.processed_dir,
        dry_run=args.dry_run,
    )
    print("[完成] 原始文档到本地 Milvus 的流程已结束。")


if __name__ == "__main__":
    main()
