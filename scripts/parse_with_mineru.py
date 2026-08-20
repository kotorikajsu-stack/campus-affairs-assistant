"""Run MinerU on PDF/image/Office files and keep a small run manifest.

MinerU is an optional heavy dependency. Install it in your active environment
before running this script:

    pip install uv
    uv pip install -U "mineru[all]"

Example:

    python scripts/parse_with_mineru.py --input data/raw/demo.pdf --backend pipeline
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


SUPPORTED_SUFFIXES = {
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".webp",
    ".gif",
    ".docx",
    ".pptx",
    ".xlsx",
}


def build_command(args: argparse.Namespace) -> list[str]:
    command = [
        "mineru",
        "-p",
        str(args.input),
        "-o",
        str(args.output),
        "-m",
        args.method,
        "-b",
        args.backend,
    ]
    if args.api_url:
        command.extend(["--api-url", args.api_url])
    if args.effort:
        command.extend(["--effort", args.effort])
    return command


def validate_input(path: Path) -> None:
    if not path.exists():
        raise SystemExit(f"Input path does not exist: {path}")

    if path.is_file() and path.suffix.lower() not in SUPPORTED_SUFFIXES:
        supported = ", ".join(sorted(SUPPORTED_SUFFIXES))
        raise SystemExit(f"Unsupported file type: {path.suffix}. Supported: {supported}")


def collect_outputs(output_dir: Path) -> list[str]:
    if not output_dir.exists():
        return []
    interesting_suffixes = {".md", ".json", ".html", ".txt", ".png", ".jpg", ".jpeg", ".pdf"}
    return [
        str(path)
        for path in sorted(output_dir.rglob("*"))
        if path.is_file() and path.suffix.lower() in interesting_suffixes
    ]


def write_manifest(
    output_dir: Path,
    *,
    command: list[str],
    input_path: Path,
    returncode: int,
    outputs: list[str],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "mineru_run_manifest.json"
    manifest = {
        "tool": "mineru",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input": str(input_path),
        "returncode": returncode,
        "command": command,
        "outputs": outputs,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Parse documents with MinerU CLI.")
    parser.add_argument("--input", required=True, type=Path, help="PDF/image/DOCX/PPTX/XLSX file or directory.")
    parser.add_argument("--output", type=Path, default=Path("data/processed/mineru"), help="Output directory.")
    parser.add_argument("--method", choices=["auto", "txt", "ocr"], default="auto", help="Parsing method.")
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
        help="Use pipeline first on Windows/CPU for better compatibility.",
    )
    parser.add_argument("--effort", choices=["medium", "high"], default=None, help="Hybrid parsing effort.")
    parser.add_argument("--api-url", default="", help="Existing mineru-api URL, optional.")
    parser.add_argument(
        "--model-source",
        choices=["huggingface", "modelscope"],
        default="modelscope",
        help="Model download source. modelscope is often easier in China.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the MinerU command without running it.")
    args = parser.parse_args()

    validate_input(args.input)

    if shutil.which("mineru") is None:
        print("MinerU CLI was not found in the current environment.", file=sys.stderr)
        print('Install it first: pip install uv && uv pip install -U "mineru[all]"', file=sys.stderr)
        print("Then retry this script in the same conda environment.", file=sys.stderr)
        return 2

    command = build_command(args)
    print("Running:", " ".join(command))

    if args.dry_run:
        return 0

    args.output.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["MINERU_MODEL_SOURCE"] = args.model_source

    result = subprocess.run(command, env=env, check=False)
    outputs = collect_outputs(args.output)
    manifest_path = write_manifest(
        args.output,
        command=command,
        input_path=args.input,
        returncode=result.returncode,
        outputs=outputs,
    )

    print(f"MinerU return code: {result.returncode}")
    print(f"Found {len(outputs)} output files.")
    print(f"Manifest: {manifest_path}")
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
