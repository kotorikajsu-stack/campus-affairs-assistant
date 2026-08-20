"""Qwen LoRA training entrypoint.

The heavy training stack is kept out of the import path used by tests and API
startup. Install GPU dependencies before running this script in a training
environment.
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/finetune/lora.yaml")
    args = parser.parse_args()

    raise SystemExit(
        "LoRA training template is ready. Wire transformers, peft, datasets, "
        f"and trl using config: {args.config}"
    )


if __name__ == "__main__":
    main()

