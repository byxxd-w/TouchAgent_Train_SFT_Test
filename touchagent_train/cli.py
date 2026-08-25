"""Command-line interface for TouchAgent SFT only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from .config import SFTConfig, load_sft_config
from .manifest import atomic_write_json, verify_data_manifest
from .trainer import audit_training_data, audit_training_runtime, train_lora


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="touchagent-train-sft")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("audit", "preflight", "verify-manifest", "train"):
        command = subparsers.add_parser(name)
        command.add_argument("--config", type=Path, required=True)
        command.add_argument("--model-path", type=Path)
        command.add_argument("--output-dir", type=Path)
        if name == "audit":
            command.add_argument("--output", type=Path)
        if name == "train":
            command.add_argument("--resume-from-checkpoint", type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_sft_config(
        args.config,
        model_path=args.model_path,
        output_dir=args.output_dir,
    )
    if args.command == "audit":
        payload = audit_training_data(config)
        if args.output is not None:
            atomic_write_json(args.output.resolve(), payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    if args.command == "preflight":
        print(json.dumps(audit_training_runtime(config), ensure_ascii=False, indent=2))
        return 0
    if args.command == "verify-manifest":
        payload = verify_data_manifest(config)
        print(
            "Frozen SFT manifest verified: "
            f"records={payload['record_count']}, "
            f"sha256={payload['selected_id_order_sha256']}"
        )
        return 0
    if args.command == "train":
        if args.resume_from_checkpoint is not None:
            config = SFTConfig.model_validate(
                {
                    **config.model_dump(),
                    "resume_from_checkpoint": str(
                        args.resume_from_checkpoint.expanduser().resolve()
                    ),
                }
            )
        train_lora(config)
        return 0
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
