#!/usr/bin/env python3
"""Render or continuously update the formal-training loss curve."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from touchagent_train.metrics import (
    read_loss_points,
    render_loss_png,
    render_loss_svg,
    write_live_html,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "runs" / "current"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path, default=DEFAULT_RUN / "training_metrics.jsonl"
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_RUN / "loss_curve.svg"
    )
    parser.add_argument("--follow", action="store_true")
    parser.add_argument("--interval", type=float, default=10.0)
    args = parser.parse_args()
    if args.interval <= 0:
        raise ValueError("--interval must be positive")

    while True:
        points = read_loss_points(args.input)
        if points:
            render_loss_svg(points, args.output)
            render_loss_png(points, args.output.with_suffix(".png"))
            write_live_html(
                args.output,
                args.output.with_name("loss_curve.html"),
                args.interval,
            )
        if not args.follow:
            if not points:
                raise FileNotFoundError(f"No loss records found in {args.input}")
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
