"""Rank-zero JSONL metrics and dependency-free loss-curve rendering."""

from __future__ import annotations

import json
import math
import struct
import time
import zlib
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO, Tuple


METRICS_FILENAME = "training_metrics.jsonl"


def build_jsonl_metrics_callback(output_dir: str):
    """Create a lazy Transformers callback that persists every log event."""

    from transformers import TrainerCallback

    metrics_path = Path(output_dir) / METRICS_FILENAME

    class JSONLMetricsCallback(TrainerCallback):
        def __init__(self) -> None:
            self._handle: Optional[TextIO] = None

        def on_train_begin(self, args, state, control, **kwargs):
            if state.is_world_process_zero:
                metrics_path.parent.mkdir(parents=True, exist_ok=True)
                self._handle = metrics_path.open("a", encoding="utf-8")

        def on_log(self, args, state, control, logs=None, **kwargs):
            if self._handle is None or not logs:
                return
            payload: Dict[str, Any] = {
                "timestamp": time.time(),
                "step": int(state.global_step),
                "epoch": float(state.epoch) if state.epoch is not None else None,
            }
            for key, value in logs.items():
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    payload[key] = value
            self._handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
            self._handle.flush()

        def on_train_end(self, args, state, control, **kwargs):
            if self._handle is not None:
                self._handle.close()
                self._handle = None

    return JSONLMetricsCallback()


def read_loss_points(path: Path) -> List[Tuple[int, float]]:
    points: List[Tuple[int, float]] = []
    if not path.is_file():
        return points
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                payload = json.loads(line)
                step = int(payload["step"])
                loss = float(payload["loss"])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
            if math.isfinite(loss):
                points.append((step, loss))
    return points


def render_loss_svg(points: List[Tuple[int, float]], output_path: Path) -> None:
    width, height = 1200, 700
    left, right, top, bottom = 90, 40, 55, 80
    plot_width = width - left - right
    plot_height = height - top - bottom

    if points:
        max_step = max(step for step, _ in points)
        values = [loss for _, loss in points]
        min_loss, max_loss = min(values), max(values)
        padding = max((max_loss - min_loss) * 0.08, 0.02)
        y_min = max(0.0, min_loss - padding)
        y_max = max_loss + padding
    else:
        max_step, y_min, y_max = 1, 0.0, 1.0

    def x_position(step: int) -> float:
        return left + plot_width * step / max(max_step, 1)

    def y_position(loss: float) -> float:
        return top + plot_height * (y_max - loss) / max(y_max - y_min, 1e-9)

    grid = []
    for index in range(6):
        fraction = index / 5
        y = top + plot_height * fraction
        value = y_max - (y_max - y_min) * fraction
        grid.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" '
            'stroke="#d9dee7" stroke-width="1"/>'
        )
        grid.append(
            f'<text x="{left-14}" y="{y+5:.1f}" text-anchor="end" '
            f'font-size="15" fill="#4b5563">{value:.3f}</text>'
        )

    for index in range(6):
        fraction = index / 5
        x = left + plot_width * fraction
        step = round(max_step * fraction)
        grid.append(
            f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{height-bottom}" '
            'stroke="#edf0f4" stroke-width="1"/>'
        )
        grid.append(
            f'<text x="{x:.1f}" y="{height-bottom+28}" text-anchor="middle" '
            f'font-size="15" fill="#4b5563">{step}</text>'
        )

    polyline = " ".join(
        f"{x_position(step):.2f},{y_position(loss):.2f}" for step, loss in points
    )
    latest = points[-1][1] if points else None
    minimum = min((loss for _, loss in points), default=None)
    status = "Waiting for the first logged optimizer step"
    if latest is not None and minimum is not None:
        status = (
            f"Step {points[-1][0]:,}  |  latest loss {latest:.4f}  |  "
            f"minimum loss {minimum:.4f}"
        )

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#ffffff"/>
<text x="{left}" y="32" font-family="sans-serif" font-size="22" font-weight="600" fill="#111827">TouchAgent Training Loss</text>
<text x="{width-right}" y="32" text-anchor="end" font-family="sans-serif" font-size="15" fill="#4b5563">{status}</text>
<g font-family="sans-serif">{''.join(grid)}</g>
<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#111827" stroke-width="1.5"/>
<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#111827" stroke-width="1.5"/>
<text x="{left+plot_width/2:.1f}" y="{height-20}" text-anchor="middle" font-family="sans-serif" font-size="17" fill="#111827">Optimizer step</text>
<text x="24" y="{top+plot_height/2:.1f}" text-anchor="middle" transform="rotate(-90 24 {top+plot_height/2:.1f})" font-family="sans-serif" font-size="17" fill="#111827">Loss</text>
<polyline points="{polyline}" fill="none" stroke="#1565c0" stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round"/>
</svg>
'''
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    temporary.write_text(svg, encoding="utf-8")
    temporary.replace(output_path)


def render_loss_png(points: List[Tuple[int, float]], output_path: Path) -> None:
    """Render a VS Code-friendly PNG with raw and 50-step mean loss."""

    width, height = 1200, 700
    left, right, top, bottom = 70, 35, 35, 45
    pixels = bytearray([255]) * (width * height * 3)

    def set_pixel(x: int, y: int, color: Tuple[int, int, int]) -> None:
        if 0 <= x < width and 0 <= y < height:
            offset = (y * width + x) * 3
            pixels[offset : offset + 3] = bytes(color)

    def draw_line(
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        color: Tuple[int, int, int],
        thickness: int = 1,
    ) -> None:
        x_start, y_start = int(round(x0)), int(round(y0))
        x_end, y_end = int(round(x1)), int(round(y1))
        dx, dy = abs(x_end - x_start), -abs(y_end - y_start)
        sx = 1 if x_start < x_end else -1
        sy = 1 if y_start < y_end else -1
        error = dx + dy
        while True:
            radius = max(0, thickness // 2)
            for x_offset in range(-radius, radius + 1):
                for y_offset in range(-radius, radius + 1):
                    set_pixel(x_start + x_offset, y_start + y_offset, color)
            if x_start == x_end and y_start == y_end:
                break
            doubled = 2 * error
            if doubled >= dy:
                error += dy
                x_start += sx
            if doubled <= dx:
                error += dx
                y_start += sy

    def moving_average(window: int) -> List[Tuple[int, float]]:
        values: List[Tuple[int, float]] = []
        running = 0.0
        queue: List[float] = []
        for step, loss in points:
            queue.append(loss)
            running += loss
            if len(queue) > window:
                running -= queue.pop(0)
            values.append((step, running / len(queue)))
        return values

    averaged = moving_average(50)
    max_step = max((step for step, _ in points), default=1)
    max_loss = max((loss for _, loss in points), default=1.0)
    max_average = max((loss for _, loss in averaged), default=1.0)
    panels = [
        (top, 360, points, max_loss, (21, 101, 192), 2),
        (405, height - bottom, averaged, max_average, (198, 74, 46), 3),
    ]
    for panel_top, panel_bottom, values, y_max, color, thickness in panels:
        plot_width = width - left - right
        plot_height = panel_bottom - panel_top
        for index in range(6):
            y = panel_top + plot_height * index / 5
            draw_line(left, y, width - right, y, (222, 226, 232))
        for index in range(6):
            x = left + plot_width * index / 5
            draw_line(x, panel_top, x, panel_bottom, (238, 240, 244))
        draw_line(left, panel_bottom, width - right, panel_bottom, (17, 24, 39), 2)
        draw_line(left, panel_top, left, panel_bottom, (17, 24, 39), 2)
        previous = None
        for step, loss in values:
            x = left + plot_width * step / max(max_step, 1)
            y = panel_top + plot_height * (y_max - loss) / max(y_max, 1e-9)
            if previous is not None:
                draw_line(previous[0], previous[1], x, y, color, thickness)
            previous = (x, y)

    def chunk(kind: bytes, payload: bytes) -> bytes:
        checksum = zlib.crc32(kind)
        checksum = zlib.crc32(payload, checksum)
        return (
            struct.pack(">I", len(payload))
            + kind
            + payload
            + struct.pack(">I", checksum & 0xFFFFFFFF)
        )

    scanlines = bytearray()
    row_size = width * 3
    for row in range(height):
        scanlines.append(0)
        start = row * row_size
        scanlines.extend(pixels[start : start + row_size])
    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(bytes(scanlines), level=6))
    png += chunk(b"IEND", b"")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    temporary.write_bytes(png)
    temporary.replace(output_path)


def write_live_html(svg_path: Path, html_path: Path, interval_seconds: float) -> None:
    refresh_ms = max(1000, int(interval_seconds * 1000))
    html = f'''<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>TouchAgent Training Loss</title></head>
<body style="margin:0;background:#fff"><img id="curve" src="{svg_path.name}" style="display:block;width:100%;height:auto"></body>
<script>setInterval(function(){{document.getElementById('curve').src='{svg_path.name}?t='+Date.now();}},{refresh_ms});</script>
</html>
'''
    temporary = html_path.with_name(html_path.name + ".tmp")
    temporary.write_text(html, encoding="utf-8")
    temporary.replace(html_path)
