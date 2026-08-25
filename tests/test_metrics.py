import json
import tempfile
import unittest
from pathlib import Path

from touchagent_train.metrics import read_loss_points, render_loss_png, render_loss_svg


class TrainingMetricsTest(unittest.TestCase):
    def test_loss_points_and_renderers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metrics = root / "metrics.jsonl"
            metrics.write_text(
                "\n".join(
                    (
                        json.dumps({"step": 1, "loss": 1.25}),
                        json.dumps({"step": 2, "learning_rate": 0.0002}),
                        "incomplete",
                        json.dumps({"step": 3, "loss": 0.75}),
                    )
                ),
                encoding="utf-8",
            )
            points = read_loss_points(metrics)
            self.assertEqual(points, [(1, 1.25), (3, 0.75)])
            svg = root / "loss.svg"
            png = root / "loss.png"
            render_loss_svg(points, svg)
            render_loss_png(points, png)
            self.assertIn("latest loss 0.7500", svg.read_text(encoding="utf-8"))
            self.assertTrue(png.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))


if __name__ == "__main__":
    unittest.main()
