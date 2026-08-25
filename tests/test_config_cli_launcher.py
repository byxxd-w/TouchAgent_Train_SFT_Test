import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from touchagent_train.cli import build_parser
from touchagent_train.config import load_sft_config


ROOT = Path(__file__).resolve().parents[1]


class ConfigCliLauncherTest(unittest.TestCase):
    def test_config_paths_resolve_from_config_directory(self):
        config = load_sft_config(ROOT / "configs/train_sft32k_instruct_v2_smoke.json")
        self.assertEqual(config.data_path, str((ROOT / "data/touchagent_sft32k_instruct_v2.json.gz").resolve()))
        self.assertEqual(config.model_name_or_path, str((ROOT / "models/Qwen2.5-7B").resolve()))
        self.assertEqual(config.max_steps, 3)

    def test_runtime_path_overrides(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = load_sft_config(
                ROOT / "configs/train_sft32k_instruct_v2.json",
                model_path=root / "model",
                output_dir=root / "output",
            )
            self.assertEqual(config.model_name_or_path, str((root / "model").resolve()))
            self.assertEqual(config.output_dir, str((root / "output").resolve()))

    def test_cli_exposes_only_sft_commands(self):
        parser = build_parser()
        help_text = parser.format_help()
        for command in ("audit", "preflight", "verify-manifest", "train"):
            self.assertIn(command, help_text)
        for forbidden in ("split", "rl", "grpo"):
            self.assertNotIn(forbidden, help_text.lower())

    def test_launcher_rejects_non_four_gpu_topology(self):
        environment = dict(os.environ)
        environment["CUDA_DEVICES"] = "0,1,2"
        result = subprocess.run(
            ["bash", str(ROOT / "scripts/run_sft32k_instruct_v2.sh")],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("exactly four", result.stderr)

    def test_configs_are_json_and_have_distinct_output_directories(self):
        formal = json.loads((ROOT / "configs/train_sft32k_instruct_v2.json").read_text())
        smoke = json.loads((ROOT / "configs/train_sft32k_instruct_v2_smoke.json").read_text())
        self.assertNotEqual(formal["output_dir"], smoke["output_dir"])
        self.assertEqual(formal["num_train_epochs"], 5.0)
        self.assertEqual(smoke["max_steps"], 3)


if __name__ == "__main__":
    unittest.main()
