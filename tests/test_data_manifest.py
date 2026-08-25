import gzip
import json
import tempfile
import unittest
from pathlib import Path

from touchagent_train.data import load_instruct_file
from touchagent_train.manifest import MANIFEST_SCHEMA, build_runtime_data_facts, verify_data_manifest

from tests.helpers import make_config, make_record


class FrozenDataManifestTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.data_path = self.root / "data.json.gz"
        self.manifest_path = self.root / "data.manifest.json"
        raw = (
            json.dumps(
                [make_record(record_id="one"), make_record(2, record_id="two")],
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
        self.data_path.write_bytes(gzip.compress(raw, compresslevel=9, mtime=0))
        self.config = make_config(self.root, self.data_path, self.manifest_path)
        manifest = {
            "schema_version": MANIFEST_SCHEMA,
            "dataset_version": "TouchAgent-Instruct-v2",
            **build_runtime_data_facts(self.config),
        }
        self.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    def tearDown(self):
        self.temporary.cleanup()

    def test_gzip_loader_preserves_record_order(self):
        self.assertEqual(
            [record["id"] for record in load_instruct_file(self.data_path)],
            ["one", "two"],
        )

    def test_manifest_verifies_and_detects_recompression(self):
        verified = verify_data_manifest(self.config)
        self.assertEqual(verified["record_count"], 2)
        raw = gzip.decompress(self.data_path.read_bytes())
        self.data_path.write_bytes(gzip.compress(raw, compresslevel=1, mtime=0))
        with self.assertRaisesRegex(ValueError, "does not match"):
            verify_data_manifest(self.config)


if __name__ == "__main__":
    unittest.main()
