"""Portable verification for the frozen TouchAgent SFT dataset."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from .config import SFTConfig
from .data import load_instruct_file, read_uncompressed_bytes
from .serialization import assistant_turn_count


MANIFEST_SCHEMA = "touchagent.sft_frozen_data_manifest.v1"
SCHEMA_TO_SOURCE = {
    "touchagent_attribute_instruct_v2": "attribute",
    "touchagent_matching_instruct_v2": "matching",
    "touchagent_interaction_instruct_v2": "interaction",
    "touchagent_scene_instruct_v2": "scene",
    "touchagent_dynamic_instruct_v2": "dynamic",
}


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_lines(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def build_runtime_data_facts(config: SFTConfig) -> Dict[str, Any]:
    data_path = Path(config.data_path)
    records = load_instruct_file(data_path)
    ids = [record["id"] for record in records]
    source_counts: Counter[str] = Counter()
    schema_counts: Counter[str] = Counter()
    assistant_supervisions = 0
    for record in records:
        schema = record["schema_version"]
        try:
            source = SCHEMA_TO_SOURCE[schema]
        except KeyError as exc:
            raise ValueError(f"Unsupported dataset schema: {schema}") from exc
        source_counts[source] += 1
        schema_counts[schema] += 1
        assistant_supervisions += assistant_turn_count(record)
    raw_payload = read_uncompressed_bytes(data_path)
    return {
        "data_file": data_path.name,
        "compressed_sha256": sha256_file(data_path),
        "uncompressed_sha256": sha256_bytes(raw_payload),
        "record_count": len(records),
        "assistant_supervision_count": assistant_supervisions,
        "global_ids_unique": len(ids) == len(set(ids)),
        "selected_id_order_sha256": sha256_lines(ids),
        "selected_source_counts": dict(sorted(source_counts.items())),
        "schema_version_counts": dict(sorted(schema_counts.items())),
    }


def verify_data_manifest(config: SFTConfig) -> Dict[str, Any]:
    manifest_path = Path(config.data_manifest_path)
    with manifest_path.open("r", encoding="utf-8") as handle:
        expected = json.load(handle)
    if expected.get("schema_version") != MANIFEST_SCHEMA:
        raise ValueError(f"Unsupported data manifest: {manifest_path}")
    if expected.get("dataset_version") != config.dataset_version:
        raise ValueError("Dataset version does not match the training config")
    actual = build_runtime_data_facts(config)
    mismatches = {
        key: {"expected": expected.get(key), "actual": value}
        for key, value in actual.items()
        if expected.get(key) != value
    }
    if mismatches:
        raise ValueError(
            "Frozen SFT data does not match its manifest: "
            + json.dumps(mismatches, ensure_ascii=False, sort_keys=True)
        )
    return expected


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    import os

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
