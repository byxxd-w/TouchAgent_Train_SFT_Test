"""Frozen gzip dataset loading and causal-LM collation."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from .serialization import encode_instruct_record, validate_instruct_record


def read_uncompressed_bytes(path: Path) -> bytes:
    if path.suffix == ".gz":
        with gzip.open(path, "rb") as handle:
            return handle.read()
    return path.read_bytes()


def load_instruct_file(path: Path) -> List[Dict[str, Any]]:
    try:
        records = json.loads(read_uncompressed_bytes(path).decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to read Instruct data: {path}") from exc
    if not isinstance(records, list) or not records:
        raise ValueError(f"Instruct file must contain a non-empty JSON list: {path}")
    seen = set()
    for record in records:
        validate_instruct_record(record)
        record_id = record["id"]
        if record_id in seen:
            raise ValueError(f"Duplicate Instruct id in {path}: {record_id}")
        seen.add(record_id)
    return records


class TouchAgentSFTDataset:
    def __init__(self, tokenizer: Any, records: Sequence[Mapping[str, Any]], max_length: int):
        self.examples = [
            encode_instruct_record(tokenizer, record, max_length=max_length)
            for record in records
        ]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        value = dict(self.examples[index])
        value.pop("id", None)
        value.pop("assistant_spans", None)
        return value


class AssistantOnlyDataCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("Training data collation requires PyTorch") from exc
        max_length = max(len(item["input_ids"]) for item in features)
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for item in features:
            padding = max_length - len(item["input_ids"])
            batch["input_ids"].append(
                list(item["input_ids"]) + [self.pad_token_id] * padding
            )
            batch["attention_mask"].append(
                list(item["attention_mask"]) + [0] * padding
            )
            batch["labels"].append(list(item["labels"]) + [-100] * padding)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}
