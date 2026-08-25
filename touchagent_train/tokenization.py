"""Compatibility helpers for tokenizer outputs across Transformers versions."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, List


def extract_input_ids(encoded: Any) -> Any:
    """Return input IDs from either a raw value or a BatchEncoding-like mapping."""

    if isinstance(encoded, Mapping):
        if "input_ids" not in encoded:
            raise ValueError("Tokenizer output does not contain input_ids")
        return encoded["input_ids"]
    return encoded


def input_ids_to_list(encoded: Any) -> List[int]:
    """Normalize unbatched tokenizer output to a flat integer list."""

    values = extract_input_ids(encoded)
    if hasattr(values, "tolist"):
        values = values.tolist()
    if isinstance(values, tuple):
        values = list(values)
    if not isinstance(values, list):
        raise ValueError(
            f"Tokenizer input_ids must be list-like, got {type(values).__name__}"
        )
    if values and isinstance(values[0], (list, tuple)):
        if len(values) != 1:
            raise ValueError("Expected one unbatched tokenizer sequence")
        values = list(values[0])
    return [int(token) for token in values]
