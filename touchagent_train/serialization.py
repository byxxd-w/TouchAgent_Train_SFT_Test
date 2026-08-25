"""Variable-turn validation and assistant-only token labeling."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Mapping, Sequence

from .prompts import SYSTEM_PROMPT
from .protocol import AgentOutput, AgentPhase, ToolName, validate_agent_output_for_phase
from .tokenization import input_ids_to_list


FORBIDDEN_PUBLIC_TEXT = (
    "/data/",
    "/home/",
    "/mnt/",
    "/private/",
    "task_type",
    "subtask",
    "gold_answer",
    "object_id",
    "raw_logit",
    "prob_success",
    "checkpoint",
)
OBSERVATION_HEADER_PATTERN = re.compile(
    r"(?m)^([A-Za-z][A-Za-z0-9_]*Tool) output:\s*$"
)


def observation_tool_names(observation: str) -> List[ToolName]:
    names = OBSERVATION_HEADER_PATTERN.findall(observation)
    if not names:
        raise ValueError("Observation does not contain a tool output header")
    return [ToolName(name) for name in names]


def serialize_assistant_output(turn: Mapping[str, Any]) -> str:
    if set(turn) != {"from", "thoughts", "actions", "value"}:
        raise ValueError("Assistant turn has unexpected or missing fields")
    output = AgentOutput.model_validate(
        {
            "thoughts": turn["thoughts"],
            "actions": turn["actions"],
            "value": turn["value"],
        }
    )
    return json.dumps(output.model_dump(mode="json"), ensure_ascii=False, indent=2)


def validate_instruct_record(record: Mapping[str, Any]) -> None:
    if set(record) != {"schema_version", "id", "conversations"}:
        raise ValueError("Instruct record has unexpected or missing top-level fields")
    schema_version = record["schema_version"]
    if not isinstance(schema_version, str) or not schema_version.endswith("_v2"):
        raise ValueError("Only TouchAgent-Instruct-v2 schemas are supported")
    if not isinstance(record["id"], str) or not record["id"]:
        raise ValueError("Instruct id must be a non-empty string")
    turns = record["conversations"]
    if not isinstance(turns, list) or len(turns) < 4 or len(turns) % 2:
        raise ValueError("Instruct data must contain an even number of at least four turns")
    expected_roles = tuple(
        "human" if index % 2 == 0 else "gpt" for index in range(len(turns))
    )
    if tuple(turn.get("from") for turn in turns) != expected_roles:
        raise ValueError("Instruct turn roles must alternate human/gpt")

    for index in range(0, len(turns), 2):
        turn = turns[index]
        if set(turn) != {"from", "value"}:
            raise ValueError("Human turn has unexpected or missing fields")
        if not isinstance(turn["value"], str) or not turn["value"].strip():
            raise ValueError("Human value must be a non-empty string")

    public_text = "\n".join(
        str(turn.get(field, ""))
        for turn in turns
        for field in ("thoughts", "value")
    ).lower()
    leaked = [marker for marker in FORBIDDEN_PUBLIC_TEXT if marker in public_text]
    if leaked:
        raise ValueError(f"Instruct record contains private text markers: {leaked}")

    assistant_indices = list(range(1, len(turns), 2))
    for turn_index in assistant_indices[:-1]:
        output = AgentOutput.model_validate_json(
            serialize_assistant_output(turns[turn_index])
        )
        validate_agent_output_for_phase(output, AgentPhase.TOOL_SELECTION)
        expected_tools = [action.tool_name for action in output.actions]
        observed_tools = observation_tool_names(turns[turn_index + 1]["value"])
        if observed_tools != expected_tools:
            raise ValueError(
                f"Observation after Assistant turn {turn_index} does not match "
                "selected tools in order"
            )

    final = AgentOutput.model_validate_json(
        serialize_assistant_output(turns[assistant_indices[-1]])
    )
    validate_agent_output_for_phase(final, AgentPhase.FINAL_ANSWER)


def assistant_turn_count(record: Mapping[str, Any]) -> int:
    validate_instruct_record(record)
    return len(record["conversations"]) // 2


def _template_ids(
    tokenizer: Any, messages: Sequence[Dict[str, str]], add_prompt: bool
) -> List[int]:
    ids = tokenizer.apply_chat_template(
        list(messages), tokenize=True, add_generation_prompt=add_prompt
    )
    return input_ids_to_list(ids)


def _require_prefix(prefix: Sequence[int], full: Sequence[int], context: str) -> None:
    if list(full[: len(prefix)]) != list(prefix):
        raise ValueError(f"Tokenizer chat template is not prefix-stable at {context}")


def encode_instruct_record(
    tokenizer: Any, record: Mapping[str, Any], max_length: int = 2048
) -> Dict[str, Any]:
    validate_instruct_record(record)
    messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    input_ids = _template_ids(tokenizer, messages, add_prompt=False)
    labels = [-100] * len(input_ids)
    assistant_spans: List[List[int]] = []
    im_end_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if not isinstance(im_end_token_id, int) or im_end_token_id < 0:
        raise ValueError("Tokenizer does not define a valid <|im_end|> token")

    for turn_index, turn in enumerate(record["conversations"]):
        if turn["from"] == "human":
            messages.append({"role": "user", "content": turn["value"]})
            expanded = _template_ids(tokenizer, messages, add_prompt=False)
            _require_prefix(input_ids, expanded, f"human turn {turn_index}")
            labels.extend([-100] * (len(expanded) - len(input_ids)))
            input_ids = expanded
            continue

        assistant_json = serialize_assistant_output(turn)
        generation_prefix = _template_ids(tokenizer, messages, add_prompt=True)
        _require_prefix(input_ids, generation_prefix, f"assistant prefix {turn_index}")
        labels.extend([-100] * (len(generation_prefix) - len(input_ids)))
        input_ids = generation_prefix

        messages.append({"role": "assistant", "content": assistant_json})
        expanded = _template_ids(tokenizer, messages, add_prompt=False)
        _require_prefix(input_ids, expanded, f"assistant content {turn_index}")
        start = len(input_ids)
        delta = expanded[start:]
        end_positions = [
            offset for offset, token_id in enumerate(delta) if token_id == im_end_token_id
        ]
        if not end_positions:
            raise ValueError(f"Assistant turn {turn_index} is missing <|im_end|>")
        supervised_end = start + end_positions[-1] + 1
        labels.extend(expanded[start:supervised_end])
        labels.extend([-100] * (len(expanded) - supervised_end))
        input_ids = expanded
        assistant_spans.append([start, supervised_end])

    if len(input_ids) != len(labels):
        raise AssertionError("input_ids and labels length mismatch")
    if len(input_ids) > max_length:
        raise ValueError(
            f"Tokenized record {record['id']} has {len(input_ids)} tokens, "
            f"exceeding max_length={max_length}; truncation is forbidden"
        )
    if not assistant_spans or len(assistant_spans) != len(record["conversations"]) // 2:
        raise ValueError("Every Assistant turn must provide a supervised target")
    return {
        "id": record["id"],
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
        "assistant_spans": assistant_spans,
    }
