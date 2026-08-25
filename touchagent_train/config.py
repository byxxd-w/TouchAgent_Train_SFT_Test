"""Strict, repository-portable configuration for TouchAgent LoRA SFT."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Literal, Optional

from pydantic import Field, field_validator, model_validator

from .protocol import StrictModel


class LoraTrainingConfig(StrictModel):
    profile: Literal["chat_tokens_v2"] = "chat_tokens_v2"
    r: Literal[128] = 128
    alpha: Literal[256] = 256
    dropout: Literal[0.05] = 0.05
    target_modules: List[str] = Field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "lm_head",
        ]
    )
    modules_to_save: List[str] = Field(default_factory=lambda: ["embed_tokens"])

    @model_validator(mode="after")
    def validate_targets(self) -> "LoraTrainingConfig":
        expected_targets = {
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "lm_head",
        }
        if (
            set(self.target_modules) != expected_targets
            or len(self.target_modules) != len(expected_targets)
        ):
            raise ValueError(
                f"chat_tokens_v2 requires target_modules={sorted(expected_targets)}"
            )
        if self.modules_to_save != ["embed_tokens"]:
            raise ValueError("chat_tokens_v2 requires modules_to_save=['embed_tokens']")
        return self


class SFTConfig(StrictModel):
    schema_version: Literal["touchagent.sft_config.v2"] = "touchagent.sft_config.v2"
    dataset_version: Literal["TouchAgent-Instruct-v2"] = "TouchAgent-Instruct-v2"
    model_name_or_path: str
    output_dir: str
    data_path: str
    data_manifest_path: str
    seed: Literal[42] = 42
    max_length: Literal[2048] = 2048
    expected_world_size: Literal[4] = 4
    per_device_train_batch_size: Literal[12] = 12
    gradient_accumulation_steps: Literal[1] = 1
    num_train_epochs: float = Field(gt=0)
    max_steps: int = Field(default=-1, ge=-1)
    learning_rate: Literal[0.00005] = 0.00005
    max_grad_norm: Literal[1.0] = 1.0
    weight_decay: Literal[0.0] = 0.0
    warmup_ratio: Literal[0.03] = 0.03
    lr_scheduler_type: Literal["cosine"] = "cosine"
    optim: Literal["adamw_torch"] = "adamw_torch"
    logging_steps: Literal[1] = 1
    save_steps: int = Field(ge=1)
    save_total_limit: int = Field(ge=1)
    bf16: Literal[True] = True
    tf32: Literal[True] = True
    gradient_checkpointing: Literal[True] = True
    dataloader_num_workers: Literal[4] = 4
    attention_implementation: Literal["flash_attention_2"] = "flash_attention_2"
    deepspeed_config_path: str
    checkpoint_selection_policy: Literal["final_adapter"] = "final_adapter"
    resume_from_checkpoint: Optional[str] = None
    lora: LoraTrainingConfig = Field(default_factory=LoraTrainingConfig)

    @field_validator(
        "model_name_or_path",
        "output_dir",
        "data_path",
        "data_manifest_path",
        "deepspeed_config_path",
    )
    @classmethod
    def validate_absolute_path(cls, value: str) -> str:
        if not Path(value).is_absolute():
            raise ValueError("Runtime paths must be absolute after config resolution")
        return value

    @field_validator("resume_from_checkpoint")
    @classmethod
    def validate_resume_path(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and not Path(value).is_absolute():
            raise ValueError("Resume checkpoint must be an absolute path")
        return value

    @model_validator(mode="after")
    def validate_steps(self) -> "SFTConfig":
        if self.max_steps == 0:
            raise ValueError("max_steps must be -1 or a positive integer")
        return self


def _resolve_config_path(value: str, base_dir: Path) -> str:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return str(path.resolve())


def load_sft_config(
    path: Path,
    *,
    model_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> SFTConfig:
    config_path = path.expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    base_dir = config_path.parent
    path_fields = (
        "model_name_or_path",
        "output_dir",
        "data_path",
        "data_manifest_path",
        "deepspeed_config_path",
    )
    for field in path_fields:
        payload[field] = _resolve_config_path(payload[field], base_dir)
    if model_path is not None:
        payload["model_name_or_path"] = str(model_path.expanduser().resolve())
    if output_dir is not None:
        payload["output_dir"] = str(output_dir.expanduser().resolve())
    return SFTConfig.model_validate(payload)
