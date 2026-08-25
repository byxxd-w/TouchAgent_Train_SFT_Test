"""Transformers Trainer and PEFT LoRA construction for TouchAgent SFT."""

from __future__ import annotations

import json
import os
import platform
import re
from pathlib import Path
from typing import Any, Dict

from .config import SFTConfig
from .data import AssistantOnlyDataCollator, TouchAgentSFTDataset, load_instruct_file
from .manifest import atomic_write_json, verify_data_manifest
from .metrics import build_jsonl_metrics_callback


def load_tokenizer(config: SFTConfig):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Tokenizer loading requires Transformers") from exc
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name_or_path,
        local_files_only=True,
        trust_remote_code=False,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer must define eos_token_id or pad_token_id")
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.chat_template is None:
        raise ValueError("Qwen tokenizer does not define a chat template")
    return tokenizer


def audit_training_data(config: SFTConfig) -> Dict[str, Any]:
    manifest = verify_data_manifest(config)
    tokenizer = load_tokenizer(config)
    records = load_instruct_file(Path(config.data_path))
    dataset = TouchAgentSFTDataset(tokenizer, records, config.max_length)
    lengths = [len(item["input_ids"]) for item in dataset.examples]
    supervised = [
        sum(value != -100 for value in item["labels"]) for item in dataset.examples
    ]
    assistant_supervisions = sum(
        len(item["assistant_spans"]) for item in dataset.examples
    )
    im_end_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    supervised_im_end_tokens = sum(
        item["input_ids"][end - 1] == im_end_token_id
        and item["labels"][end - 1] == im_end_token_id
        for item in dataset.examples
        for _, end in item["assistant_spans"]
    )
    if supervised_im_end_tokens != assistant_supervisions:
        raise ValueError("Not every Assistant span supervises its <|im_end|> token")
    payload = {
        "schema_version": "touchagent.sft_audit.v2",
        "dataset_version": config.dataset_version,
        "records": len(dataset),
        "assistant_supervisions": assistant_supervisions,
        "supervised_im_end_tokens": supervised_im_end_tokens,
        "im_end_token_id": im_end_token_id,
        "source_counts": manifest["selected_source_counts"],
        "selected_id_order_sha256": manifest["selected_id_order_sha256"],
        "min_tokens": min(lengths),
        "max_tokens": max(lengths),
        "min_supervised_tokens": min(supervised),
        "max_supervised_tokens": max(supervised),
        "max_length": config.max_length,
        "truncated_records": 0,
        "tokenizer": "Qwen/Qwen2.5-7B@d149729398750b98c0af14eb82c78cfe92750796",
        "data_file": Path(config.data_path).name,
        "data_manifest": Path(config.data_manifest_path).name,
    }
    contract = manifest.get("tokenizer_audit_contract", {})
    expected = {
        "records": manifest["record_count"],
        "assistant_supervisions": manifest["assistant_supervision_count"],
        "supervised_im_end_tokens": contract.get(
            "expected_supervised_im_end_tokens"
        ),
        "min_tokens": contract.get("expected_min_tokens"),
        "max_tokens": contract.get("expected_max_tokens"),
        "min_supervised_tokens": contract.get("expected_min_supervised_tokens"),
        "max_supervised_tokens": contract.get("expected_max_supervised_tokens"),
        "truncated_records": contract.get("expected_truncated_records"),
    }
    mismatches = {
        key: {"expected": expected_value, "actual": payload[key]}
        for key, expected_value in expected.items()
        if expected_value is None or payload[key] != expected_value
    }
    if mismatches:
        raise ValueError(
            "Tokenizer audit does not match the frozen contract: "
            + json.dumps(mismatches, ensure_ascii=False, sort_keys=True)
        )
    return payload


def audit_training_runtime(config: SFTConfig) -> Dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Runtime preflight requires PyTorch") from exc

    model_path = Path(config.model_name_or_path)
    if not model_path.is_dir():
        raise RuntimeError(f"Local model directory does not exist: {model_path}")
    required_model_files = ("config.json", "model.safetensors.index.json", "tokenizer.json")
    missing_model_files = [name for name in required_model_files if not (model_path / name).is_file()]
    if missing_model_files:
        raise RuntimeError(f"Local model directory is incomplete: {missing_model_files}")
    verify_data_manifest(config)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the training environment")
    visible_gpu_count = torch.cuda.device_count()
    if visible_gpu_count != config.expected_world_size:
        raise RuntimeError(
            "Visible GPU count does not match config: "
            f"expected={config.expected_world_size}, actual={visible_gpu_count}"
        )
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("The visible GPUs do not report BF16 support")

    try:
        import flash_attn
    except ImportError as exc:
        raise RuntimeError("FlashAttention 2 is required but is not importable") from exc

    deepspeed_path = Path(config.deepspeed_config_path)
    if not deepspeed_path.is_file():
        raise RuntimeError(f"DeepSpeed config does not exist: {deepspeed_path}")
    with deepspeed_path.open("r", encoding="utf-8") as handle:
        deepspeed_payload = json.load(handle)
    zero_stage = deepspeed_payload.get("zero_optimization", {}).get("stage")
    if zero_stage != 2:
        raise RuntimeError(f"TouchAgent SFT requires DeepSpeed ZeRO stage 2, got {zero_stage}")
    try:
        import deepspeed
    except ImportError as exc:
        raise RuntimeError("DeepSpeed is required but is not importable") from exc

    kernel_release = platform.release()
    match = re.match(r"^(\d+)\.(\d+)", kernel_release)
    kernel_below_recommended = bool(
        match and (int(match.group(1)), int(match.group(2))) < (5, 5)
    )
    warnings = []
    if kernel_below_recommended:
        warnings.append("Linux kernel is below 5.5; complete the smoke run before formal training")
    return {
        "schema_version": "touchagent.sft_runtime_preflight.v2",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": True,
        "visible_gpu_count": visible_gpu_count,
        "visible_gpu_names": [torch.cuda.get_device_name(index) for index in range(visible_gpu_count)],
        "bf16_supported": True,
        "attention_implementation": config.attention_implementation,
        "flash_attention": flash_attn.__version__,
        "deepspeed": deepspeed.__version__,
        "deepspeed_zero_stage": zero_stage,
        "kernel_release": kernel_release,
        "kernel_below_recommended": kernel_below_recommended,
        "model_path": config.model_name_or_path,
        "warnings": warnings,
    }


def train_lora(config: SFTConfig) -> None:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != config.expected_world_size:
        raise ValueError(
            f"Training world size must be {config.expected_world_size}, got {world_size}"
        )
    if config.resume_from_checkpoint is not None and not Path(
        config.resume_from_checkpoint
    ).is_dir():
        raise ValueError(f"Resume checkpoint does not exist: {config.resume_from_checkpoint}")
    output_path = Path(config.output_dir)
    if (
        config.resume_from_checkpoint is None
        and output_path.is_dir()
        and any(output_path.iterdir())
    ):
        raise ValueError(f"Training output directory is not empty: {output_path}")
    runtime = audit_training_runtime(config)
    if int(os.environ.get("RANK", "0")) == 0:
        print("Training runtime: " + json.dumps(runtime, ensure_ascii=False))
    global_batch_size = (
        config.per_device_train_batch_size
        * config.gradient_accumulation_steps
        * world_size
    )
    print(
        "Training batch: "
        f"per_device={config.per_device_train_batch_size}, "
        f"gradient_accumulation={config.gradient_accumulation_steps}, "
        f"world_size={world_size}, global={global_batch_size}"
    )

    try:
        import torch
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import AutoModelForCausalLM, Trainer, TrainingArguments
    except ImportError as exc:
        raise RuntimeError(
            "Training requires Transformers, PEFT, Accelerate, and PyTorch"
        ) from exc

    tokenizer = load_tokenizer(config)
    records = load_instruct_file(Path(config.data_path))
    dataset = TouchAgentSFTDataset(tokenizer, records, config.max_length)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name_or_path,
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.bfloat16,
        attn_implementation=config.attention_implementation,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.lora.r,
            lora_alpha=config.lora.alpha,
            lora_dropout=config.lora.dropout,
            target_modules=config.lora.target_modules,
            modules_to_save=config.lora.modules_to_save,
            bias="none",
        ),
    )
    model.enable_input_require_grads()
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    print(
        "LoRA parameters: "
        f"trainable={trainable_parameters:,}, total={total_parameters:,}, "
        f"ratio={100.0 * trainable_parameters / total_parameters:.4f}%"
    )
    arguments = TrainingArguments(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.num_train_epochs,
        max_steps=config.max_steps,
        learning_rate=config.learning_rate,
        max_grad_norm=config.max_grad_norm,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type=config.lr_scheduler_type,
        optim=config.optim,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        save_strategy="steps",
        evaluation_strategy="no",
        bf16=config.bf16,
        tf32=config.tf32,
        gradient_checkpointing=config.gradient_checkpointing,
        dataloader_num_workers=config.dataloader_num_workers,
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        report_to=[],
        deepspeed=config.deepspeed_config_path,
        seed=config.seed,
        data_seed=config.seed,
    )
    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=dataset,
        data_collator=AssistantOnlyDataCollator(tokenizer.pad_token_id),
        tokenizer=tokenizer,
        callbacks=[build_jsonl_metrics_callback(config.output_dir)],
    )
    train_result = trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)
    trainer.save_state()
    trainer.save_model(config.output_dir)
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(config.output_dir)
        atomic_write_json(
            Path(config.output_dir) / "training_run_summary.json",
            {
                "schema_version": "touchagent.sft_run_summary.v2",
                "completed": True,
                "final_global_step": trainer.state.global_step,
                "checkpoint_selection_policy": config.checkpoint_selection_policy,
                "selected_artifact": config.output_dir,
                "best_model_checkpoint": trainer.state.best_model_checkpoint,
                "evaluation_strategy": "no",
                "resume_from_checkpoint": config.resume_from_checkpoint,
                "attention_implementation": config.attention_implementation,
                "deepspeed_config_path": config.deepspeed_config_path,
                "lora_profile": config.lora.profile,
                "lora_target_modules": config.lora.target_modules,
                "lora_modules_to_save": config.lora.modules_to_save,
                "effective_global_batch_size": global_batch_size,
                "configured_max_steps": config.max_steps,
                "train_metrics": train_result.metrics,
            },
        )
