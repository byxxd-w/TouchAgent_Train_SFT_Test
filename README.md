# TouchAgent Main-Agent SFT

This repository contains the reproducible supervised fine-tuning pipeline for the
TouchAgent main Agent. It trains a LoRA adapter on the frozen 32,000-record
TouchAgent-Instruct-v2 SFT split.

The repository is intentionally limited to SFT. It does not contain reinforcement
learning code, SLIME, tool workers, inference orchestration, private data, model
weights, historical checkpoints, or generated adapters. Tool observations are
already frozen as text in the training conversations; training never invokes a
real tool or reads tactile media.

## Tested configuration

| Component | Tested value |
|---|---|
| Operating system | Linux x86_64 |
| Python | 3.10.20 |
| GPUs | 4 x NVIDIA A800 80GB |
| NVIDIA driver | 535.129.03 (reference only, not an exact requirement) |
| PyTorch | 2.1.2+cu121 |
| Transformers | 4.42.3 |
| PEFT | 0.12.0 |
| DeepSpeed | 0.13.5, ZeRO stage 2 |
| FlashAttention | 2.5.9.post1 |
| Precision | BF16, TF32 enabled |
| Base model | Qwen/Qwen2.5-7B |

Use four NVIDIA GPUs with at least 80GB of memory each. The existing host driver
must support CUDA 12.x workloads; do not upgrade, downgrade, or reinstall the
NVIDIA driver just for this repository. CUDA 12.x requires driver 525.60.13 or
newer on Linux, while a newer driver can run applications built with an older
CUDA 12.x runtime through NVIDIA backward compatibility. See NVIDIA's
[CUDA compatibility documentation](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)
for the authoritative driver table. The minimum version is a CUDA runtime floor,
not a guarantee that every GPU model supports that driver; keep the driver
recommended for the host GPU by its administrator or vendor. The `CUDA Version`
shown by `nvidia-smi` is the maximum CUDA version supported by the driver, not
the CUDA runtime bundled with PyTorch.

The launcher intentionally exposes exactly four selected GPUs, even when the
machine has more. Reserve at least 50GB of free disk space for the base model,
dependency caches, checkpoints, and final adapter.

## Repository layout

```text
configs/             Formal, smoke, DeepSpeed, and base-model checksum contracts
data/                Frozen 32k SFT data and its portable manifest
reports/             Tokenizer audit and smoke validation evidence
scripts/             Four-GPU launcher and optional loss-curve renderer
tests/               SFT-only unit and launcher tests
touchagent_train/    Data validation, serialization, LoRA Trainer, and CLI
```

## Prepare the Python environment

A newly created Conda environment is not required. Prefer an existing,
dedicated Conda environment if it uses Python 3.10 and can accept the locked
package versions in `requirements.txt`. Do not install these packages into a
shared production environment whose existing dependencies must remain
unchanged.

```bash
conda activate <existing-environment>
python --version
python -m pip --version
nvidia-smi
```

The Python version must be 3.10.x. The NVIDIA driver is managed by the host and
must not be changed by this setup. Install the locked Python dependencies into
the activated environment:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

`requirements.txt` is the locked non-FlashAttention environment validated by the
original formal run. If no suitable existing environment is available, creating
a Python 3.10 Conda environment is a fallback, not a requirement:

```bash
conda create -n touchagent-train-sft python=3.10 -y
conda activate touchagent-train-sft
```

## Install FlashAttention

FlashAttention is installed separately because its binary compatibility is
platform-specific. Version 2.5.9.post1 is the validated reference for this
repository and PyTorch 2.1.2; do not replace it with the latest release without
revalidating the full stack.

Prefer a prebuilt wheel matching all of the following:

- CPython 3.10;
- PyTorch 2.1;
- the CUDA runtime reported by `torch.version.cuda` (12.1 in the reference
  environment);
- Linux x86_64;
- the PyTorch C++ ABI setting;
- the target GPU architecture.

The wheel does not need to match the `CUDA Version` text from `nvidia-smi`
exactly. It must match the installed PyTorch stack, and the unchanged host driver
must be new enough to run that stack. Install a compatible local wheel with:

```bash
python -m pip install /path/to/compatible/flash_attn-2.5.9.post1*.whl
```

If no compatible wheel is available, a source installation is the fallback:

```bash
python -m pip install flash-attn==2.5.9.post1 --no-build-isolation
```

Building from source requires a compatible CUDA toolkit with `nvcc`, a compiler,
and sufficient RAM. Installing that user-space toolkit is separate from the
NVIDIA driver and must not trigger a driver replacement.

Verify the driver/PyTorch boundary and important imports before continuing:

```bash
python - <<'PY'
import accelerate, deepspeed, flash_attn, peft, torch, transformers
print("torch:", torch.__version__)
print("PyTorch CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")
print("GPU capability:", torch.cuda.get_device_capability(0) if torch.cuda.is_available() else "N/A")
print("PyTorch C++11 ABI:", torch._C._GLIBCXX_USE_CXX11_ABI)
print("transformers:", transformers.__version__)
print("peft:", peft.__version__)
print("accelerate:", accelerate.__version__)
print("deepspeed:", deepspeed.__version__)
print("flash_attn:", flash_attn.__version__)
PY
```

## Prepare the base model

The launcher never downloads a model automatically. For an online machine,
download the pinned public Qwen base-model revision:

```bash
mkdir -p models/Qwen2.5-7B
hf download Qwen/Qwen2.5-7B \
  --revision d149729398750b98c0af14eb82c78cfe92750796 \
  --local-dir models/Qwen2.5-7B
```

Verify every model file against the checkpoint used by the original run:

```bash
(cd models/Qwen2.5-7B && sha256sum -c ../../configs/qwen2_5_7b_sha256.txt)
```

On an offline machine, obtain the same model revision separately, copy it to any
local directory, and verify it with the same checksum file. Also preinstall the
locked Python packages and a compatible FlashAttention wheel from local sources.
Set `MODEL_PATH` to the copied model directory when running commands. This
repository does not ship an offline model or dependency bundle.

## Frozen SFT dataset

`data/touchagent_sft32k_instruct_v2.json.gz` contains the exact formal SFT ID
order selected from the original 40,000 records. The excluded 8,000 RL records
and their IDs are not included. The loader reads gzip directly and does not
create an extracted copy.

| Source | SFT records |
|---|---:|
| Attribute | 7,782 |
| Matching | 7,782 |
| Interaction | 7,786 |
| Scene | 7,680 |
| Dynamic | 970 |
| Total | 32,000 |

Each record has exactly `schema_version`, `id`, and `conversations`. Conversations
strictly alternate `human` and `gpt`. Assistant turns contain `thoughts`, ordered
`actions`, and `value`; intervening Human turns contain frozen textual tool
observations. Dynamic records have 6 messages, Attribute and Matching have 8,
and Interaction and Scene have 10.

Only serialized Assistant JSON and each Assistant `<|im_end|>` token contribute
to causal-LM loss. System text, user questions, frozen observations, role
boundaries, and padding use label `-100`. Records longer than 2,048 tokens are
rejected instead of truncated.

## Validate before training

Run SFT-only unit tests and verify the compressed dataset without loading the
model:

```bash
python -m unittest discover -s tests -v
python -m touchagent_train.cli verify-manifest \
  --config configs/train_sft32k_instruct_v2.json
```

Run the full 32k tokenizer and loss-mask audit. This is CPU-heavy and may take
several minutes:

```bash
MODEL_PATH=/path/to/Qwen2.5-7B
python -m touchagent_train.cli audit \
  --config configs/train_sft32k_instruct_v2.json \
  --model-path "$MODEL_PATH" \
  --output reports/tokenizer_audit.local.json
```

The expected result is 32,000 records, 142,496 supervised Assistant spans,
142,496 supervised `<|im_end|>` tokens, token lengths 424-948, supervised token
lengths 227-496, and zero truncation.

Finally, expose exactly four GPUs and run the runtime preflight:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m touchagent_train.cli preflight \
  --config configs/train_sft32k_instruct_v2_smoke.json \
  --model-path "$MODEL_PATH"
```

## Run training

Always run the three-step smoke before formal training. Select any four suitable
GPU IDs on a larger server:

```bash
RUN_MODE=smoke \
CUDA_DEVICES=0,1,2,3 \
MODEL_PATH=/path/to/Qwen2.5-7B \
bash scripts/run_sft32k_instruct_v2.sh
```

After confirming finite loss, normal GPU memory use, and a saved adapter, start
the five-epoch formal run:

```bash
RUN_MODE=formal \
CUDA_DEVICES=0,1,2,3 \
MODEL_PATH=/path/to/Qwen2.5-7B \
bash scripts/run_sft32k_instruct_v2.sh
```

The tested formal configuration uses per-device batch size 12, gradient
accumulation 1, global batch size 48, LoRA rank 128, alpha 256, learning rate
`5e-5`, cosine scheduling, and seed 42. It runs for five epochs, approximately
3,335 optimizer steps.

To write outside the repository, set `OUTPUT_DIR` and optionally `LOG_FILE`.
Fresh training refuses a non-empty output directory. Resume is explicit and is
the only mode allowed to reuse an existing output directory:

```bash
RUN_MODE=formal \
CUDA_DEVICES=0,1,2,3 \
MODEL_PATH=/path/to/Qwen2.5-7B \
OUTPUT_DIR=/path/to/existing-run \
RESUME_FROM_CHECKPOINT=/path/to/existing-run/checkpoint-1000 \
bash scripts/run_sft32k_instruct_v2.sh
```

The selected training artifact is the final adapter in `OUTPUT_DIR`. Important
files include `adapter_model.safetensors`, `adapter_config.json`, tokenizer files,
`trainer_state.json`, `training_metrics.jsonl`, and
`training_run_summary.json`. No validation split or validation-loop checkpoint
selection is used.

Render a loss curve after training with:

```bash
python scripts/plot_training_loss.py \
  --input /path/to/run/training_metrics.jsonl \
  --output /path/to/run/loss_curve.svg
```

## Reference result

The original formal run completed on 4 x A800 80GB with 3,335 optimizer steps in
five epochs. Its final aggregate training loss was `0.021799509403138035`. A
fresh run should reproduce the same code, model, data order, prompt, tokenizer,
hyperparameters, and training workflow. Exact bitwise equality is not guaranteed
across different GPU models, drivers, kernels, and distributed runtime versions.

## Troubleshooting

- **Visible GPU count does not match config:** set `CUDA_VISIBLE_DEVICES` or
  `CUDA_DEVICES` to exactly four distinct IDs.
- **The host driver cannot run the PyTorch CUDA runtime:** use a machine whose
  existing driver supports CUDA 12.x, or ask the machine administrator to assess
  the host. Do not replace the driver as part of this repository's setup.
- **FlashAttention is not importable:** first confirm `torch.cuda.is_available()`
  and `torch.version.cuda`, then install a wheel matching Python, PyTorch, CUDA
  runtime, ABI, and GPU architecture. If no wheel exists, install a compatible
  CUDA toolkit and rebuild with `--no-build-isolation` after PyTorch; do not
  replace the host driver.
- **Local model directory is incomplete:** download or copy the pinned revision,
  then run the SHA-256 check before retrying.
- **Frozen SFT data does not match its manifest:** restore both files from the
  same repository revision; do not edit or recompress the dataset.
- **Training output is not empty:** choose a new `OUTPUT_DIR`, or explicitly set
  `RESUME_FROM_CHECKPOINT` to a valid Trainer checkpoint.
- **CUDA out of memory:** confirm that only the selected processes use the four
  GPUs and that each GPU has at least 80GB. Changing batch size or world size is
  outside the reproduced formal configuration.
- **Kernel below 5.5 warning:** the original environment used kernel 5.4 and
  completed successfully, but a full smoke run is mandatory before formal
  training because older kernels can increase distributed hang risk.
