# Clean-Machine Reproducibility Runbook

This runbook documents the exact pinned revisions, hardware prerequisites, and command sequence to reproduce **`early-american-htr-halo`** from a clean Linux/macOS host.

> **Verification Status:** CPU unit tests (`46 passed`), Tier-1 static YAML checks (`7/7`), and Tier-2 clean-room Halo parser validation (`7/7` against external `whitecircle/halo@53fe34adcc4ef4bef9979ffaa1d50b5b5d4948b2`) are verified passing. On a live 24 GB Colab Pro+ NVIDIA L4 (`driver=580.82.07`, `22.49 GiB`), 4-bit QLoRA (`bnb_4bit_quant_storage=bfloat16`, `HALO_FLEX_SLIDING=0`, `peft_bf16_autocast`) passed `VLMDataCollator` batch-0 (`proc=1092`, `batch=1092`, `eot_id=106`), 5-step `sft.py` (`1.644 → 1.194`, `train_loss=1.462`, `175.7s`, trainable vision LoRA updates), `merge_peft_adapters.py` + `verify_mlx_export.py`, natural `<turn|>` stop, Stage 1→2 `SMPOTrainer` step-0 continuity (`base→SFT shift=3.31727`, `stage2→base=3.587596`; `0.0` diff under matching FP32-norm + `autocast`), and 5-step `smpo.py` (`2.242 → 2.503`, `train_loss=2.359`, `320.7s`). **An 80 GB BF16 (`>= 56 GiB`) smoke run is still required before Dec 1**, and full-corpus Gemma 4 `12B` validation accuracy remains unmeasured (see [results/README.md](../results/README.md)).

---

## 1. Pinned Software, Model & Corpus Versions

| Component | Pinned Version / Revision | License | Notes |
|---|---|---|---|
| **Repository** | `early-american-htr-halo` (`main`) | Apache-2.0 | Never vendors external Halo code |
| **External Trainer** | `whitecircle/halo` (`53fe34adcc4ef4bef9979ffaa1d50b5b5d4948b2` / `53fe34a`) | Halo License | Cloned externally to `--halo-repo` (`/tmp/halo` or `/workspace/halo`); no `v1.3.0rc2` git tag exists (`hopper-1.3.0rc2` / `blackwell-1.3.0rc2` refer to ECR images) |
| **Optimizer Dependency** | `Dao-AILab/gram-newton-schulz@244a451d684a39d68f593aef03b6fe7cfa6a31d1` | Open Source | Installed `--no-deps` by `run_halo_smoke.sh` |
| **Primary Base Model** | `google/gemma-4-12B-it` (`rev 43c9d014b231`) | Apache-2.0 | Ungated; staged to `1,120` soft tokens before training |
| **Fallback Base Models** | `google/gemma-4-26B-A4B-it` (`rev 4d7ae4984b7db7de8f8457170b3f1a419ee76d52`, MoE `EP=8`), `google/gemma-4-31B-it` (`rev 842da3794eaa0b77d5f08bae87a17459d91ff475`, Dense) | Apache-2.0 | Ungated; staged to `1,120` soft tokens via `run_gemma4_halo.sh` |
| **Conditional Pilot Model** | `thinkingmachines/Inkling-2-Small` (`rev 885d78bfca61a7e9a498e33cfa7ad35850533b8c`) | Apache-2.0 | Ungated; requires `EP=8` Dec 1 serving/memory pilot |
| **Training Corpus** | Revolutionary City HTR (`Zenodo 15776323`) | `CC-BY-4.0` | `9,988,850,095` bytes; MD5 `c291f2ebda6e5698f788a06b9520c709` |
| **Eval-Only Challenge Panel** | LOC *By the People* (`LCCN 2020446971`) | `Public Domain` | Off by default; never mixed into training splits |
| **Core Python Stack** | Python `3.12`, `torch==2.11.0`, `torchvision==0.26.0`, `transformers==5.23.0`, `trl==1.14.2`, `peft==0.18.1`, `accelerate>=1.11.0,<1.12.0`, `bitsandbytes>=0.49.2,<0.50.0` | — | Halo requires `>=3.12,<3.13` |

---

## 2. Hardware & Container Prerequisites

- **CPU Validation Host (macOS or Linux):** Python `3.12`, no GPU required for `pytest` or Tier-1/Tier-2 config validation.
- **1× GPU Smoke Host (`scripts/run_halo_smoke.sh`):**
  - NVIDIA driver **`>= 560.28`** (`select_pytorch_cuda_index` picks `cu126` for `>= 560.28`, `cu128` for `>= 570.26`, `cu130` for `>= 580.65`, and fails closed below `560.28`).
  - VRAM adaptive thresholds: `< 56 GiB` runs 4-bit QLoRA (`load_in_4bit: true`, `bnb_4bit_quant_storage: bfloat16` required so Gemma 4 `patch_ln1` does not fail with Float/BFloat16 mismatch during SFT; `< 32 GiB` such as a 24 GB L4 also sets `HALO_FLEX_SLIDING=0` and SMPO smoke `max_length: 2048` / `max_prompt_length: 1350`); `>= 56 GiB` (80 GB H100/A100) runs full BF16 at `max_length: 4096`.
  - Disk: `~120 GB` free for Gemma 4 weights, virtual environment, working cache, and merged checkpoint.
  - Persistent-volume overrides (recommended on RunPod/cloud pods whose container `/` is small):
    ```bash
    export HF_HOME=/workspace/.cache/huggingface
    export UV_CACHE_DIR=/workspace/.cache/uv
    export HALO_DIR=/workspace/halo
    export SMOKE_VENV=/workspace/halo-gpu-smoke-venv
    export WORK_DIR=/workspace/eahtr_halo_smoke
    ```
- **Multi-GPU Competition Containers:** `public.ecr.aws/whitecircle/halo:hopper-1.3.0rc2` or `:blackwell-1.3.0rc2` (AWS, Modal, RunPod, or SkyPilot).

---

## 3. Clean-Machine Command Sequence

### Step A — Local Unit Tests & Two-Tier Config Validation (CPU)

```bash
git clone https://github.com/markandster/early-american-htr-halo.git
cd early-american-htr-halo
pip install -e ".[dev]"

pytest -v
python scripts/validate_halo_configs.py

# Clean-room Tier-2 parser validation against external Halo checkout (Python 3.12; install Halo deps first):
git clone --filter=blob:none --no-checkout https://github.com/whitecircle/halo.git /tmp/halo
git -C /tmp/halo fetch --depth 1 origin 53fe34adcc4ef4bef9979ffaa1d50b5b5d4948b2
git -C /tmp/halo checkout 53fe34adcc4ef4bef9979ffaa1d50b5b5d4948b2
pip install "torch==2.11.0" "transformers==5.23.0" "trl==1.14.2" "peft==0.18.1" "accelerate>=1.11.0,<1.12.0" "datasets>=4.8.5,<5.0.0" "ruamel.yaml>=0.19.1,<0.20.0" "pydantic>=2.6.4,<3.0.0" "huggingface-hub>=1.0.0,<2.0.0"
pip install --no-deps "gram-newton-schulz @ git+https://github.com/Dao-AILab/gram-newton-schulz.git@244a451d684a39d68f593aef03b6fe7cfa6a31d1"
pip install --no-deps -e /tmp/halo
python scripts/validate_halo_configs.py --halo-repo /tmp/halo
```

### Step B — 1× GPU 5-Step Halo VLM Smoke (`sft.py` → merge → continuity → `smpo.py`)

```bash
bash scripts/run_halo_smoke.sh 2>&1 | tee smoke.log
# Optional overrides for faster/custom reruns (default MAX_STEPS=5, OUTPUT_DIR=$WORK_DIR):
# REUSE_VENV=1 MAX_STEPS=10 OUTPUT_DIR=/tmp/smoke_rerun bash scripts/run_halo_smoke.sh 2>&1 | tee smoke_rerun.log
```

This verifies NVIDIA driver `>= 560.28`, Halo clean-room installation, `1,120` realized image tokens on `VLMDataCollator` batch-0 (`1,092` on sampled page), live Halo PEFT vision-parameter updates, `MAX_STEPS` distinct finite SFT losses, adapter merge + `verify_mlx_export.py`, natural `<turn|>` termination on a short page, fresh-process Stage 1→2 `SMPOTrainer` teacher-forced log-prob continuity (`peft_bf16_autocast` with matching FP32-norm casting when `< 56 GiB`), and `MAX_STEPS` finite SMPO losses.

### Step C — Build Disjoint Public Manifests & Export SFT Arrow Dataset

Generate portable relative-path manifests cleanly from the command line via `scripts/build_revcity_manifests.py` (or optionally via Cell 4 of [notebooks/colab_competition_pretrain_sft.ipynb](../notebooks/colab_competition_pretrain_sft.ipynb); see [docs/DATA_SHEET.md](DATA_SHEET.md)), stage the high-resolution `1,120`-token Gemma 4 directory, and export the Arrow `DatasetDict` (mapping `val` → Halo internal `test`):

```bash
python scripts/build_revcity_manifests.py \
  --download --zip-path data/revcity/revcity-htr.zip --out-dir data/revcity

python scripts/stage_gemma4_highres.py \
  --model google/gemma-4-12B-it --revision 43c9d014b231 \
  --out-dir checkpoints/gemma4-12b-highres-1120 --max-soft-tokens 1120 \
  --chat-template configs/halo/templates/gemma4_htr_nothink.jinja

python scripts/export_halo_datasets.py \
  --mode sft \
  --train-jsonl data/revcity/manifests/train.jsonl \
  --val-jsonl data/revcity/manifests/val.jsonl \
  --out-dir data/halo_vlm_sft \
  --tokenizer checkpoints/gemma4-12b-highres-1120 \
  --max-length 4096 --vision-tokens 1120
```

### Step D — Stage 1 Gemma 4 Halo SFT & Parity Export (`12b`, `26b_a4b`, `31b`)

`scripts/run_gemma4_halo.sh` writes each run to a fresh `RUN_ID`-stamped directory (refusing stale non-empty outputs unless `RESUME=1`), validates the resolved YAML in an isolated folder, enforces `EP` / `NUM_GPUS` parity, and either merges PEFT adapters (`12b`, `31b`) or converts `fp32_experts` full checkpoints via Halo's `convert_to_bf16.py --verify` while preserving the `1,120`-token processor & template (`26b_a4b`, budget `~300 GB` disk):

```bash
# Primary: 12B Dense LoRA (1 GPU)
STAGE=sft MODEL_SIZE=12b DATASET_DIR=data/halo_vlm_sft HALO_DIR=/tmp/halo NUM_GPUS=1 \
  bash scripts/run_gemma4_halo.sh

# Fallback MoE: 26B-A4B Full SFT (EP=8, 8x 80 GB GPUs)
STAGE=sft MODEL_SIZE=26b_a4b DATASET_DIR=data/halo_vlm_sft HALO_DIR=/tmp/halo NUM_GPUS=8 \
  bash scripts/run_gemma4_halo.sh

# Fallback Dense: 31B LoRA (H200/B200 or 2x GPU FSDP)
STAGE=sft MODEL_SIZE=31b DATASET_DIR=data/halo_vlm_sft HALO_DIR=/tmp/halo NUM_GPUS=2 \
  bash scripts/run_gemma4_halo.sh
```

### Step E — Validation Eval, Image-Swap Causal Check & Repetition-Penalty Baseline

```bash
# 0. Zero-shot baseline on val (produces results/val_zeroshot.json; repeat post-freeze with --allow-frozen-test on manifests/test_public.jsonl for results/test_zeroshot.json)
python scripts/merge_and_eval_halo.py \
  --model-dir checkpoints/gemma4-12b-highres-1120 \
  --eval-jsonl manifests/val.jsonl \
  --out-json results/val_zeroshot.json \
  --dual-convention

# 1. Stage 1 validation score (dual-convention + newline/order metrics + clustered bootstrap CI)
python scripts/merge_and_eval_halo.py \
  --model-dir checkpoints/gemma4-12b-eahtr-sft-merged \
  --eval-jsonl manifests/val.jsonl \
  --baseline-predictions results/val_zeroshot.json \
  --out-json results/val_sft.json \
  --dual-convention

# 2. Same-checkpoint image-swap visual-grounding check
python scripts/merge_and_eval_halo.py \
  --model-dir checkpoints/gemma4-12b-eahtr-sft-merged \
  --eval-jsonl manifests/val.jsonl \
  --out-json results/val_sft_swap.json \
  --swap-images

# 3. Decode-time repetition-penalty baseline (benchmark Stage 2 SMPO against this + continued gold SFT)
python scripts/merge_and_eval_halo.py \
  --model-dir checkpoints/gemma4-12b-eahtr-sft-merged \
  --eval-jsonl manifests/val.jsonl \
  --baseline-predictions results/val_zeroshot.json \
  --out-json results/val_sft_reppen.json \
  --repetition-penalty 1.1
```

### Step F — Conditional Stage 2 VLM SMPO (Out-of-Fold Rollouts Only)

Run Stage 2 only if Stage 1 validation failure rate justifies preference optimization (see [docs/TRAINING_PLAN.md](TRAINING_PLAN.md)). Because `generate_halo_rollouts.py` verifies `split == "train"` on `--train-jsonl`, true **out-of-fold** rollouts require first training fold-held-out Stage 1 models (`OOF_NUM_FOLDS > 1`, e.g. excluding `manifests/oof_fold0_train.jsonl`) and passing each held-out Stage 1 checkpoint rather than sampling from a Stage 1 model trained on all of `train`. To monitor SMPO loss cleanly without leaking into training, build `data/pref_val.jsonl` only from `manifests/val.jsonl` with `--allow-eval-split` (never passed to `--train-jsonl`):

```bash
python scripts/generate_halo_rollouts.py \
  --model-dir checkpoints/gemma4-12b-eahtr-sft-oof0-merged \
  --train-jsonl manifests/oof_fold0_train.jsonl \
  --out-pref-jsonl data/pref_train.jsonl \
  --out-rollouts-jsonl data/raw_rollouts.jsonl \
  --source-fold fold0

python scripts/generate_halo_rollouts.py \
  --model-dir checkpoints/gemma4-12b-eahtr-sft-merged \
  --train-jsonl manifests/val.jsonl \
  --allow-eval-split \
  --out-pref-jsonl data/pref_val.jsonl \
  --source-fold val

python scripts/export_halo_datasets.py \
  --mode preference \
  --train-jsonl data/pref_train.jsonl \
  --val-jsonl data/pref_val.jsonl \
  --out-dir data/halo_vlm_smpo \
  --tokenizer checkpoints/gemma4-12b-highres-1120 \
  --max-length 4096 --max-prompt-length 1600 --vision-tokens 1120

STAGE=smpo MODEL_SIZE=12b \
  SFT_MERGED_DIR=checkpoints/gemma4-12b-eahtr-sft-merged \
  DATASET_DIR=data/halo_vlm_smpo \
  HALO_DIR=/tmp/halo NUM_GPUS=1 \
  bash scripts/run_gemma4_halo.sh

# Full-weight 26B-A4B or 31B SMPO example (EP=8 / EP=4 respectively on 8 GPUs):
# STAGE=smpo MODEL_SIZE=26b_a4b EP_SIZE=8 \
#   SFT_MERGED_DIR=checkpoints/gemma4-26b-a4b-sft-bf16 \
#   DATASET_DIR=data/halo_vlm_smpo \
#   HALO_DIR=/tmp/halo NUM_GPUS=8 \
#   bash scripts/run_gemma4_halo.sh

# For 12B PEFT runs, merge the Stage 1 or Stage 2 PEFT adapter into a full HuggingFace directory
# (full-weight 26B-A4B and 31B runs convert checkpoint-N via Halo's convert_to_bf16.py --verify
# inside run_gemma4_halo.sh into a fresh *_final_bf16 directory):
python /tmp/halo/scripts/after_training/merge_peft_adapters.py \
  --adapter_dir checkpoints/gemma4-12b-eahtr-final-adapter \
  --output_dir checkpoints/gemma4-12b-eahtr-final-merged
```

### Step G — Single Post-Freeze `test_public` Run & Optional Local `<= 24 GB` MLX Check

```bash
# Run ONCE only after locking final checkpoint and decoding parameters:
python scripts/merge_and_eval_halo.py \
  --model-dir checkpoints/gemma4-12b-eahtr-final-merged \
  --eval-jsonl manifests/test_public.jsonl \
  --allow-frozen-test \
  --baseline-predictions results/test_zeroshot.json \
  --out-json results/test_public_final.json \
  --dual-convention

# Optional local 8-bit MLX conversion + 1120-token / pixel / template parity verification:
python -m mlx_vlm convert --hf-path checkpoints/gemma4-12b-eahtr-final-merged --mlx-path mlx_out -q --q-bits 8
python scripts/verify_mlx_export.py mlx_out --reference-dir checkpoints/gemma4-12b-eahtr-final-merged
```
