# Clean-Machine Reproducibility Runbook

This runbook documents the exact pinned revisions, hardware prerequisites, and command sequence to reproduce **`early-american-htr-halo`** from a clean Linux/macOS host.

> **Verification Status:** CPU unit tests (`46 passed`), Tier-1 static YAML checks (`7/7`), and Tier-2 clean-room Halo parser validation (`7/7` against external `whitecircle/halo@53fe34a`) are verified passing. Live GPU runtime, peak VRAM, and Gemma 4 `12B` accuracy metrics remain pending GPU execution (see [results/README.md](../results/README.md)).

---

## 1. Pinned Software, Model & Corpus Versions

| Component | Pinned Version / Revision | License | Notes |
|---|---|---|---|
| **Repository** | `early-american-htr-halo` (`main`) | Apache-2.0 | Never vendors external Halo code |
| **External Trainer** | `whitecircle/halo` (`v1.3.0rc2`, commit `53fe34a`) | Halo License | Cloned externally to `--halo-repo` (`/tmp/halo` or `/workspace/halo`) |
| **Optimizer Dependency** | `Dao-AILab/gram-newton-schulz@244a451d684a39d68f593aef03b6fe7cfa6a31d1` | Open Source | Installed `--no-deps` by `run_halo_smoke.sh` |
| **Primary Base Model** | `google/gemma-4-12B-it` (`rev 43c9d014b231`) | Apache-2.0 | Ungated; staged to `1,120` soft tokens before training |
| **Conditional Pilot Model** | `thinkingmachines/Inkling-2-Small` (`rev 885d78bfca61`) | Apache-2.0 | Ungated; requires `EP=8` Dec 1 serving/memory pilot |
| **Training Corpus** | Revolutionary City HTR (`Zenodo 15776323`) | `CC-BY-4.0` | `9,988,850,095` bytes; MD5 `c291f2ebda6e5698f788a06b9520c709` |
| **Eval-Only Challenge Panel** | LOC *By the People* (`LCCN 2020446971`) | `Public Domain` | Off by default; never mixed into training splits |
| **Core Python Stack** | Python `3.12`, `torch==2.11.0`, `torchvision==0.26.0`, `transformers==5.23.0`, `trl==1.14.2`, `peft==0.18.1`, `accelerate>=1.11.0,<1.12.0`, `bitsandbytes>=0.49.2,<0.50.0` | — | Halo requires `>=3.12,<3.13` |

---

## 2. Hardware & Container Prerequisites

- **CPU Validation Host (macOS or Linux):** Python `3.12`, no GPU required for `pytest` or Tier-1/Tier-2 config validation.
- **1× GPU Smoke Host (`scripts/run_halo_smoke.sh`):**
  - NVIDIA driver **`>= 560.28`** (`select_pytorch_cuda_index` picks `cu126` for `>= 560.28`, `cu128` for `>= 570.26`, `cu130` for `>= 580.65`, and fails closed below `560.28`).
  - VRAM adaptive thresholds: `< 32 GiB` (e.g., 24 GB L4) runs 4-bit QLoRA with `HALO_FLEX_SLIDING=0` (and `max_length: 2048` for SMPO smoke); `>= 56 GiB` (80 GB H100/A100) runs full BF16 at `max_length: 4096`.
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

# Clean-room Tier-2 parser validation against external Halo checkout (Python 3.12):
git clone --depth 50 https://github.com/whitecircle/halo.git /tmp/halo
git -C /tmp/halo checkout 53fe34a
python scripts/validate_halo_configs.py --halo-repo /tmp/halo
```

### Step B — 1× GPU 5-Step Halo VLM Smoke (`sft.py` → merge → continuity → `smpo.py`)

```bash
bash scripts/run_halo_smoke.sh 2>&1 | tee smoke.log
```

This verifies NVIDIA driver `>= 560.28`, Halo clean-room installation, `1,120` realized image tokens on `VLMDataCollator` batch-0, live Halo PEFT vision-parameter updates, `5` distinct finite SFT losses, adapter merge + `verify_mlx_export.py`, natural `<turn|>` termination on a short page, fresh-process Stage 1→2 `SMPOTrainer` teacher-forced log-prob continuity, and `5` finite SMPO losses.

### Step C — Export Disjoint Public SFT Arrow Dataset

Generate public manifests via Cell 4 of [notebooks/colab_competition_pretrain_sft.ipynb](../notebooks/colab_competition_pretrain_sft.ipynb) (see [docs/DATA_SHEET.md](DATA_SHEET.md)), stage the high-resolution `1,120`-token Gemma 4 directory, and export the Arrow `DatasetDict` (mapping `val` → Halo internal `test`):

```bash
python scripts/stage_gemma4_highres.py \
  --model google/gemma-4-12B-it --revision 43c9d014b231 \
  --out-dir checkpoints/gemma4-12b-highres-1120 --max-soft-tokens 1120 \
  --chat-template configs/halo/templates/gemma4_htr_nothink.jinja

python scripts/export_halo_datasets.py \
  --mode sft \
  --train-jsonl manifests/train.jsonl \
  --val-jsonl manifests/val.jsonl \
  --out-dir data/halo_vlm_sft \
  --tokenizer checkpoints/gemma4-12b-highres-1120 \
  --max-length 4096 --vision-tokens 1120
```

### Step D — Stage 1 Gemma 4 `12B` Halo SFT & Parity Merge

```bash
STAGE=sft MODEL_SIZE=12b DATASET_DIR=data/halo_vlm_sft HALO_DIR=/tmp/halo NUM_GPUS=1 \
  bash scripts/run_gemma4_halo.sh
```

### Step E — Validation Eval, Image-Swap Causal Check & Repetition-Penalty Baseline

```bash
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

Run Stage 2 only if Stage 1 validation failure rate justifies preference optimization (see [docs/TRAINING_PLAN.md](TRAINING_PLAN.md)):

```bash
python scripts/generate_halo_rollouts.py \
  --model-dir checkpoints/gemma4-12b-eahtr-sft-merged \
  --train-jsonl manifests/oof_fold0_train.jsonl \
  --out-pref-jsonl data/pref_train.jsonl \
  --out-rollouts-jsonl data/raw_rollouts.jsonl \
  --source-fold fold0

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
