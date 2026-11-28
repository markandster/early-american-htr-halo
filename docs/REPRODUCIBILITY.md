# Clean-Machine Reproducibility Runbook

This runbook documents the exact pinned revisions, hardware prerequisites, and command sequence to reproduce **`early-american-htr-halo`** from a clean Linux/macOS host.

> **Verification Status:** CPU unit tests, Tier-1 static YAML checks (`11/11`), and Tier-2 clean-room Halo parser validation (`11/11` against external `whitecircle/halo@53fe34adcc4ef4bef9979ffaa1d50b5b5d4948b2`) are verified passing. On a live 24 GB Colab Pro+ NVIDIA L4 (`driver=580.82.07`, `22.49 GiB`), 4-bit QLoRA (`bnb_4bit_quant_storage=bfloat16`, `HALO_FLEX_SLIDING=0`, `peft_bf16_autocast`) passed `VLMDataCollator` batch-0 (`proc=1092`, `batch=1092`, `eot_id=106`), 5-step `sft.py` (`1.644 → 1.194`, `train_loss=1.462`, `175.7s`, trainable vision LoRA updates), `merge_peft_adapters.py` + `verify_mlx_export.py`, natural `<turn|>` stop, Stage 1→2 `SMPOTrainer` step-0 continuity (`base→SFT shift=3.31727`, `stage2→base=3.587596`; `0.0` diff under matching FP32-norm + `autocast`), and 5-step `smpo.py` (`2.242 → 2.503`, `train_loss=2.359`, `320.7s`). On the standalone Colab pilot notebook (`google/gemma-4-12B-it`, 24 stratified `val` pages), `baseline_zeroshot` scored `strict_wer_pct=159.99`, `strict_cer_pct=151.79`, `lenient_cer_pct=103.03`, `raw_strict_cer_pct=149.43`, `loop_pages=7`, `budget_capped_pages=8`, `thinking_leak_pages=0`, `mean_verifiable_reward=-0.1715`; after applying completion-only `logits_to_keep=keep_mask` + bf16 embedding cast, the live 24 GB L4 `ARM=real` QLoRA smoke (`main@89af415`) trained `124/124` pages with `0` OOMs (`10m55s`, `Epoch 1 loss=1.4739`, `1,066` realized image tokens, trainable `model.embed_vision` `294,912` + LoRA `65,640,448` params) and scored `strict_wer_pct=57.35`, `strict_cer_pct=38.12`, `lenient_cer_pct=33.98`, `raw_strict_cer_pct=37.80`, `newline_cer_pct=38.69`, `line_boundary_f1_pct=14.84`, `order_tolerant_cer_pct=35.89`, `mean_verifiable_reward=0.5595`, `loop_pages=2`, `budget_capped_pages=2`, `non_terminated_pages=2`, `editorial_leaks=0`, `modernizations=0`, `thinking_leak_pages=0`, `independent_recount_verified=True`, with 16-page same-checkpoint image-swap `swap CER=117.01%` (`+79.75 pp` degradation). **An 80 GB BF16 (`>= 56 GiB`) smoke run and full-corpus (`2,897`-page) Gemma 4 `12B` Halo validation/test runs remain pending** (see [results/README.md](../results/README.md)).

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

# Matched-compute Halo sham control (deranged train transcripts; val remains unpermuted):
python scripts/export_halo_datasets.py \
  --mode sft --sham \
  --train-jsonl data/revcity/manifests/train.jsonl \
  --val-jsonl data/revcity/manifests/val.jsonl \
  --out-dir data/halo_vlm_sft_sham \
  --tokenizer checkpoints/gemma4-12b-highres-1120 \
  --max-length 4096 --vision-tokens 1120

# Optional standalone eval-only LOC By the People genre-shift panel (never mixed into training):
python scripts/build_loc_challenge.py \
  --out-dir data/loc_btp --max-pages 150 \
  --revcity-manifests-dir data/revcity/manifests
```

### Step D — Stage 1 Gemma 4 Halo SFT & Parity Export (`12b`, `26b_a4b`, `31b`)

First run pre-flight pod readiness verification (GPU VRAM/driver, free NVMe disk, Python versions, Tier-2 Halo config check, and staged Arrow datasets/manifests):

```bash
python scripts/preflight_pod.py \
  --work-dir . --halo-repo /tmp/halo \
  --dataset-dir data/halo_vlm_sft \
  --dataset-dir data/halo_vlm_sft_sham \
  --manifest data/revcity/manifests/val.jsonl \
  --manifest data/revcity/manifests/test_public.jsonl
```

`scripts/run_gemma4_halo.sh` writes each run to a fresh `RUN_ID`-stamped directory (refusing stale non-empty outputs unless `RESUME=1`, which sets `resume_from_checkpoint: true`), validates the resolved YAML in an isolated folder, enforces `EP` / `NUM_GPUS` parity, and either merges PEFT adapters (`12b`, `31b`) or verifies `RESOLVE_CKPT` parity and converts `fp32_experts` full checkpoints via Halo's `convert_to_bf16.py --verify` (`26b_a4b`, budget `~400 GB` disk because checkpoint rotation briefly holds two `97 GB` checkpoints):

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

### Step D.2 — Conditional `Inkling-2-Small` Export, Training (`EP=8`), and `vLLM` Serving

`thinkingmachines/Inkling-2-Small` (`rev 885d78bfca61a7e9a498e33cfa7ad35850533b8c`) requires streaming PNG/RGB export resized to exact `40x40` grid-cell multiples within `--max-image-tokens 2000` (`LANCZOS` resampling; export metadata records the active Pillow version and filter — Halo's container locks `Pillow==12.3.0`), with `--measure-vision-tokens` using `proc.image_token_id`. Training defaults to `EP=8` (`~73–85 GiB/GPU`), and serving packed weights requires `~343 GB` across GPUs on `halo:vllm-0.1.dev25249` with `--chat-template configs/halo/templates/inkling2_htr_nothink.jinja --enable-lora` (load the exported engine adapter via `/v1/load_lora_adapter` before evaluation):

```bash
python scripts/export_halo_datasets.py \
  --mode sft \
  --train-jsonl data/revcity/manifests/train.jsonl \
  --val-jsonl data/revcity/manifests/val.jsonl \
  --out-dir data/halo_inkling2_vlm_sft \
  --tokenizer thinkingmachines/Inkling-2-Small \
  --revision 885d78bfca61a7e9a498e33cfa7ad35850533b8c \
  --max-length 4096 --max-image-tokens 2000 --cell-size 40 \
  --measure-vision-tokens

STAGE=sft DATASET_DIR=data/halo_inkling2_vlm_sft HALO_DIR=/tmp/halo NUM_GPUS=8 PILOT_MODE=1 \
  bash scripts/run_inkling2_halo.sh

# On serving node (halo:vllm-0.1.dev25249), launch with:
#   --chat-template configs/halo/templates/inkling2_htr_nothink.jinja --enable-lora
# then register the exported engine adapter via POST /v1/load_lora_adapter and evaluate:
python scripts/eval_vllm_server.py \
  --base-url http://127.0.0.1:8000 --model inkling2-sft \
  --eval-jsonl data/revcity/manifests/val.jsonl \
  --engine-adapter-dir checkpoints/inkling2-small-eahtr-sft-engine-<RUN_ID> \
  --tokenizer thinkingmachines/Inkling-2-Small \
  --revision 885d78bfca61a7e9a498e33cfa7ad35850533b8c \
  --max-image-tokens 2000 --out-json results/val_inkling2_sft.json --dual-convention
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

# 4. Per-page diagnostic error analysis (word-aligned line-break drift, loop/cap buckets, two-page openings)
python scripts/analyze_eval_predictions.py \
  --predictions results/val_sft.json \
  --baseline results/val_zeroshot.json \
  --manifest manifests/val.jsonl
```

### Step F — Conditional Stage 2 VLM SMPO (Out-of-Fold Rollouts Only)

Run Stage 2 only if Stage 1 validation failure rate justifies preference optimization (see [docs/TRAINING_PLAN.md](TRAINING_PLAN.md)). Because `generate_halo_rollouts.py` verifies `split == "train"` on `--train-jsonl`, true **out-of-fold** rollouts require first partitioning `train.jsonl` via `scripts/build_oof_folds.py`, training each fold-held-out Stage 1 model on `oof_fold{k}_train.jsonl`, and passing that fold's held-out checkpoint against `oof_fold{k}_holdout.jsonl` rather than sampling from a Stage 1 model trained on those pages. To monitor SMPO loss cleanly without leaking into training, build `data/pref_val.jsonl` only from `manifests/val.jsonl` with `--allow-eval-split` (never passed to `--train-jsonl`):

```bash
python scripts/build_oof_folds.py \
  --train-jsonl data/revcity/manifests/train.jsonl \
  --out-dir data/revcity/manifests/oof --num-folds 4

python scripts/generate_halo_rollouts.py \
  --model-dir checkpoints/gemma4-12b-eahtr-sft-oof0-merged \
  --train-jsonl data/revcity/manifests/oof/oof_fold0_holdout.jsonl \
  --out-pref-jsonl data/pref_fold0.jsonl \
  --out-rollouts-jsonl data/raw_rollouts_fold0.jsonl \
  --source-fold fold0

# After generating pref_fold{0..3}.jsonl across all 4 held-out folds, merge and leakage-audit against val/test:
python scripts/merge_oof_preferences.py \
  --inputs data/pref_fold0.jsonl data/pref_fold1.jsonl data/pref_fold2.jsonl data/pref_fold3.jsonl \
  --out-jsonl data/pref_train.jsonl \
  --eval-manifests manifests/val.jsonl manifests/test_public.jsonl \
  --out-summary data/pref_train_audit.json

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
  SFT_MERGED_DIR=checkpoints/gemma4-12b-eahtr-sft-merged-<RUN_ID> \
  DATASET_DIR=data/halo_vlm_smpo \
  HALO_DIR=/tmp/halo NUM_GPUS=1 \
  bash scripts/run_gemma4_halo.sh

# Fallback Stage 2 SMPO examples:
#   26B-A4B MoE (EP=8 on 8x GPUs):
#     STAGE=smpo MODEL_SIZE=26b_a4b EP_SIZE=8 SFT_MERGED_DIR=checkpoints/gemma4-26b_a4b-eahtr-sft-merged-<RUN_ID> NUM_GPUS=8 bash scripts/run_gemma4_halo.sh
#   31B Dense LoRA (2x–8x GPU FSDP, no EP):
#     STAGE=smpo MODEL_SIZE=31b SFT_MERGED_DIR=checkpoints/gemma4-31b-eahtr-sft-merged-<RUN_ID> NUM_GPUS=8 bash scripts/run_gemma4_halo.sh

# For PEFT runs (12b, 31b), merge the winning PEFT adapter into a full HuggingFace directory
# (full-weight 26b_a4b runs convert checkpoint-N via Halo's convert_to_bf16.py --verify inside run_gemma4_halo.sh):
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

# Optional local 8-bit MLX conversion + 1120-token / pixel / template parity verification
# (verified against mlx 0.32.5 + mlx-vlm 0.8.1 native `gemma4_unified` support):
python -m mlx_vlm convert --hf-path checkpoints/gemma4-12b-eahtr-final-merged --mlx-path mlx_out -q --q-bits 8
python scripts/verify_mlx_export.py mlx_out --reference-dir checkpoints/gemma4-12b-eahtr-final-merged
```
