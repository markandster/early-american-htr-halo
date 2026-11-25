# Halo Competition Training Plan (`early-american-htr-halo`)

This document specifies the public, reproducible post-training plan for **writer-general 18th-century American diplomatic handwriting recognition** (c. 1750–1800 English round hand) using [White Circle Halo](https://github.com/whitecircle/halo).

> **Model Sequence:** Execute **Gemma 4 (`12B` and `26B-A4B`) first**, followed immediately by **Inkling 2 Mini** (and optionally `Qwen3.6-35B-A3B` if partner compute credits permit). Both primary tracks share one rights-cleared corpus, one target-hygiene pipeline, and one deterministic verifiable GRPO reward (`eahtr.rewards`).

---

## 1. Objective, Scope, and Non-Goals

- **Objective:** Train open-weight **Apache-2.0** Vision-Language Models (VLMs) that transcribe unsegmented 18th-century manuscript pages accurately on **unseen writers** while preserving original spelling, capitalization, punctuation, abbreviations, and physical line breaks (`\n`), and fit within **`<= 24 GB` peak RAM** under local MLX quantization on 36 GB Apple Silicon.
- **In Scope:** Whole-page diplomatic transcription with exact line breaks from rights-cleared (`CC-BY-4.0`, `CC0-1.0`, `Public Domain`) manuscript scans.
- **Out of Scope (Non-Goals):**
  - **Word-level bounding-box JSON emission:** Autoregressive coordinate emission increases sequence length ~3–5× per page, exacerbating token-cap non-termination and repetition loops while slowing GRPO rollouts. Spatial alignment belongs downstream via dedicated line segmentation and CTC forced alignment.
  - **Writer identification / authorship attribution:** HTR optimizes *writer-invariant* transcription; writer identification is a separate task with distinct objectives and annotation requirements and is excluded here.

---

## 2. Track & Model Sequence (Gemma 4 First, Then Inkling 2 Mini)

All candidates meet four mandatory gates: **(1) native multimodal vision**, **(2) Apache-2.0 license**, **(3) <= 24 GB local Apple Silicon MLX footprint**, and **(4) full-resolution input support**.

| Order | Model | Halo Track(s) | Architecture | Local Target (36 GB Apple Silicon MLX) | License |
|---|---|---|---|---|---|
| **1A** | `google/gemma-4-12B-it` | **Best Gemma post-train** + Modal/AWS | 12B dense **encoder-free** unified VLM (256K ctx) | `<= 12.5 GiB` (8-bit) / `~7 GiB` (4-bit) | Apache-2.0 |
| **1B** | `google/gemma-4-26B-A4B-it` | **Best Gemma post-train** + Modal/AWS | 26B total / **4B active** MoE VLM | `~14.3–15.5 GiB` (4-bit) | Apache-2.0 |
| **2** | `thinkingmachines/Inkling-2-Mini` | **Best Inkling post-train** + Modal/AWS | 26.4B total / **3.4B base (~5.4B eff. active)** looped MoE VLM | Verify 4-bit MLX `<= 24 GiB` | Apache-2.0 |
| **3** | `Qwen/Qwen3.6-35B-A3B` *(optional)* | Modal / AWS / Overall | 35B total / **3B active** hybrid-attention MoE VLM | `~20.4–21.5 GiB` (4-bit) | Apache-2.0 |

### Why This Sequence
1. **Gemma 4 (`12B` then `26B-A4B`) first:** `google/gemma-4-12B-it` is dense and encoder-free—image patches feed directly into the backbone, making it fast to validate end-to-end vision gradient flow and easy to quantize safely at 8-bit (`<= 12.5 GiB`) without 4-bit decoding instability. `google/gemma-4-26B-A4B-it` extends capacity at 4B active parameters using Halo's native Gemma 4 MoE and sliding-attention kernels.
2. **Inkling 2 Mini second:** `thinkingmachines/Inkling-2-Mini` is natively supported in Halo (`inkling_mm_model` / `EPInklingMoELayer`) with a hierarchical pixel patch encoder (up to 4096 px) and ungated Apache-2.0 weights. Prior evaluation shows Inkling 2 Mini reads clean cursive competitively but suffers from token-cap non-termination (`finish_reason == "length"`) on dense multi-paragraph pages—making it an ideal target for Stage 2 verifiable GRPO.

---

## 3. Rights-Cleared Public Corpus & Split Hygiene

All training, validation, and public test splits enforce the strict allowlist in `eahtr.corpus.verify_source_license` (`CC-BY-4.0`, `CC0-1.0`, `Public Domain`):

| Source | License | Role & Filtering Rules |
|---|---|---|
| **Revolutionary City HTR Corpus** (Zenodo `15776323`) | `CC-BY-4.0` | Primary multi-hand diplomatic corpus (ALTO-XML parsed via `eahtr.corpus.parse_alto_xml_lines`). Split deterministically by archival series (`assign_deterministic_split`) into `train`, `val`, and `test_public`. |
| **Library of Congress *By the People*** (`labs.loc.gov/data/`) | `Public Domain` | Secondary public-domain manuscript expansion. Exclude tabular leader-dot pages (`has_table_leader_dots`) that induce token-limit repetition loops. |
| **Additional Audited 18th-c English Sets** | `CC-BY-4.0` / `CC0-1.0` only | Only admitted after explicit per-dataset license verification (non-commercial `CC-BY-NC` datasets are rejected automatically). |

### Mandatory Target Hygiene (`eahtr.hygiene`)
1. **Strip `[...]` editorial insertions uniformly** (`strip_brackets=True`) so models do not transfer archival XML bracket markup onto clean manuscript scans.
2. **Normalize `ſ → s` and strip `^` caret markers** (`normalize_long_s=True`, `strip_caret_markers=True`) for consistent diplomatic targets.
3. **Remove malformed standalone `-` lines and adjacent copy-slip duplicates** (`has_adjacent_duplicate_lines`), preventing absorbing `-\n` loop states triggered by line-wrap hyphens.
4. **Verify strict split disjointness** (`audit_split_disjointness`): fail closed on any overlap across `document_id`, `writer_key`, `image_sha256`, or 32-word SHA-256 n-grams between `train` and `val` / `test_public`.

---

## 4. Two-Stage Halo Post-Training Pipeline (SFT → GRPO)

### 4.1 Stage 1: Full-Resolution Trainable-Vision SFT (1–1.5 Epochs)
Plain frozen-vision tuning fails to adapt stroke perception on cursive ink. Stage 1 unfreezes the visual pathway alongside attention/MLP LoRA (`r = 16`, `lora_alpha = 32`, `lora_dropout = 0.05`) for a short 1–1.5 epoch horizon:

- **Gemma 4–Specific Requirements (aligned with `gemma.ai/tuning/`):**
  - **Start from `-it` checkpoints** (`google/gemma-4-12B-it`, `google/gemma-4-26B-A4B-it`), as recommended for SFT and RL/GRPO.
  - **Explicit `1120` vision token budget:** Gemma 4 processors dynamically resize images to a selectable vision token budget (`70`, `140`, `280`, `560`, or `1120`). Because the default `280` tokens under-resolves dense handwritten ascenders/descenders and wrap hyphens, configs must explicitly set the **`1120` token budget** (alongside `max_pixels >= 2048 * 2048` where applicable).
  - **Multimodal tensor & loss masking:** Preserve `mm_token_type_ids`, apply the official Gemma 4 chat template with completion-only `-100` loss masking (`assistant_only_loss: true`) over prompt and image spans, and suppress thinking/reasoning blocks during transcription.
  - **Trainable vision modules:** Unfreeze `embed_vision` / `multimodal_embedder` on encoder-free `12B`, and `vision_tower` + `multi_modal_projector` on `26B-A4B`, verified via `resolve_vision_modules_to_save` (fail closed if zero modules match).
- **Inkling 2 Mini–Specific Requirements:**
  - **Full-resolution hierarchical patches:** Supply full-resolution scans (`max_pixels: 4194304`, well within Inkling's 4096 px encoder envelope) with `assistant_only_loss: true`.
  - **Trainable modules & Halo parallelism:** Unfreeze visual projection (`vision_projection` / `multi_modal_projector` / `patch_embed`) plus attention LoRA (`q_proj`, `k_proj`, `v_proj`, `o_proj`), using Halo's native `inkling_mm_model` and `EPInklingMoELayer` with Expert Parallelism (`expert_parallel_size: 8`).
- **Matched-Compute Sham Control:**
  - Run a deterministic mismatched image–text derangement (`make_sham_derangement`) at matched optimization steps to prove improvements come from visual ink reading rather than language-prior memorization.

### 4.2 Stage 2: Critic-Free GRPO (RLVR) with Verifiable Archival Rewards
Stage 2 optimizes the Stage 1 checkpoint against a deterministic, non-gameable Python verifier (`eahtr.rewards.score_transcription`):

$$R(y, y^*) = 0.6 \max\left(0,\, 1 - \text{CER}_{\text{strict}}(y, y^*)\right) + 0.4 \max\left(0,\, 1 - \text{WER}_{\text{strict}}(y, y^*)\right) - 0.05\, P_{\text{mod}}(y, y^*) - 0.10\, P_{\text{fmt}}(y) - 0.25\, P_{\text{trunc}}(y, y^*)$$

- **Hard Failure Floor (`reward = -1.0`):** Overrides all positive score if `finish_reason == "length"` (token-cap non-termination), `>= 4` consecutive repeated lines, `>= 4` consecutive standalone `-` tokens, `>= 5` consecutive repeated 4-grams, or non-empty text on a blank verso.
- **Archaic Spelling Guard (`P_mod`):** Penalizes attested period-to-modern normalizations (`chear → cheer`, `usefull → useful`, `publick → public`, etc.).
- **Format & Boilerplate Guard (`P_fmt`):** Penalizes unprompted `[...]` spans, Markdown code fences, and repository copyright watermarks absent from $y^*$.
- **KL Anchor:** `beta = 0.03` prevents catastrophic drift from out-of-distribution generalization.
- **Why GRPO Targets Both Model Families:**
  - On **Gemma 4**, eliminates wrap-hyphen `-` repetition loops and editorial bracket leakage.
  - On **Inkling 2 Mini**, directly penalizes `finish_reason == "length"` non-termination on dense cursive pages so the model learns clean EOS stopping.

### 4.3 Online GRPO Primary vs. Offline GRPO Fallback
- **Primary (`configs/halo/archival_grpo_online.yaml`):** Run `halo launch grpo/online` (`num_generations: 8`) with vLLM/SGLang rollouts and NCCL weight sync.
- **Immediate Fallback (`configs/halo/archival_grpo_offline.yaml`):** If multimodal online rollouts exhibit batch or processor bugs on either Gemma 4 or Inkling 2 Mini on Dec 1, immediately generate multi-temperature Stage 1 rollouts (`T ∈ {0.3, 0.7, 1.0}`), score deterministically via:
  ```bash
  python -m eahtr.rewards --input rollouts.jsonl --output scored_rollouts.jsonl
  ```
  and train via `halo launch grpo/offline` without stalling on rollout-server internals.

---

## 5. Evaluation Rubric & Local Apple Silicon MLX Gate

Every baseline, sham, SFT, and SFT+GRPO checkpoint is evaluated on the held-out `test_public` split and recorded in [`results/README.md`](../results/README.md):

1. **Strict WER / CER** (`eahtr.metrics.normalize_strict` — Unicode NFC, collapsed whitespace, preserving case, punctuation, period spelling, and line breaks).
2. **Lenient WER / CER** (`eahtr.metrics.normalize_lenient` — lowercased, punctuation-stripped).
3. **Decoding Health Counters:** `loop_pages`, `non_terminated_pages` (`finish_reason == "length"`), `editorial_leaks`, `modernizations`, and mean deterministic reward.
4. **Local Apple Silicon MLX Verification:** Export promoted checkpoints via `mlx_vlm.convert` (8-bit for `Gemma 4 12B`; 4-bit for `Gemma 4 26B-A4B`, `Inkling 2 Mini`, and optional `Qwen3.6-35B-A3B`) and verify:
   - Peak physical RAM `<= 24.0 GB` on 36 GB Apple Silicon
   - Zero repetition loops or non-terminations across `test_public`
   - Exact reproducibility of strict/lenient WER & CER after quantization

---

## 6. Execution Schedule & Kill Switches

| Window | Milestone | Concrete Actions & Exit Gates |
|---|---|---|
| **Nov 25–30 (Pre-Comp)** | **Pipeline & Smoke Prep** | Complete public corpus hygiene, unit tests, Halo YAML schema alignment, and pre-competition smoke runs verifying vision gradient flow and zero-shot baselines (`Gemma 4 12B`, `Inkling 2 Mini`). |
| **Dec 1** | **Track Rules & 50-Page Smoke** | Verify published Track Rules for `Gemma 4` (`12B` / `26B-A4B`) and `Inkling 2 Mini`. Run 50-page end-to-end Halo smoke test on partner hardware confirming Gemma 4 `1120` vision token budget, trainable vision grads, and GRPO rollouts. |
| **Dec 2–4** | **Track 1: Gemma 4 (SFT + GRPO)** | Run Stage 1 full-res SFT + matched-compute sham on `Gemma 4 12B` (and `26B-A4B` if credits allow), followed by Stage 2 GRPO and dev ablations (SFT-only vs. SFT+GRPO). |
| **Dec 5–7** | **Track 2: Inkling 2 Mini (SFT + GRPO)** | Run Stage 1 full-res SFT + matched-compute sham on `Inkling 2 Mini`, followed by Stage 2 GRPO specifically targeting `finish_reason == "length"` non-termination and dense-page EOS calibration. *(Run `Qwen3.6-35B-A3B` only if credits remain.)* |
| **Dec 8–9** | **Final Eval, MLX Gate & Freeze** | Score all frozen candidates once on `test_public`; run `mlx_vlm.convert` and 36 GB Apple Silicon peak-RAM/loop checks; update `results/README.md` with exact hashes and metrics. |
| **Dec 10** | **Submit** | Submit verified Apache-2.0 repo, Halo configs, reproducible training recipes, dataset/model attributions (`NOTICE`), and final results before deadline. |

### Operational Kill Switches
- **Online GRPO VLM failure on Dec 1:** Switch immediately to `halo launch grpo/offline` (`archival_grpo_offline.yaml`) rather than spending >1 day debugging rollout server multimodal batching.
- **Inkling 2 Mini 4-bit MLX exceeds 24 GB RAM:** Still submit the trained `Inkling 2 Mini` checkpoint to the **Best Inkling post-train** track if it improves on `test_public`, while promoting `Gemma 4 12B` / `26B-A4B` as the local Apple Silicon engine.
- **Gemma 4 4-bit MLX decoding instability:** Promote **Gemma 4 12B 8-bit MLX** (`<= 12.5 GiB`, comfortably within the 24 GB cap) alongside GRPO loop suppression.
- **Compute credit pressure:** Drop optional `Qwen3.6-35B-A3B` first and concentrate all remaining credits on **Gemma 4 12B** + **Inkling 2 Mini**.
