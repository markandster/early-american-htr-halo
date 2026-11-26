# Pre-Competition Prior Work & Empirical Motivation

> **Clean-Room Boundary & Benchmark Separation**
>
> This document summarizes **pre-competition exploratory experiments** conducted on separate historical HTR models prior to building `early-american-htr-halo`. **None of these historical numbers belong in [results/README.md](../results/README.md).**
>
> Competition training starts from fresh Apache-2.0 base checkpoints (`google/gemma-4-12B-it@43c9d014b231`, conditional `thinkingmachines/Inkling-2-Small@885d78bfca61`). Zero prior trained weights or adapters, zero private or restricted scans or transcripts, and zero closed-model synthetic labels carry over. The public competition data scope is **Revolutionary City HTR** (`CC-BY-4.0`) and **Library of Congress *By the People*** (`Public Domain`, opt-in separately evaluated challenge panel). Live Gemma 4 12B competition runs and MLX deployment results are pending GPU execution (see [results/README.md](../results/README.md) and [NOTICE](../NOTICE)).

---

## Executive Motivation: Validation Gains vs. Fresh-Writer Reversal

Across three open-weight vision-language readers (`LightOnOCR-3-4B`, `olmOCR-2-7B`, and `Qwen3-VL-8B-Instruct`), supervised whole-page fine-tuning improved accuracy on the same **24 held-out validation sides** (9 writers absent from training), and a **shuffled-image causal control** confirmed strong dependence on paired manuscript images.

However, when the selected epoch-1 `LightOnOCR-3-4B` checkpoint was evaluated on an **independent transfer panel of 6 letters by 6 fresh writers** (17 sides, 3,778 words), error rates **reversed sharply** (including a 655-line absorbing dash loop on one page and bracketed editorial leakage rising from `0` to `11`). That pairing—real validation and image-grounding signal alongside brittle out-of-distribution transfer—directly explains why `early-american-htr-halo` invests in verifiable repetition floors, format/modernization penalties, layout metrics, causal controls, and strict held-out transfer testing.

---

## Prior Models Trained & Verified Results

All values below are percentages reported as **WER / CER** (lower is better). Each row compares the base checkpoint and fine-tuned checkpoint on the exact same held-out material.

| Prior Model & Training | Evaluation Scope | Base WER / CER (%) | Trained WER / CER (%) |
|---|---|---:|---:|
| `LightOnOCR-3-4B` (supervised whole-page training, language LoRA + trainable visual projection, epoch 1) | 24 validation sides, 9 writers absent from training, 2,506 words | 23.78 / 9.76 | 20.59 / 8.51 |
| `olmOCR-2-7B` (supervised whole-page training, language LoRA + trainable visual merger) | Same 24 validation sides | 30.73 / 12.97 | 21.27 / 8.51 |
| `Qwen3-VL-8B-Instruct` (supervised whole-page training, language LoRA + trainable visual mergers) | Same 24 validation sides | 27.02 / 12.26 | 25.62 / 10.88 |
| `LightOnOCR-3-4B` (same epoch-1 checkpoint as above) | Independent test: 6 letters by 6 fresh writers, 17 sides, 3,778 words | 21.65 / 10.61 | 36.26 / 12.53 |
| Kraken RevCity optical reader (full-network supervised adaptation on 7,311 human-aligned lines) | Within-collection test: 10 complete letters by 10 held-out writers, 34 sides, 3,956 words, primary reference convention | 39.69 / 19.78 | 34.98 / 17.76 |
| `olmOCR-2-7B` (local selective 8-bit MLX inference, frozen visual encoder + LoRA) | 4 held-out first pages, 1 per writer, 1,601 words | 9.74 / 6.38 | 9.68 / 6.33 |

### Methodological Caveats (Required Context)

- **Never rank across panels:** The 4-page MLX test, 24-side validation panel, 17-side fresh-writer test, and 34-side Kraken test use different holdings, lengths, and conventions and must not be treated as a single comparable benchmark.
- **4-page MLX caveat:** On the 4 held-out first pages, word edits fell from `156` to `155`, but rose from `150` to `153` when line-end hyphens were joined consistently. That result establishes **local 8-bit MLX workstation feasibility**, not a reliable accuracy gain.
- **Fresh-writer lenient scoring:** On the 17-side fresh-writer `LightOnOCR-3-4B` test, even lenient scoring deteriorated (`15.89 / 9.17` base vs. `18.26 / 11.68` trained).
- **Factor attribution:** The whole-page VLM validation runs combined high-resolution inputs and trainable visual components; the independent contribution of resolution vs. projector adaptation was not isolated in those experiments.

---

## What Worked (Transferable Signal & Causal Controls)

1. **Whole-page VLM adaptation across families:** All three open-weight VLM families improved on the 24-side validation panel, led by `olmOCR-2-7B` (`30.73% → 21.27%` WER, `12.97% → 8.51%` CER).
2. **Full-network adaptation over output-layer-only tuning (Kraken RevCity):** Holding line segmentation fixed across the 10-letter within-collection test, WER decreased on all 10 held-out letters. On the pooled test, output-layer-only adaptation reached `37.69%` WER / `19.00%` CER vs. `34.98%` WER / `17.76%` CER for full-network adaptation (from `39.69% / 19.78%` base).
3. **Shuffled-image causal control:** At equal `504` training presentations on the 24-side validation panel, `LightOnOCR-3-4B` scored `21.31%` WER / `8.80%` CER with true image–transcript pairs vs. `517.04%` WER / `498.74%` CER with shuffled pairs (runaway insertions can push edit rates past `100%`). Crucially, this control and the fresh-writer test answer different questions: visual pairing mattered, yet out-of-distribution transfer still failed.

---

## Observed Failures → Direct Safeguards in `early-american-htr-halo`

These safeguards are implemented and unit-tested in the repository; their quantitative impact on live Halo competition runs remains to be measured once GPU execution completes.

| Observed Prior Failure / Risk | Public-Safe Evidence | Fix Built into `early-american-htr-halo` |
|---|---|---|
| **Runaway repetition loops** | One fresh-writer prediction emitted `655` consecutive standalone dash lines after fine-tuning. | Hard `-1.0` reward floor on consecutive repeated lines, standalone dash loops, n-gram cycles, and token-cap non-termination ([`src/eahtr/rewards.py`](../src/eahtr/rewards.py)), plus conditional Stage 2 VLM SMPO (`chosen_sft_ratio: 1.0`). |
| **Editorial notation leaking into transcriptions** | Across 17 fresh sides, bracketed spans in model outputs rose from `0` to `11`. | Dual raw/cleaned reference retention, deletion-aligned prompt, and `P_fmt` penalty for bracket spans beyond the reference ([`README.md`](../README.md), [`src/eahtr/rewards.py`](../src/eahtr/rewards.py)). |
| **Modernization deltas masking flat exact spelling** | Across `223` unambiguous historical forms, modernizations fell from `49` to `37` while exact historical readings remained unchanged at `150`. | `P_mod` historical-to-modern substitution penalty ([`src/eahtr/rewards.py`](../src/eahtr/rewards.py)) paired with reporting exact spelling retention separately. |
| **Aggregate whitespace-collapsed CER hiding layout drift** | Strict competition CER preserves case, punctuation, and spelling while collapsing whitespace, including line breaks. | Explicit `newline_cer_pct`, `line_boundary_f1_pct`, and `order_tolerant_cer_pct` metrics alongside strict/lenient CER/WER ([`src/eahtr/metrics.py`](../src/eahtr/metrics.py)). |
| **Validation gains reversing on unseen writers** | `LightOnOCR-3-4B` improved on the 24-side panel yet regressed on 6 fresh writers (prior split leakage was not audited). | Document-group disjoint RevCity splits (`max_id_gap=30` + `REVCITY_COLLECTION_OVERRIDES`), creator grouping when attributed (`PORTAL_CREATOR_MAP_PATH`), SHA-256 / exact-text / 32-gram audits, and a separate LOC `challenge_loc` panel ([`README.md`](../README.md)). |

---

## How This Shapes the Halo Competition Strategy

For **Gemma 4 `12B` (`google/gemma-4-12B-it`)**, the competition pipeline in [docs/TRAINING_PLAN.md](TRAINING_PLAN.md) directly operationalizes these lessons:

- **Stage 1 full-resolution supervised adaptation (`scripts/training/sft.py`):** Stages a verified `1,120`-soft-token Gemma 4 processor directory (`sdpa`, exact non-thinking Jinja prefix parity), verifies live vision-tower updates, and pairs training with an internal Halo matched-compute sham control and a same-checkpoint image-swap causal test.
- **Earned Stage 2 VLM SMPO (`scripts/training/preference/smpo.py`):** Rather than blindly stacking RL over SFT, Stage 2 constructs gold-anchored preference pairs (`chosen_sft_ratio: 1.0`, out-of-fold Stage 1 failures penalized down to `-1.0`) and is **promoted only if it beats both decode-time repetition penalty and continued gold SFT** on held-out validation failures without hurting transcription fidelity.
- **Explicit transfer & deployment gates:** Out-of-collection generalization is checked separately on **LOC *By the People*** (`challenge_loc`), and optional `8`-bit `<= 24 GB` local MLX conversion must pass `1,120`-token and multimodal chat-template/pixel-value parity checks ([`scripts/verify_mlx_export.py`](../scripts/verify_mlx_export.py)).
