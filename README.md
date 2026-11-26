# Early American HTR — Halo Post-Training (`early-american-htr-halo`)

[![CI](https://github.com/markandster/early-american-htr-halo/actions/workflows/ci.yml/badge.svg)](https://github.com/markandster/early-american-htr-halo/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

Standalone **Apache-2.0** post-training toolkit, corpus hygiene pipeline, two-tier config validator, and verifiable archival preference/reward framework for **document-group disjoint (and optional creator-disjoint) 18th-century American diplomatic handwriting recognition** (c. 1750–1800 English round hand), designed for [White Circle Halo](https://github.com/whitecircle/halo) (`v1.3.0rc2`) and optional local quantized (MLX) workstation inference.

---

## Problem & Motivation

Off-the-shelf document OCR models trained on modern printed text perform poorly on unsegmented 18th-century cursive manuscripts, while plain cross-entropy Supervised Fine-Tuning (SFT) on vision-language models exhibits four failure modes on unseen writers:

1. **Absorbing Repetition Loops:** Line-wrap hyphens (`con-\n`) and tabular leader dots can trap autoregressive decoders in hundreds of consecutive `-` lines or token-cap non-termination.
2. **Modernization Drift:** Language priors silently "autocorrect" period orthography (`chear` → `cheer`, `usefull` → `useful`, `publick` → `public`).
3. **Editorial Bracket Leakage:** Unstripped `[...]` editorial insertions and placeholder tokens bleed from archival XML conventions into clean manuscript transcriptions.
4. **Premature Truncation & Blank-Page Hallucinations:** Stopping mid-letter or emitting archive boilerplate on blank versos.

This project post-trains models strictly for writer-general diplomatic transcription preserving period spelling, capitalization, punctuation, abbreviations, and exact physical line breaks (`\n`), pairing **full-resolution trainable-vision SFT** (`scripts/training/sft.py`) with **conditional verifiable Stage 2 VLM SMPO** (`scripts/training/preference/smpo.py` with `chosen_sft_ratio: 1.0`, promoted only if `val` CER improves cleanly over Stage 1) scored deterministically by `eahtr.rewards` / `eahtr.preferences.build_preference_pairs`. *(Text-only Halo GRPO configs are kept as contingency recipes.)*

See [docs/BACKGROUND.md](docs/BACKGROUND.md) for the pre-competition empirical motivation—how earlier exploratory HTR runs exposed absorbing dash loops, editorial bracket leakage, and unseen-writer reversals despite validation gains.

---

## Verifiable Archival Reward Function & VLM Preference Engine

For a completion $y$ and diplomatic ground truth $y^*$:

$$R(y, y^*) = 0.6 \max\left(0,\, 1 - \text{CER}_{\text{strict}}(y, y^*)\right) + 0.4 \max\left(0,\, 1 - \text{WER}_{\text{strict}}(y, y^*)\right) - 0.05\, P_{\text{mod}}(y, y^*) - 0.10\, P_{\text{fmt}}(y) - 0.25\, P_{\text{trunc}}(y, y^*)$$

- **Hard Failure Floor (`-1.0`):** Applied unconditionally on `finish_reason == "length"`, consecutive repeated lines (`>= 4`), repeated standalone dash tokens (`>= 4`), repeated n-gram loops, or non-empty output on blank versos.
- **Strict Diplomatic & Line-Break Scoring (`eahtr.metrics`):** Preserves case, punctuation, original spelling, and line breaks (`Unicode NFC`, collapsed whitespace) alongside lenient scoring, explicit `newline_cer`, `line_boundary_f1`, `order_tolerant_cer`, and clustered `paired_bootstrap_ci`.
- **Stage 2 Pair Builder (`eahtr.preferences.build_preference_pairs`):** Converts multi-temperature rollouts into Halo-contract `(images, prompt, chosen, rejected)` preference pairs with `prefer_gold_chosen=True`, positive non-blank quality gates, post-newline-norm deduplication, and `min_reward_margin` filtering.

---

## Rights-Cleared Corpus Policy

All competition and public training recipes enforce a strict license allowlist (`CC-BY-4.0`, `CC0-1.0`, `Public Domain`) via `eahtr.corpus.verify_source_license`:

- **Revolutionary City HTR Corpus** (Zenodo `15776323`, `CC-BY-4.0`, `INCLUDE_REVCITY = True`) — pairs both `.jpg`/`.jpeg` and `.png` images with ALTO-XML (`eahtr.corpus.parse_alto_xml_lines(..., sort_regions=False)` preserving annotated reading order by default), verifies `REVCITY_ZENODO_MD5` integrity with retry, retains both `raw_text` and cleaned `text`, flags two-page openings (`detect_two_page_opening`), groups related adjacent/recto-verso item IDs (`assign_revcity_item_groups(max_id_gap=30)` with `REVCITY_COLLECTION_OVERRIDES`), and assigns size-capped, per-institution balanced splits (`VAL_FRACTION=0.05`, `TEST_FRACTION=0.05`, `assign_balanced_group_splits`) yielding `train=2,897` (`17` large volumes), `val=126` (`21` groups), and `test_public=122` (`21` groups) — exhausting all `42` small groups across eval, with `val_slice` stratified evenly across institutions and document groups (document-group disjoint by default unless `PORTAL_CREATOR_MAP_PATH` or collection metadata is supplied; non-APS per-institution slices are not standalone reportable — see [results/README.md](results/README.md)). Empty-ALTO (`missing_or_unverified_gt`), leader-dot, and duplicate-line pages route to `hard_case_panel.jsonl` and strictly non-train/non-test `hard_case_held_out.jsonl` (`8` sparse diagnostic rows).
- **Library of Congress *By the People*** (`LCCN 2020446971`, `Public Domain`, opt-in `INCLUDE_LOC_BY_THE_PEOPLE = False`) — evaluated separately as a **genre-shift challenge panel**.
- **Strict Dev vs. Frozen Test Discipline:** Development notebooks and sweeps score **`val` only** (with hash-keyed caches and same-checkpoint image-swap causal checks; Stage-2 preference rollouts use strictly held-out `oof_train` shards); `test_public` remains untouched until final checkpoint freeze.
- **Opt-in Hugging Face Hub Export (`PUBLISH_TO_HF_DATASET = False`):** Validates **all pairwise splits** (`train` vs `val`, `train` vs `test_public`, `val` vs `test_public`) via `eahtr.hf_dataset` and emits Halo-ready `messages` + `images` (`Sequence(Image())`) columns.

All splits enforce hard disjointness via `eahtr.hygiene.audit_split_disjointness` across `document_id`, `writer_key`, `image_sha256`, normalized exact-text hashes (`>= 8` words), and 32-word SHA-256 n-grams, plus advisory perceptual `dHash` review (`dhash_review_candidates`, Hamming `<= 4`).

---

## Target Apache-2.0 Base Models

| Priority | Model | Role & Architecture | Optional Local Quantized Target (`<= 24 GB` MLX) | HF Access / License |
|---|---|---|---|---|
| **Primary (1)** | `google/gemma-4-12B-it` | 12B dense unified VLM (`gemma4_unified`, `1120` soft tokens; Stage 1 SFT → conditional Stage 2 SMPO) | 8-bit MLX (`<= 12.5 GiB`) | Ungated / Apache-2.0 (`rev 43c9d014b231`) |
| **Conditional Pilot (2)** | `thinkingmachines/Inkling-2-Small` | 48L × 384E NVFP4/MXFP8 MoE VLM (frozen-vision image-conditioned pilot post-Dec 1, `EP=8`) | Excluded from 24 GB MLX gate | Ungated / Apache-2.0 (`rev 885d78bfca61`) |

See [docs/TRAINING_PLAN.md](docs/TRAINING_PLAN.md) for the full competition training plan, chat-template parity rules (`gemma4_htr_nothink.jinja`, `inkling2_htr_nothink.jinja`), Stage 1 SFT → Stage 2 VLM SMPO pipeline (`chosen_sft_ratio: 1.0`), and MLX gate.

---

## Repository Layout

```text
├── .github/workflows/ci.yml      # Automated pytest, Halo YAML Tier-1, & notebook AST CI
├── configs/halo/                 # Halo SFT, VLM SMPO, contingency GRPO YAMLs & Jinja templates
├── docs/
│   ├── BACKGROUND.md             # Pre-competition prior HTR findings & design rationale
│   └── TRAINING_PLAN.md          # Full competition training plan & verification ledger
├── notebooks/
│   ├── colab_competition_pretrain_sft.ipynb  # Public-corpus Colab SFT, val eval, & Stage-2 rollouts
│   └── colab_halo_smoke.ipynb                # Pre-competition 1x GPU 5-step Halo smoke notebook
├── results/                      # Held-out evaluation & ablation ledger
├── scripts/
│   ├── validate_halo_configs.py  # Two-tier Halo config validator (CI + --halo-repo CPU Tier-2)
│   ├── stage_gemma4_highres.py   # Writable config + symlinked weight 1120-token Gemma 4 staging
│   ├── export_halo_datasets.py   # Post-expansion VLM Arrow DatasetDict exporter
│   ├── run_halo_smoke.sh         # Clean-room Python 3.12 1x GPU SFT + merge + continuity + SMPO smoke
│   ├── run_gemma4_halo.sh        # Canonical Gemma 4 12B Halo launcher & post-train merge/verify
│   ├── merge_and_eval_halo.py    # Halo adapter merge & frozen single-run val/test evaluator
│   ├── generate_halo_rollouts.py # OOF-guarded Stage 1 K=4 rollout sampler & GPU-hour estimator
│   └── verify_mlx_export.py      # Hardened multimodal Jinja2/token-parity & 1120-token export verifier
├── src/eahtr/                    # Modular Apache-2.0 Python package (<= 450 lines/module)
└── tests/                        # Pytest unit suite (metrics, rewards, hygiene, corpus, HF, Halo)
```

---

## Validation & Quickstart

Run unit tests and Tier-1 static validation locally:

```bash
pip install -e ".[dev]"
pytest -v
python scripts/validate_halo_configs.py
```

Run clean-room Tier-2 parser validation against an external `whitecircle/halo` (`v1.3.0rc2`) checkout inside Python 3.12:

```bash
python scripts/validate_halo_configs.py --halo-repo /tmp/halo
```

Run the 1× GPU 5-step Halo SFT → merge → continuity → SMPO smoke test (or open [notebooks/colab_halo_smoke.ipynb](notebooks/colab_halo_smoke.ipynb)):

```bash
bash scripts/run_halo_smoke.sh
```

Run canonical Gemma 4 `12B` Halo Stage 1 training & export:

```bash
STAGE=sft MODEL_SIZE=12b DATASET_DIR=/path/to/sft_arrow bash scripts/run_gemma4_halo.sh
```

Evaluate a merged checkpoint on `val.jsonl` (`--swap-images` & `--dual-convention` supported; `test_public.jsonl` requires explicit `--allow-frozen-test` after checkpoint freeze):

```bash
python scripts/merge_and_eval_halo.py --model-dir checkpoints/gemma4-12b-eahtr-sft-merged --eval-jsonl manifests/val.jsonl --out-json results/val_sft.json --dual-convention
```

Score a JSONL batch of candidate completions for Stage 2 preferences / Offline GRPO:

```bash
python -m eahtr.rewards --input rollouts.jsonl --output scored_rollouts.jsonl
```

---

## Colab Pro Training & Halo Reproducibility

[![Open In Colab (Pilot)](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/markandster/early-american-htr-halo/blob/main/notebooks/colab_competition_pretrain_sft.ipynb) [![Open In Colab (Halo Smoke)](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/markandster/early-american-htr-halo/blob/main/notebooks/colab_halo_smoke.ipynb)

All active base checkpoints (`google/gemma-4-12B-it` and `thinkingmachines/Inkling-2-Small`) are **ungated**; `HF_TOKEN` in Colab Secrets is only required when `PUBLISH_TO_HF_DATASET = True`. Both [colab_competition_pretrain_sft.ipynb](notebooks/colab_competition_pretrain_sft.ipynb) and [colab_halo_smoke.ipynb](notebooks/colab_halo_smoke.ipynb) enforce competition-cleared data, `val`-only dev evaluation, and full-resolution vision processing. `scripts/run_halo_smoke.sh` also runs standalone on any 1× CUDA Linux host (`>= 24 GB` VRAM, driver `>= 560.28`, e.g., Colab, Modal, RunPod, or AWS).

---

## License

Code, configs, templates, and reward scripts in this repository remain under the [Apache License, Version 2.0](LICENSE). White Circle Halo (`whitecircle/halo@v1.3.0rc2`) is never vendored here and remains separate under the Halo License. External datasets and base model weights remain under their respective permissive licenses (see [NOTICE](NOTICE)).
