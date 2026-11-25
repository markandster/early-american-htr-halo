# Early American HTR — Halo Post-Training (`early-american-htr-halo`)

[![CI](https://github.com/markandster/early-american-htr-halo/actions/workflows/ci.yml/badge.svg)](https://github.com/markandster/early-american-htr-halo/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

Standalone **Apache-2.0** post-training toolkit, corpus hygiene pipeline, and verifiable GRPO (RLVR) reward framework for **writer-general 18th-century American diplomatic handwriting recognition** (c. 1750–1800 English round hand), designed for [White Circle Halo](https://github.com/whitecircle/halo) and local Apple Silicon (M3 Max 36 GB, MLX) inference.

---

## Problem & Motivation

Off-the-shelf document OCR models trained on modern printed text perform poorly on unsegmented 18th-century cursive manuscripts, while plain cross-entropy Supervised Fine-Tuning (SFT) on vision-language models exhibits four failure modes on unseen writers:

1. **Absorbing Repetition Loops:** Line-wrap hyphens (`con-\n`) and tabular leader dots can trap autoregressive decoders in hundreds of consecutive `-` lines or token-cap non-termination.
2. **Modernization Drift:** Language priors silently "autocorrect" period orthography (`chear` → `cheer`, `usefull` → `useful`, `publick` → `public`).
3. **Editorial Bracket Leakage:** Unstripped `[...]` editorial insertions and placeholder tokens bleed from archival XML conventions into clean manuscript transcriptions.
4. **Premature Truncation & Blank-Page Hallucinations:** Stopping mid-letter or emitting archive boilerplate on blank versos.

This project pairs **full-resolution SFT** (unfrozen vision projector / encoder-free patch path + LoRA/FSDP2) with **critic-free GRPO (RLVR)** scored by a deterministic archival verifier that directly penalizes all four failure modes.

---

## Scope & Architectural Boundaries (Non-Goals)

- **In Scope — Diplomatic Transcription with Exact Line Breaks:** Models are post-trained strictly for writer-general diplomatic transcription preserving period spelling, capitalization, punctuation, abbreviations, and **exact physical line breaks (`\n`)**, which directly supports downstream line-to-transcript alignment at zero extra token cost.
- **Out of Scope — Word-Level Bounding-Box Generation:** Autoregressive per-word JSON or coordinate emission increases sequence length ~3–5× per page, worsening token-cap non-termination and repetition loops while slowing GRPO rollouts. Word localization is better handled downstream by dedicated line segmentation plus CTC forced alignment. (Separate-prompt *line-level* bounding-box experiments over ALTO-XML `<TextLine>` polygons are deferred until transcription converges.)
- **Out of Scope — Writer Identification / Authorship Attribution:** HTR optimizes *writer-invariant* representations (mapping varied hands to the same text), whereas writer identification requires *content-invariant, writer-specific* stroke geometry and calibrated open-set likelihood scoring. Training an autoregressive VLM on writer labels risks lexical/place-name shortcuts and conflates archival correspondent metadata with verified penmanship (many 18th-century official letters were written by clerks or aides).

---

## Verifiable Archival Reward Function

For a completion $y$ and diplomatic ground truth $y^*$:

$$R(y, y^*) = w_c \max\left(0,\, 1 - \text{CER}_{\text{strict}}(y, y^*)\right) + w_w \max\left(0,\, 1 - \text{WER}_{\text{strict}}(y, y^*)\right) - \lambda_{\text{mod}} P_{\text{mod}}(y, y^*) - \lambda_{\text{fmt}} P_{\text{fmt}}(y) - \lambda_{\text{trunc}} P_{\text{trunc}}(y, y^*)$$

- **Hard Failure Floor (`-1.0`):** Applied unconditionally on `finish_reason == "length"`, consecutive repeated lines (`>= 4`), repeated standalone dash tokens (`>= 4`), repeated 4-gram loops, or non-empty output on blank versos.
- **Strict Diplomatic Scoring (`eahtr.metrics`):** Preserves case, punctuation, original spelling, and line breaks (`Unicode NFC`, collapsed whitespace) alongside lenient scoring (lowercased, punctuation-stripped).
- **Archival Spelling & Leakage Penalties:** Penalizes attested 18th-century spelling modernizations and unprompted `[...]` / Markdown / watermark boilerplate beyond what appears in $y^*$.

---

## Rights-Cleared Corpus Policy

All competition and public training recipes enforce a strict license allowlist (`CC-BY-4.0`, `CC0-1.0`, `Public Domain`) via `eahtr.corpus.verify_source_license`:

- **Revolutionary City HTR Corpus** (Zenodo `15776323`, `CC-BY-4.0`) — ALTO-XML parsed by `eahtr.corpus.parse_alto_xml_lines`.
- **Library of Congress *By the People*** (`Public Domain`) — volunteer as-written manuscript transcriptions paired with public-domain master scans.

All splits are gated by `eahtr.hygiene.audit_split_disjointness` for zero collisions across `document_id`, `writer_key`, `image_sha256`, and 32-word SHA-256 n-grams.

---

## Target Apache-2.0 Base Models

| Priority | Model | Architecture | Local Target (M3 Max 36 GB MLX) | License |
|---|---|---|---|---|
| **1A** | **Gemma 4 12B** | 12B dense encoder-free unified VLM | 8-bit / 4-bit MLX (`<= 12 GiB`) | Apache-2.0 |
| **1B** | **Gemma 4 26B-A4B** | 26B total / 4B active MoE VLM | 4-bit MLX (`~14.3 GiB`) | Apache-2.0 |
| **2** | **Inkling 2 Mini** | 26.4B looped MoE VLM (`~5.4B` eff. active) | Verify 4-bit MLX `<= 24 GiB` | Apache-2.0 |
| **3** | **Qwen3.6-35B-A3B** | 35B total / 3B active MoE VLM | 4-bit MLX (`~20.4 GiB`) | Apache-2.0 |

---

## Repository Layout

```text
├── .github/workflows/ci.yml # Automated pytest, YAML, and notebook AST checks
├── configs/halo/            # Halo SFT & Online/Offline GRPO YAML templates
├── notebooks/               # Self-contained Colab Pro public-corpus SFT & rollout notebook
├── results/                 # Public benchmark & ablation ledger (held-out public splits)
├── src/eahtr/
│   ├── colab_pipeline.py    # Public-data Colab notebook generator & candidate specs
│   ├── metrics.py           # Strict & lenient diplomatic WER/CER via rapidfuzz
│   ├── rewards.py           # Deterministic GRPO archival reward & JSONL CLI scorer
│   ├── hygiene.py           # Target normalization, slip/leader-dot filters, split audit
│   └── corpus.py            # License allowlist gate & ALTO-XML line extractor
└── tests/                   # Pytest unit suite covering metrics, rewards, hygiene, corpus
```

---

## Colab Pro Training & Halo Reproducibility

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/markandster/early-american-htr-halo/blob/main/notebooks/colab_competition_pretrain_sft.ipynb)

The notebook [colab_competition_pretrain_sft.ipynb](notebooks/colab_competition_pretrain_sft.ipynb) mirrors our full-resolution (`max_pixels=2048*2048`), trainable-vision, matched-compute sham-controlled training harness while strictly enforcing **competition-cleared public datasets only** (`Revolutionary City` Zenodo `15776323` CC-BY-4.0, plus opt-in `Library of Congress By the People` Public Domain).

- **First-run smoke test:** `SMOKE_TEST_MODE = True` is enabled by default in Cell 3 (400 RevCity pages, 1 epoch, 128-presentation sham control) so you can verify end-to-end CUDA, processor, vision-gradient flow, and evaluation in ~15–25 minutes before flipping `SMOKE_TEST_MODE = False` for the full ~3,200-page run.
- **Hugging Face access:** For gated Gemma 4 checkpoints (`google/gemma-4-12B-it`), accept the model terms on Hugging Face and store `HF_TOKEN` in Colab Secrets (the notebook reads `userdata.get('HF_TOKEN')` automatically). `thinkingmachines/Inkling-2-Mini` is ungated.

> **Competition Rule Note:** Pre-December 1 Colab runs establish baseline benchmarks, target hygiene, vision-gradient verification, and Stage 2 offline-GRPO rollouts on the public dataset. Final competition submissions after December 1 must be reproduced end-to-end using the official **Halo** training recipe (`github.com/whitecircle/halo`) under published Track Rules.

---

## Quickstart

```bash
pip install -e ".[dev]"
pytest
```

Score a JSONL batch of candidate completions for Halo Offline GRPO:

```bash
python -m eahtr.rewards --input rollouts.jsonl --output scored_rollouts.jsonl
```

---

## License

Code, configs, and reward scripts in this repository are licensed under the [Apache License, Version 2.0](LICENSE). External datasets and base model weights remain under their respective permissive licenses (see [NOTICE](NOTICE)).
