# Early American HTR — Halo Post-Training (`early-american-htr-halo`)

[![CI](https://github.com/markandster/early-american-htr-halo/actions/workflows/ci.yml/badge.svg)](https://github.com/markandster/early-american-htr-halo/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

Standalone **Apache-2.0** post-training toolkit, corpus hygiene pipeline, two-tier config validator, and verifiable archival preference/reward framework for **writer-general 18th-century American diplomatic handwriting recognition** (c. 1750–1800 English round hand), designed for [White Circle Halo](https://github.com/whitecircle/halo) (`v1.3.0rc2`) and local Apple Silicon (36 GB MLX) inference.

---

## Problem & Motivation

Off-the-shelf document OCR models trained on modern printed text perform poorly on unsegmented 18th-century cursive manuscripts, while plain cross-entropy Supervised Fine-Tuning (SFT) on vision-language models exhibits four failure modes on unseen writers:

1. **Absorbing Repetition Loops:** Line-wrap hyphens (`con-\n`) and tabular leader dots can trap autoregressive decoders in hundreds of consecutive `-` lines or token-cap non-termination.
2. **Modernization Drift:** Language priors silently "autocorrect" period orthography (`chear` → `cheer`, `usefull` → `useful`, `publick` → `public`).
3. **Editorial Bracket Leakage:** Unstripped `[...]` editorial insertions and placeholder tokens bleed from archival XML conventions into clean manuscript transcriptions.
4. **Premature Truncation & Blank-Page Hallucinations:** Stopping mid-letter or emitting archive boilerplate on blank versos.

This project pairs **full-resolution trainable-vision SFT** (`scripts/training/sft.py`) with **verifiable Stage 2 VLM preference optimization (`scripts/training/preference/smpo.py` primary, `dpo.py` secondary)** scored deterministically by `eahtr.rewards` (`build_preference_pairs`). *(Text-only Halo GRPO recipes are kept as contingency configs since Halo's GRPO trainers refuse image columns.)*

---

## Scope & Architectural Boundaries (Non-Goals)

- **In Scope — Diplomatic Transcription with Exact Line Breaks:** Models are post-trained strictly for writer-general diplomatic transcription preserving period spelling, capitalization, punctuation, abbreviations, and **exact physical line breaks (`\n`)**, supporting downstream line-to-transcript alignment at zero extra token cost.
- **Out of Scope — Word-Level Bounding-Box Generation:** Autoregressive per-word coordinate emission increases sequence length ~3–5× per page, worsening token-cap non-termination and repetition loops while slowing rollouts.
- **Out of Scope — Writer Identification / Authorship Attribution:** HTR optimizes *writer-invariant* text recognition rather than author attribution.

---

## Verifiable Archival Reward Function & VLM Preference Engine

For a completion $y$ and diplomatic ground truth $y^*$:

$$R(y, y^*) = 0.6 \max\left(0,\, 1 - \text{CER}_{\text{strict}}(y, y^*)\right) + 0.4 \max\left(0,\, 1 - \text{WER}_{\text{strict}}(y, y^*)\right) - 0.05\, P_{\text{mod}}(y, y^*) - 0.10\, P_{\text{fmt}}(y) - 0.25\, P_{\text{trunc}}(y, y^*)$$

- **Hard Failure Floor (`-1.0`):** Applied unconditionally on `finish_reason == "length"`, consecutive repeated lines (`>= 4`), repeated standalone dash tokens (`>= 4`), repeated n-gram loops, or non-empty output on blank versos.
- **Strict Diplomatic Scoring (`eahtr.metrics`):** Preserves case, punctuation, original spelling, and line breaks (`Unicode NFC`, collapsed whitespace) alongside lenient scoring (lowercased, punctuation-stripped).
- **Stage 2 Pair Builder (`eahtr.rewards.build_preference_pairs`):** Converts multi-temperature rollouts into Halo-contract `(images, prompt, chosen, rejected)` preference pairs filtered by `min_reward_margin`.

---

## Rights-Cleared Corpus Policy

All competition and public training recipes enforce a strict license allowlist (`CC-BY-4.0`, `CC0-1.0`, `Public Domain`) via `eahtr.corpus.verify_source_license`:

- **Revolutionary City HTR Corpus** (Zenodo `15776323`, `CC-BY-4.0`, `INCLUDE_REVCITY = True`) — ALTO-XML parsed by `eahtr.corpus.parse_alto_xml_lines`.
- **Bentham ICFHR-2016 HTR Corpus** (Zenodo `1164045`, `CC-BY-4.0`, opt-in `INCLUDE_BENTHAM = False`) — PAGE-XML parsed by `eahtr.corpus.parse_page_xml_lines` (direct `<TextLine>/<TextEquiv>` first) and isolated to `train` so shared hands never enter held-out splits.
- **Library of Congress *By the People*** (`LCCN 2020446971`, `Public Domain`, opt-in `INCLUDE_LOC_BY_THE_PEOPLE = False`).
- **Opt-in Hugging Face Hub Export (`PUBLISH_TO_HF_DATASET = False`):** Validates **all pairwise splits** (`train` vs `val`, `train` vs `test_public`, `val` vs `test_public`) via `eahtr.hf_dataset` and emits Halo-ready `messages` + `images` (`Sequence(Image())`) columns.

All splits are gated by `eahtr.hygiene.audit_split_disjointness` across `document_id`, `writer_key`, `image_sha256`, normalized full-page text hashes (`>= 8` words), and 32-word SHA-256 n-grams.

---

## Target Apache-2.0 Base Models

| Priority | Model | Role & Architecture | Local Target (36 GB Apple Silicon MLX) | HF Access / License |
|---|---|---|---|---|
| **1A** | `google/gemma-4-12B-it` | 12B dense unified VLM (`gemma4_unified`, `1120` soft tokens) | 8-bit MLX (`<= 12.5 GiB`) | Ungated / Apache-2.0 (`rev 43c9d014b231`) |
| **1B** | `google/gemma-4-26B-A4B-it` | 26B / 4B-active MoE VLM (`gemma4`, `EP=8`) | 4-bit MLX (`~14.3–15.5 GiB`) | Ungated / Apache-2.0 (`rev 4d7ae4984b7d`) |
| **2A** | `thinkingmachines/Inkling-2-Small` | 48L × 384E NVFP4/MXFP8 MoE VLM (Track 2 Primary, default `EP=8`) | Server / Halo `EP>=2` | Ungated / Apache-2.0 (`rev 885d78bfca61`) |
| **2B** | `thinkingmachines/Inkling-2-Mini` | Looped MoE VLM (`inkling2_loop`, **blocked** on upstream loader support) | Revisit post-Dec 1 | Ungated / Apache-2.0 (`rev c9d028302cd0`) |
| **3** | `Qwen/Qwen3.6-35B-A3B` | 35B / 3B-active MoE VLM (`qwen3_5_moe`, optional) | 4-bit MLX (`~20.4–21.5 GiB`) | Ungated / Apache-2.0 |

See [docs/TRAINING_PLAN.md](docs/TRAINING_PLAN.md) for the full competition training plan, chat-template parity rules (`gemma4_htr_nothink.jinja`, `inkling2_htr_nothink.jinja`), Stage 1 SFT → Stage 2 VLM SMPO/DPO pipeline, and MLX gate.

---

## Repository Layout

```text
├── .github/workflows/ci.yml      # Automated pytest, Halo YAML Tier-1, & notebook AST CI
├── configs/halo/                 # Halo SFT, VLM SMPO/DPO, contingency GRPO YAMLs & Jinja templates
├── docs/TRAINING_PLAN.md         # Full competition training plan & verification ledger
├── notebooks/
│   ├── colab_competition_pretrain_sft.ipynb  # Public-corpus Colab SFT, eval, & Stage-2 rollouts
│   └── colab_halo_smoke.ipynb                # Pre-competition 1x GPU 5-step Halo smoke notebook
├── results/                      # Public held-out benchmark & ablation ledger
├── scripts/
│   ├── validate_halo_configs.py  # Two-tier Halo config validator (CI + --halo-repo CPU Tier-2)
│   └── run_halo_smoke.sh         # Clean-room Python 3.12 1x GPU Halo SFT + SMPO smoke script
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

Run the 1× GPU 5-step Halo SFT + SMPO smoke test (or open [notebooks/colab_halo_smoke.ipynb](notebooks/colab_halo_smoke.ipynb)):

```bash
bash scripts/run_halo_smoke.sh
```

Score a JSONL batch of candidate completions for Halo Offline GRPO:

```bash
python -m eahtr.rewards --input rollouts.jsonl --output scored_rollouts.jsonl
```

---

## Colab Pro Training & Halo Reproducibility

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/markandster/early-american-htr-halo/blob/main/notebooks/colab_competition_pretrain_sft.ipynb)

All active base checkpoints (`google/gemma-4-12B-it`, `google/gemma-4-26B-A4B-it`, `thinkingmachines/Inkling-2-Small`, and `Qwen/Qwen3.6-35B-A3B`) are **ungated**; `HF_TOKEN` in Colab Secrets is only required when `PUBLISH_TO_HF_DATASET = True`. Both [colab_competition_pretrain_sft.ipynb](notebooks/colab_competition_pretrain_sft.ipynb) and [colab_halo_smoke.ipynb](notebooks/colab_halo_smoke.ipynb) enforce competition-cleared data and full-resolution vision processing.

---

## License

Code, configs, templates, and reward scripts in this repository remain under the [Apache License, Version 2.0](LICENSE). White Circle Halo (`whitecircle/halo@v1.3.0rc2`) is never vendored here and remains separate under the Halo License. External datasets and base model weights remain under their respective permissive licenses (see [NOTICE](NOTICE)).
