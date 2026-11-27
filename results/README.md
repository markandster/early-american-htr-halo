# Public Competition Benchmark Results

Development tuning, ablations, and checkpoint selection run exclusively on **`val`** (`CC-BY-4.0` Revolutionary City in-collection balanced split across all four institutions: `train=2,897` pages / `92.1%` across `17` large volumes [`1,424` APS / `1,303` HSP / `124` LCP / `46` MSS], `val=124` pages / `3.9%` across `21` groups [`83` APS / `31` HSP / `4` LCP / `6` MSS], and frozen `test_public=124` pages / `3.9%` across `21` groups [`94` APS / `10` HSP / `16` LCP / `4` MSS]; hard-disjoint by item/collection group `max_id_gap=30` + `REVCITY_COLLECTION_OVERRIDES`, `writer_key`, image SHA-256, `>= 8`-word exact text, and 32-word n-grams, with advisory perceptual `dHash` review). Split shape & reporting limits: all `42` small groups (`<= 40` pages, `248` pages total) go to eval (`val=3.9%`, `test_public=3.9%`, falling short of `5%` because small groups are exhausted) while `train` consists of `17` large volumes; default RevCity metadata has `attributed_hands = 0` unless `PORTAL_CREATOR_MAP_PATH` is supplied (so splits are **document-group disjoint** rather than writer-attributed or collection-verified across all collections); per-institution numbers outside APS (`HSP test=10`, `LCP val=4`, `MSS val=6 / test=4`) are **not reportable** as standalone per-institution benchmarks; and `hard_case_held_out.jsonl` contains `8` sparse (`<= 7`-word) diagnostic pages. Frozen **`test_public`** numbers below are populated only by a single post-freeze evaluation command after checkpoint, decoding settings, and metrics are locked. **LOC *By the People*** (`LCCN 2020446971`, Public Domain) is reported separately as an out-of-collection genre-shift challenge panel.

Every reported evaluation records:
- **Strict & Lenient WER / CER:** Strict preserves case, punctuation, period spelling, and line breaks (`Unicode NFC`); lenient strips punctuation and lowercases; zero-shot baselines also report dual-convention `raw_strict_cer_pct` against uncleaned XML.
- **Line-Break & Order Metrics:** `newline_cer_pct`, `line_boundary_f1_pct`, and `order_tolerant_cer_pct` (`eahtr.metrics.score_line_breaks`).
- **Causal & Statistical Controls:** Planned matched-compute Halo sham control (full-mode Colab `sham_short`, capped at `<= 512` pages, is a short unmatched control; smoke mode matches `real` at `128`), same-checkpoint **image-swap delta** (`swap_cer_degradation` via `merge_and_eval_halo.py --swap-images`), and clustered 95% **paired bootstrap CIs** (`paired_bootstrap_ci` by `document_id`, weighted by normalized `ref_chars` with pre-inference unique 1-to-1 baseline `record_id` matching).
- **Decoding Health Denominators:** Separately reports `loop_pages`, `budget_capped_pages`, `non_terminated_pages`, `editorial_leaks`, `modernizations`, `thinking_leak_pages`, and mean `eahtr.rewards`.

## Leaderboard & Ablation Tracker

| Model | Stage / Arm | Split Scope | Strict WER / CER | Newline F1 / Order-Tol CER | Swap Δ / 95% CI | Loop / Cap Pages | Mean Reward | Local MLX (`<= 24 GB`) | Notes |
|---|---|---|---|---|---|---|---|---|---|
| `google/gemma-4-12B-it` | Zero-shot baseline (cleaned + raw) | `revcity-val` / frozen `test_public` | _pending_ | _pending_ | — | _pending_ | _pending_ | — | Primary track; 1120 soft tokens, `sdpa`, dual-convention |
| `google/gemma-4-12B-it` | Planned matched-compute sham (Halo) / short unmatched control (`sham_short` in full Colab) | `revcity-val` / frozen `test_public` | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ | — | Proves image-conditioned learning; full Colab `sham_short` capped `<= 512` pages |
| `google/gemma-4-12B-it` | Stage 1 VLM SFT (Halo Primary) | `revcity-val` / frozen `test_public` | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ (8-bit <= 12.5 GiB) | Non-thinking parity template, trainable vision |
| `google/gemma-4-12B-it` | Stage 2 VLM SMPO (Conditional) | `revcity-val` / frozen `test_public` | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ | Merged Stage 1 start + `chosen_sft_ratio: 1.0`; promoted only if `val` improves |
| `google/gemma-4-12B-it` | LOC Genre-Shift Challenge | `challenge_loc` | _pending_ | _pending_ | — | _pending_ | _pending_ | — | Out-of-distribution Public Domain panel |
| `thinkingmachines/Inkling-2-Small` | Conditional Image Pilot | `revcity-val` | _conditional_ | _conditional_ | _conditional_ | _conditional_ | _conditional_ | Excluded (>24 GB MLX) | Frozen-vision Stage 1 SFT (Stage 2 SMPO only if expert-shard reload is proven) pending Dec 1 EP=8 memory & serving check |

## Pre-Competition Colab Smoke Pilot (24 GB L4, 124 Train / 24 Val Pages)

| Model | Stage / Arm | Strict WER / CER | Lenient / Raw Strict CER | Newline CER / Line-Boundary F1 / Order-Tol CER | Image-Swap Δ (16 pages) | Loop / Cap / Non-Term | Editorial / Mod / Thinking Leaks | Mean Reward | Notes |
|---|---|---|---|---|---|---|---|---|---|
| `google/gemma-4-12B-it` | `baseline_zeroshot` | `159.99%` / `151.79%` | `103.03%` / `149.43%` | — | — | `7/24` / `8/24` / — | — / — / `0/24` | `-0.1715` | 24 stratified `val` pages; `1,120` soft tokens |
| `google/gemma-4-12B-it` | `ARM=real` QLoRA smoke (`main@89af415`) | `57.35%` / `38.12%` | `33.98%` / `37.80%` | `38.69%` / `14.84%` / `35.89%` | `+79.75 pp` (`swap CER 117.01%`) | `2/24` / `2/24` / `2/24` | `0` / `0` / `0` | `0.5595` | `124/124` pages, `0` OOMs (`10m55s`), `1,066` image tokens, trainable `model.embed_vision` |

Diagnostic note: `line_boundary_f1_pct=14.84%` and `2/24` loop/capped pages on this 124-page smoke are primary targets for pre-competition error analysis, full-corpus (`2,897`-page) training, and Stage 2 SMPO / repetition-penalty gating.
