# Public Competition Benchmark Results

Development tuning, ablations, and checkpoint selection run exclusively on **`val`** (`CC-BY-4.0` Revolutionary City split, disjoint by item group, perceptual dHash, image SHA-256, and 32-word n-grams). Frozen **`test_public`** numbers below are populated only by a single post-freeze evaluation command after checkpoint, decoding settings, and metrics are locked. **LOC *By the People*** (`LCCN 2020446971`, Public Domain) is reported separately as a genre-shift challenge panel.

Every reported evaluation records:
- **Strict & Lenient WER / CER:** Strict preserves case, punctuation, period spelling, and line breaks (`Unicode NFC`); lenient strips punctuation and lowercases; zero-shot baselines also report dual-convention `raw_strict_cer_pct` against uncleaned XML.
- **Line-Break & Order Metrics:** `newline_cer_pct`, `line_boundary_f1_pct`, and `order_tolerant_cer_pct` (`eahtr.metrics.score_line_breaks`).
- **Causal & Statistical Controls:** Matched-compute `sham_short` control, same-checkpoint **image-swap delta** (`swap_cer_degradation`), and clustered 95% **paired bootstrap CIs** (`paired_bootstrap_ci` by `document_id`).
- **Decoding Health Denominators:** Separately reports `loop_pages`, `budget_capped_pages`, `non_terminated_pages`, `editorial_leaks`, `modernizations`, `thinking_leak_pages`, and mean `eahtr.rewards`.

## Leaderboard & Ablation Tracker

| Model | Stage / Arm | Split Scope | Strict WER / CER | Newline F1 / Order-Tol CER | Swap Δ / 95% CI | Loop / Cap Pages | Mean Reward | Apple Silicon 24 GB MLX | Notes |
|---|---|---|---|---|---|---|---|---|---|
| `google/gemma-4-12B-it` | Zero-shot baseline (cleaned + raw) | `revcity-val` / frozen `test_public` | _pending_ | _pending_ | — | _pending_ | _pending_ | — | Primary track; 1120 soft tokens, `sdpa`, dual-convention |
| `google/gemma-4-12B-it` | Matched-compute sham | `revcity-val` / frozen `test_public` | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ | — | Proves image-conditioned learning vs optimization budget |
| `google/gemma-4-12B-it` | Stage 1 VLM SFT (Halo Primary) | `revcity-val` / frozen `test_public` | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ (8-bit <= 12.5 GiB) | Non-thinking parity template, trainable vision |
| `google/gemma-4-12B-it` | Stage 2 VLM SMPO (Conditional) | `revcity-val` / frozen `test_public` | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ | Merged Stage 1 start + `chosen_sft_ratio: 1.0`; promoted only if `val` improves |
| `google/gemma-4-12B-it` | LOC Genre-Shift Challenge | `challenge_loc` | _pending_ | _pending_ | — | _pending_ | _pending_ | — | Out-of-distribution Public Domain panel |
| `google/gemma-4-26B-A4B-it` | Deferred reference | `revcity-val` | _deferred_ | _deferred_ | — | _deferred_ | _deferred_ | Excluded (>24 GB MLX) | Deferred behind 12B smoke & budget |
| `google/gemma-4-12B-it` | Stage 2 VLM DPO (Deferred Ablation) | `revcity-val` | _deferred_ | _deferred_ | — | _deferred_ | _deferred_ | — | Deferred ablation |
| `thinkingmachines/Inkling-2-Small` | Conditional Image Pilot | `revcity-val` | _conditional_ | _conditional_ | _conditional_ | _conditional_ | _conditional_ | Excluded (>24 GB MLX) | Frozen-vision SFT → SMPO (`chosen_sft_ratio: 1.0`) pending Dec 1 EP=8 memory & serving check |
| `thinkingmachines/Inkling-2-Mini` | Blocked reference | — | _blocked_ | — | — | — | — | Excluded | Gated on upstream `inkling2_loop` loader support |
