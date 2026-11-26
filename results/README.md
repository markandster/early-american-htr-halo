# Public Competition Benchmark Results

Development tuning, ablations, and checkpoint selection run exclusively on **`val`** (`CC-BY-4.0` Revolutionary City balanced split across institutions, `val=287` pages / `39` groups vs. frozen `test_public=276` pages / `40` groups; hard-disjoint by item group, `writer_key`, image SHA-256, `>= 8`-word exact text, and 32-word n-grams, with advisory perceptual `dHash` review and `hard_case_held_out.jsonl`). Frozen **`test_public`** numbers below are populated only by a single post-freeze evaluation command after checkpoint, decoding settings, and metrics are locked. **LOC *By the People*** (`LCCN 2020446971`, Public Domain) is reported separately as a genre-shift challenge panel.

Every reported evaluation records:
- **Strict & Lenient WER / CER:** Strict preserves case, punctuation, period spelling, and line breaks (`Unicode NFC`); lenient strips punctuation and lowercases; zero-shot baselines also report dual-convention `raw_strict_cer_pct` against uncleaned XML.
- **Line-Break & Order Metrics:** `newline_cer_pct`, `line_boundary_f1_pct`, and `order_tolerant_cer_pct` (`eahtr.metrics.score_line_breaks`).
- **Causal & Statistical Controls:** Matched-compute `sham_short` control, same-checkpoint **image-swap delta** (`swap_cer_degradation`), and clustered 95% **paired bootstrap CIs** (`paired_bootstrap_ci` by `document_id`, weighted by normalized `ref_chars` with 1-to-1 baseline `record_id` matching).
- **Decoding Health Denominators:** Separately reports `loop_pages`, `budget_capped_pages`, `non_terminated_pages`, `editorial_leaks`, `modernizations`, `thinking_leak_pages`, and mean `eahtr.rewards`.

## Leaderboard & Ablation Tracker

| Model | Stage / Arm | Split Scope | Strict WER / CER | Newline F1 / Order-Tol CER | Swap Δ / 95% CI | Loop / Cap Pages | Mean Reward | Local MLX (`<= 24 GB`) | Notes |
|---|---|---|---|---|---|---|---|---|---|
| `google/gemma-4-12B-it` | Zero-shot baseline (cleaned + raw) | `revcity-val` / frozen `test_public` | _pending_ | _pending_ | — | _pending_ | _pending_ | — | Primary track; 1120 soft tokens, `sdpa`, dual-convention |
| `google/gemma-4-12B-it` | Matched-compute sham | `revcity-val` / frozen `test_public` | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ | — | Proves image-conditioned learning vs optimization budget |
| `google/gemma-4-12B-it` | Stage 1 VLM SFT (Halo Primary) | `revcity-val` / frozen `test_public` | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ (8-bit <= 12.5 GiB) | Non-thinking parity template, trainable vision |
| `google/gemma-4-12B-it` | Stage 2 VLM SMPO (Conditional) | `revcity-val` / frozen `test_public` | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ | Merged Stage 1 start + `chosen_sft_ratio: 1.0`; promoted only if `val` improves |
| `google/gemma-4-12B-it` | LOC Genre-Shift Challenge | `challenge_loc` | _pending_ | _pending_ | — | _pending_ | _pending_ | — | Out-of-distribution Public Domain panel |
| `thinkingmachines/Inkling-2-Small` | Conditional Image Pilot | `revcity-val` | _conditional_ | _conditional_ | _conditional_ | _conditional_ | _conditional_ | Excluded (>24 GB MLX) | Frozen-vision Stage 1 SFT (Stage 2 SMPO only if expert-shard reload is proven) pending Dec 1 EP=8 memory & serving check |
