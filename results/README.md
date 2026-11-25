# Public Competition Benchmark Results

All metrics below are measured exclusively on the **rights-cleared held-out public evaluation split** (`test_public`, `CC-BY-4.0` / `Public Domain` only, writer/series-disjoint and verified against 32-word n-gram and image SHA-256 overlap).

Every run reports:
- **Strict WER / CER:** Unicode NFC, collapsed whitespace, preserving case, punctuation, period spelling, and line breaks.
- **Lenient WER / CER:** Lowercased and punctuation-stripped.
- **Decoding health:** Runaway repetition loop pages (`loop_pages`), token-cap non-terminations (`non_terminated_pages`), unprompted `[...]` / watermark leaks (`editorial_leaks`), archaic-spelling modernizations (`modernizations`), and mean deterministic reward (`eahtr.rewards`).

## Leaderboard & Ablation Tracker

| Model | Stage / Arm | Public Split Scope | Strict WER / CER | Lenient WER / CER | Loop / Non-Term Pages | Mean Reward | M3 Max MLX Peak RAM | Notes |
|---|---|---|---|---|---|---|---|---|
| `google/gemma-4-12B-it` | Zero-shot baseline | `revcity-test_public` | _pending_ | _pending_ | _pending_ | _pending_ | — | Pre-Dec 1 Colab smoke/baseline |
| `google/gemma-4-12B-it` | Matched-compute sham | `revcity-test_public` | _pending_ | _pending_ | _pending_ | _pending_ | — | Proves real visual learning |
| `google/gemma-4-12B-it` | Stage 1 SFT (Halo / Colab) | `revcity-test_public` | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ | Full-res (`2048x2048`), unfrozen vision |
| `google/gemma-4-12B-it` | Stage 2 GRPO (Halo RLVR) | `revcity-test_public` | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ | Targets Dec 1–10 submission |
| `google/gemma-4-26B-A4B-it` | Stage 1 SFT + Stage 2 GRPO | `revcity-test_public` | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ | 26B/4B-active MoE candidate |
| `thinkingmachines/Inkling-2-Mini` | Zero-shot baseline | `revcity-test_public` | _pending_ | _pending_ | _pending_ | _pending_ | — | Ungated Apache-2.0 looped MoE |
| `thinkingmachines/Inkling-2-Mini` | Stage 1 SFT + Stage 2 GRPO | `revcity-test_public` | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ | Targets non-termination fix via RLVR |
