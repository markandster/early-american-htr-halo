# Public Competition Benchmark Results

All metrics below are measured exclusively on the **rights-cleared held-out public evaluation split** (`test_public`, `CC-BY-4.0` / `Public Domain` only, writer/series-disjoint and verified against document, writer, image SHA-256, normalized full-page text hash, and 32-word n-gram overlap).

Every run reports:
- **Strict WER / CER:** Unicode NFC, collapsed whitespace, preserving case, punctuation, period spelling, and line breaks.
- **Lenient WER / CER:** Lowercased and punctuation-stripped.
- **Decoding health:** Runaway repetition loop pages (`loop_pages`), token-cap non-terminations (`non_terminated_pages`), unprompted `[...]` / watermark leaks (`editorial_leaks`), archaic-spelling modernizations (`modernizations`), silent thinking leaks (`thinking_leak_pages`), and mean deterministic reward (`eahtr.rewards`).

## Leaderboard & Ablation Tracker

| Model | Stage / Arm | Public Split Scope | Strict WER / CER | Lenient WER / CER | Loop / Non-Term Pages | Mean Reward | M3 Max MLX Peak RAM | Notes |
|---|---|---|---|---|---|---|---|---|
| `google/gemma-4-12B-it` | Zero-shot baseline | `revcity-test_public` | _pending_ | _pending_ | _pending_ | _pending_ | — | Ungated; 1120 vision tokens, `sdpa` |
| `google/gemma-4-12B-it` | Matched-compute sham | `revcity-test_public` | _pending_ | _pending_ | _pending_ | _pending_ | — | Proves real visual learning |
| `google/gemma-4-12B-it` | Stage 1 VLM SFT (Halo / Colab) | `revcity-test_public` | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ | Non-thinking parity template, trainable vision |
| `google/gemma-4-12B-it` | Stage 2 VLM SMPO (Halo Primary) | `revcity-test_public` | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ | Smooth lower-bound margin + SFT anchor |
| `google/gemma-4-12B-it` | Stage 2 VLM DPO (Halo Ablation) | `revcity-test_public` | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ | PEFT implicit reference ablation |
| `google/gemma-4-26B-A4B-it` | Stage 1 SFT + Stage 2 VLM SMPO | `revcity-test_public` | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ | 26B/4B-active MoE (`EP=8`, explicit expert LoRA) |
| `thinkingmachines/Inkling-2-Small` | Zero-shot baseline | `revcity-test_public` | _pending_ | _pending_ | _pending_ | _pending_ | — | Track 2 Primary (frozen NVFP4/MXFP8, `EP=8`) |
| `thinkingmachines/Inkling-2-Small` | Stage 1 SFT + Stage 2 VLM SMPO | `revcity-test_public` | _pending_ | _pending_ | _pending_ | _pending_ | — | Zero-effort template; `export-engine-adapter` serving |
| `thinkingmachines/Inkling-2-Mini` | Blocked reference | `revcity-test_public` | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ | Gated on upstream `inkling2_loop` loader support |
