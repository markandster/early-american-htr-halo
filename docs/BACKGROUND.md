# Pre-Competition Prior Work & Empirical Motivation

> **Clean-Room Boundary & Benchmark Separation**
>
> No weights, adapters, scans, transcriptions, or model-generated labels from earlier work are used. Competition training reads only the Revolutionary City corpus (`CC-BY-4.0`, Zenodo `15776323`) and starts from the published Apache-2.0 base checkpoints at pinned revisions (`google/gemma-4-12B-it@43c9d014b231`, conditional `thinkingmachines/Inkling-2-Small@885d78bfca61`); Library of Congress *By the People* pages (`Public Domain`, `LCCN 2020446971`) are used for evaluation only and are off by default. Earlier experiments—some of which trained on restricted-licence collections—informed the design and nothing else.
>
> - **All prior fine-tuning was supervised (SFT):** zero preference or RL runs existed before the competition.
> - **Both competition base models are untested on handwriting:** zero prior results exist for `google/gemma-4-12B-it` or `thinkingmachines/Inkling-2-Small`, no fine-tuned checkpoint has been converted to MLX or benchmarked locally yet, and none of the historical numbers below belong in [results/README.md](../results/README.md).
> - **Heuristic provenance disclosure:** a few reward heuristics (two boilerplate phrases and `5` of the `12` built-in historical spelling pairs) were motivated by failures seen in earlier private evaluations; zero scans, transcriptions, labels, or weights from those evaluations enter this repository.

---

## Number Categories (Never Ranked Together)

Because mixing incompatible evaluation setups misleads readers, every historical figure below carries its category:

- **Kind S:** Collection-trained optical line recognizer evaluated on cropped line images.
- **Kind Z:** Untuned general VLM reading whole manuscript pages zero-shot.
- **Kind F:** Fine-tuned general VLM reading whole manuscript pages.

*Strict* metrics preserve case, punctuation, and original period spelling. *Lenient* metrics ignore case and punctuation. Percentages are reported as **WER / CER** (lower is better).

---

## Executive Motivation: Lead with the Transfer Regression

Earlier validation runs across three VLM families plus deranged-label sham controls (>500% WER across `4B` and `7B` runs) supported image-grounded learning on held-out writers sampled from a collection represented in training. **Yet when the selected epoch-1 4B model was evaluated on a separate 6-writer panel (`17` sides, `3,778` words), error rates reversed sharply from `21.65 / 10.61` to `36.26 / 12.53` (`818 → 1,370` word edits).**

Inspecting saved outputs traced the `+552` word-edit spike to specific failure modes: one page emitted **`655` consecutive standalone `-` lines** after a line-end hyphen, bracketed markup from training labels rose from **`0` to `11`**, and one page truncated (`1,015 → 676` characters)—masking genuine gains where invented boilerplate dropped `4/17 → 0/17` pages and modernizations fell `49 → 37` of `223` unambiguous forms. Every safeguard in `early-american-htr-halo` answers one of those observed mechanisms.

---

## Prior Fine-Tuning Results (Kind F, plus Kraken Kind S)

*Note on training scope:* Earlier full-resolution SFT runs trained on `3,738` pages (`759,047` words): `3,204` Revolutionary City (`CC-BY-4.0`) pages plus `534` pages from two restricted-licence scholarly collections. None of those restricted pages, labels, or adapters carry over into `early-american-htr-halo`.

| Kind | Prior Model & Training Setup | Evaluation Scope & Sample | Base WER / CER (%) | Tuned WER / CER (%) |
|---|---|---|---|---|
| **F** | **`olmOCR-2-7B` small language-only adapter (vision frozen):** rank-4 LoRA on last 4 decoder blocks (`180,224` params), `23` RevCity training pages, `46` steps, images shrunk to `896 px` | 4 previously examined held-out writers (1 first page each, `1,601` words) | `9.74 / 6.38` | `9.68 / 6.33` *(variant: `9.87 / 6.43`; `150 → 153` word edits with joined hyphens; `1/9` spelling probes kept before/after — **no gain**)* |
| **F** | **`LightOnOCR-3-4B` full-res SFT:** BF16, up to `2048×2048`, full vision merger (`27.3M` params) + language LoRA `r=16` (`21.2M` params), `3,738` pages, `936` steps, `2` epochs, `0` OOMs | 24 validation sides (`9` writers absent from training, **same collection** as part of training, `2,506` words) | `23.78 / 9.76` *(lenient `16.59 / 7.53`)* | **Ep 1:** `20.59 / 8.51` *(lenient `13.53 / 6.46`, `8/9` better)*<br>**Ep 2:** `19.39 / 8.65` *(lenient `13.41 / 6.69`, `9/9` better)* |
| **F** | **`olmOCR-2-7B` full-res SFT:** same recipe (LoRA `r=16` + full vision merger) | Same 24 validation sides | `30.73 / 12.97` *(lenient `19.40 / 9.74`)* | `21.27 / 8.51` at `1,512` pages *(lenient `13.49 / 6.34`; no further gain at `2,016` / `2,520`; `9/9` better)* |
| **F** | **`Qwen3-VL-8B` full-res SFT:** same recipe | Same 24 validation sides | `27.02 / 12.26` | `25.62 / 10.88` at `504` pages (`6/9` better), then **collapsed to `67.60 / 41.19` at `1,512` pages** |
| **F** | **Deranged-label sham controls (`504` pages):** identical recipe with mismatched transcripts | Same 24 validation sides | `—` | `517.04 / 498.74` (4B vs. real `21.31 / 8.80`)<br>`542.66 / 523.88` (7B vs. real `24.02 / 10.11`) |
| **F** | **`LightOnOCR-3-4B` (same epoch-1 checkpoint):** independent transfer test, never used for selection | Separate panel: `6` letters by `6` unseen writers (`17` sides, `3,778` words) | `21.65 / 10.61` *(lenient `15.89 / 9.17`)* | **`36.26 / 12.53`** *(lenient `18.26 / 11.68` — **regression**)* |
| **S** | **Kraken RevCity optical reader:** full-network supervised adaptation on `7,311` human-aligned lines *(including training lines from a restricted-licence collection)* | Within-collection test: `10` complete held-out letters by `10` unseen writers (`34` sides, `3,956` words, automatic lines) | `39.69 / 19.78` | `34.98 / 17.76` *(all `10/10` letters improved; output-layer-only reached `37.69 / 19.00`)* |

---

## Untuned Public-Data Reference Points (Kind Z & Kind S)

- **Untuned local `35B` open VLM (`Z`) on 40 hash-selected RevCity pages** *(public training corpus, not an unseen-writer score)*: `76.55%` strict WER overall; `26.09% / 10.94%` without `2` runaway replies; `25.63% / 10.38%` also excluding `2` false blanks (`36` pages, `7,391` words; lenient `15.62% / 7.57%`). Removing raised-letter carets shifted that `36`-page figure from `25.63 / 10.38` to `24.26 / 9.66`.
- **Same untuned local `35B` open VLM (`Z`) on 60 Library of Congress *By the People* pages:** `157.52%` WER overall; `42.46% / 28.35%` without `3` runaway table-loop replies.
- **Published Apache-2.0 RevCity Kraken line recognizer (`S`, trained from scratch on RevCity):** `6.84% WER / 1.70% CER` on `4` RevCity pages given publisher line polygons (pages chosen after scoring and likely in its training split), `14.64% / 4.33%` on those same `4` pages with automatic line detection, and `39.69% / 19.78%` on `10` complete held-out letters by unseen writers. This is a collection-trained line recognizer and is not comparable to whole-page general VLMs.

---

## Methodological Caveats

- **Never rank across `S`, `Z`, and `F` or across separate panels:** The 4-page, 24-side, 17-side, 34-side, 40-page, and 60-page evaluations differ in reader kind, holdings, and text conventions.
- **Strict vs. lenient divergence on loops:** On the 17-side unseen-writer regression, `LightOnOCR-3-4B` lost `14.61` strict WER points vs. `2.37` lenient WER points because the `655` looped `-` lines are punctuation.
- **RevCity brackets mark deletions/damage, not additions:** About `39%` of sampled RevCity pages contain `[...]` spans, which represent crossed-out ink or physical damage rather than added editorial glosses.
- **Reference noise in public ground truth:** A manual check of `8` RevCity pages (`1,687` words) found `8` confirmed transcription slips (and `20` more probable), including `2` duplicated lines.
- **Meaning of "held-out writers":** The 24-side validation set holds out writers from a collection represented in the `534`-page restricted slice of those prior runs; only the 17-side 6-writer panel was independent. Between the frozen-vision 4-page experiment and the full-resolution VLM runs, data scale, resolution, LoRA rank, and vision trainability all changed together—no single-factor ablation isolated their individual shares.

---

## Observed Failures → Safeguards Built into `early-american-htr-halo`

> **Execution status:** All safeguards below exist in code and unit tests at `HEAD`. A live 24 GB Colab L4 4-bit QLoRA 5-step Halo smoke passed `VLMDataCollator` batch-0 (`1,092` image tokens), 5-step `sft.py` (with trainable vision LoRA updates), `merge_peft_adapters.py` + `verify_mlx_export.py`, natural `<turn|>` termination, Stage 1→2 `SMPOTrainer` step-0 weight continuity (`0.0` exact match under matching FP32-norm + `peft_bf16_autocast` precision), and 5-step `smpo.py`. **An 80 GB BF16 smoke (`>= 56 GiB`), Halo-native matched-compute sham controls, and all full-corpus Gemma 4 `12B` validation accuracy metrics remain unmeasured pending GPU execution.**

| Failure in Prior Runs | Evidence (Public-Safe) | Fix in `early-american-htr-halo` | Implementation Status |
|---|---|---|---|
| **Repetition loops** | One fresh page emitted `655` consecutive `-` lines after a line-end hyphen; untuned models looped on tables/lists. | `clean_diplomatic_target` drops hyphen-only target lines; `filter_corpus_record` excludes leader-dot pages; `detect_repetition_loop` assigns `-1.0` for `>= 4` repeated lines/dash tokens, `>= 5` n-gram cycles, or `finish_reason == "length"`; `merge_and_eval_halo.py` supports `--repetition-penalty`. | Code & tests in [`rewards.py`](../src/eahtr/rewards.py), [`hygiene.py`](../src/eahtr/hygiene.py), [`merge_and_eval_halo.py`](../scripts/merge_and_eval_halo.py). Reward applies in conditional Stage 2 SMPO. |
| **Token-cap non-termination** | Dense pages hit `2,048` / `6,144` caps with runaway or empty output. | Reference-scaled decode cap `recommended_max_new_tokens` (`ceil(1.5 × ref_tokens)`, minimum `2,048`) plus `-1.0` length-cap floor. | Code & tests in [`rewards.py`](../src/eahtr/rewards.py). |
| **Bracket markup leakage** | Bracket spans on 17 fresh sides rose `0 → 11` after tuning. | `[...]` stripped from clean targets while preserving `raw_text`; deletion-aligned prompt; `count_editorial_leaks` (`-0.10` penalty); `--dual-convention` zero-shot reporting. | Code & tests in [`hygiene.py`](../src/eahtr/hygiene.py), [`rewards.py`](../src/eahtr/rewards.py). |
| **Modernized spelling** | Modernizations fell `49 → 37` on `223` historical forms while exact readings stayed flat at `150`. | Prompt constraints + `count_modernizations` (`-0.05` penalty over `12` built-in pairs), tracking exact spelling retention separately. | Code & tests in [`rewards.py`](../src/eahtr/rewards.py) *(corpus-derived expansion list not built)*. |
| **Frozen-vision plateau** | Small rank-4 frozen-vision adapter gained nothing (`9.74 → 9.68` / `9.87` WER). | Colab fails closed if `0` vision modules are trainable (`resolve_vision_modules_to_save`); Gemma 4 processor staged at `1,120` soft tokens instead of `280`. | Colab path confirmed in tests; Halo GPU smoke must prove `all-linear` LoRA updates vision weights (`Inkling-2-Small` is frozen-vision by design). |
| **Line-break & order drift** | Reading order and physical line breaks drift on multi-block pages while strict CER collapses whitespace. | Separate `newline_cer_pct`, `line_boundary_f1_pct`, and `order_tolerant_cer_pct`; annotated ALTO order preserved (`sort_regions=False`); two-page openings flagged. | Code & tests in [`metrics.py`](../src/eahtr/metrics.py), [`corpus.py`](../src/eahtr/corpus.py). |
| **Split leakage** | Random line splits with supplied line polygons and likely training-set overlap inflate scores (`6.84 / 1.70` vs. `39.69 / 19.78` on unseen letters). | RevCity grouped by consecutive ID runs (`max_id_gap=30`) + collection overrides + `audit_split_disjointness` across document, writer key, SHA-256, exact text, and 32-word n-grams + separate `challenge_loc` panel. | Code & tests in [`candidates.py`](../src/eahtr/candidates.py), [`hygiene.py`](../src/eahtr/hygiene.py). Splits are document-group disjoint (`attributed_hands = 0` by default). |
| **Truncation & false blanks** | Pages truncated (`1,015 → 676` chars) or misclassified as blank versos. | `-0.25` penalty under `70%` reference length (`P_trunc`) + non-blank hallucination guard (`-1.0`). | Code & tests in [`rewards.py`](../src/eahtr/rewards.py). |
| **Archive boilerplate** | Invented image-credit text on `4/17` untuned pages. | Boilerplate watermark phrase detector in [`rewards.py`](../src/eahtr/rewards.py). | Code & tests present (inert on RevCity scans without embedded credit captions). |
| **Visual grounding checks** | Shuffled controls and swap checks distinguish image reading from text memorization. | `make_sham_derangement` in Colab + `--swap-images` in `merge_and_eval_halo.py`. | Colab sham & swap wired; Halo-native matched-compute sham planned for competition runs. |
