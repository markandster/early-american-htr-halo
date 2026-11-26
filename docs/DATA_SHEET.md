# Public Corpus & Split Data Sheet

This data sheet documents the exact public sources, parsing rules, split construction, and reporting limits implemented in `early-american-htr-halo`.

---

## 1. Rights-Cleared Corpus Inventory

All ingestion passes through [`verify_source_license`](../src/eahtr/corpus.py), restricting sources strictly to `CC-BY-4.0`, `CC0-1.0`, and `Public Domain`. Non-commercial (`CC-BY-NC*`), restricted, or unverified third-party archives are rejected by construction.

| Source | Identifier & Integrity Check | License | Competition Role | Scope Summary |
|---|---|---|---|---|
| **Revolutionary City HTR Corpus** | Zenodo `15776323` (`9,988,850,095` bytes; MD5 `c291f2ebda6e5698f788a06b9520c709`) | `CC-BY-4.0` | **Training, dev (`val`), and frozen `test_public`** | `3,316` paired stems (`3,236` `.jpg` + `80` `.png`) across 4 institutions (`apsrevcity`, `hsprevcity`, `lcprevcity`, `mss`) |
| **Library of Congress *By the People*** | `LCCN 2020446971` | `Public Domain` | **Evaluation-only genre-shift challenge (`challenge_loc`)**, off by default (`INCLUDE_LOC_BY_THE_PEOPLE = False`) | `593` rows total; first `150` filtered rows span `3` item IDs (military interrogations, receipts, farm records) |

---

## 2. Parsing, Cleaning & Hard-Case Routing

- **Archive integrity (`src/eahtr/colab_cells_corpus.py`, `src/eahtr/revcity_range.py`):** Verifies full-archive byte size and `REVCITY_ZENODO_MD5` (retrying mismatches); HTTP Range reads verify `429`/5xx backoff and `Content-Range` consistency.
- **ALTO reading order (`eahtr.corpus.parse_alto_xml_lines`):** Preserves annotated ALTO reading order by default (`sort_regions=False`, with opt-in `sort_regions=True` available for geometric ablation).
- **Diplomatic cleaning (`eahtr.hygiene.clean_diplomatic_target`):** Retains both `raw_text` and cleaned `text`; normalizes long-s (`ſ → s`), strips `^` carets, removes crossed-out `[...]` spans cleanly without corrupting intra-word damage tokens (`m[_]ung`), drops standalone `-` artifact lines, and preserves case, punctuation, period orthography, and physical line breaks (`\n`).
- **Two-page openings (`eahtr.corpus.detect_two_page_opening`):** Flags `Main A`/`Main B` ALTO tags and wide multi-page spreads for stratified reporting.
- **Hard-case routing:** Empty-ALTO pages (`16` RevCity pages tagged `has_empty_alto_text` and `ground_truth_status="missing_or_unverified_gt"`, never labeled blank without visual adjudication), leader-dot pages, and duplicate-line pages route to `hard_case_panel.jsonl` and strictly non-train/non-test `hard_case_held_out.jsonl` (`8` sparse `<= 7`-word diagnostic rows).

---

## 3. Grouping, Overrides & Verified Split Counts

RevCity file stems are numeric portal page IDs rather than document boundaries. Splits are constructed deterministically via:

1. **Item-run & creator grouping (`assign_revcity_item_groups(max_id_gap=30)`):** Groups consecutive numeric IDs within gap `<= 30` per institution, hyphenated manuscript stems (`mss_*`), and optional `PORTAL_CREATOR_MAP_PATH` entries.
2. **Collection overrides (`REVCITY_COLLECTION_OVERRIDES` in [`src/eahtr/candidates.py`](../src/eahtr/candidates.py)):** Keeps multi-gap archival runs undivided inside `train`:
   - `apsrevcity` `16255–16706` — Richard Henry Lee correspondence (`81` pages)
   - `apsrevcity` `7228–7545` — US Lottery managers' letters (`166` pages)
   - `apsrevcity` `14933–15244` — Gratz / Croghan / Simon / Philadelphia merchants correspondence (`152` pages)
3. **Transitive text-overlap closure:** Union-find merges any remaining groups linked by `>= 8`-word exact matches or `32`-word n-grams before split assignment.
4. **Balanced group allocation (`assign_balanced_group_splits`):** With `VAL_FRACTION = 0.05`, `TEST_FRACTION = 0.05`, and `max_eval_group_pages = 40`, large volumes (`> 40` pages) go to `train` while small groups (`<= 40` pages) alternate across institutions between `val` and `test_public`.

### Active RevCity Split Breakdown (`3,145` Active Pages)

| Split | Pages (% Active) | Groups | `apsrevcity` | `hsprevcity` | `lcprevcity` | `mss` | Notes |
|---|---:|---:|---:|---:|---:|---:|---|
| **`train`** | `2,897` (`92.1%`) | `17` | `1,424` | `1,303` | `124` | `46` | All `17` large volumes (`> 40` pages) |
| **`val`** | `124` (`3.9%`) | `21` | `83` | `31` | `4` | `6` | Development tuning & checkpoint selection only |
| **`test_public`** | `124` (`3.9%`) | `21` | `94` | `10` | `16` | `4` | Frozen until single post-freeze evaluation |

- **Disjointness audits:** All pairwise split audits (`train` vs. `val`, `train` vs. `test_public`, `val` vs. `test_public`) record **`0` collisions** across `document_id`, `writer_key`, `image_sha256`, `>= 8`-word normalized exact text, and `32`-word SHA-256 n-grams, plus advisory perceptual `dHash` review candidates (`Hamming <= 4`). `0` `val`/`test_public` pairs fall within `25` portal IDs.
- **Out-of-fold Stage 2 split hygiene:** Stage 2 preference rollouts use page-balanced `document_id`/`writer_key` cross-validation folds (`_assign_balanced_folds`) via `ROLLOUT_SOURCE = 'oof_train'` so Stage 2 never builds `rejected` pairs from memorized in-sample pages or eval/test rows.

---

## 4. Known Limitations & Reporting Rules

1. **Document-group disjoint only by default (`attributed_hands = 0`):** Unless `PORTAL_CREATOR_MAP_PATH` is provided, RevCity metadata lacks scribe/hand labels. Do **not** claim unseen-writer generalization on default RevCity splits; report them strictly as **document-group disjoint in-collection** evaluations, paired with `challenge_loc` for out-of-collection shift.
2. **Eval split shape & small-group exhaustion:** All `42` small groups (`<= 40` pages, `248` pages total) are allocated across `val` (`124` pages, `3.9%`) and `test_public` (`124` pages, `3.9%`), falling slightly under the `5%` target because small groups are exhausted while `train` holds the `17` large volumes.
3. **Non-APS per-institution slices are not standalone reportable:** Outside `apsrevcity`, per-institution sample counts are too small (`HSP test=10`, `LCP val=4`, `MSS val=6 / test=4`) to support standalone per-institution claims.
4. **Deleted/damaged spans & reference slips:** Roughly `39%` of sampled RevCity pages contain `[...]` crossed-out deletions or damage (not added glosses), so zero-shot baselines report `--dual-convention` against both `raw_text` and cleaned `text`. An `8`-page (`1,687`-word) spot check of RevCity ground truth confirmed `8` transcription slips (`20` more probable), including `2` duplicated lines.
5. **Sparse hard-case held-out panel:** `hard_case_held_out.jsonl` contains `8` sparse (`<= 7`-word) non-train/non-test diagnostic rows and should be treated strictly as a sanity panel, not an aggregate CER benchmark.
