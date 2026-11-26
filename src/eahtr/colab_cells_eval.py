"""Evaluation helpers and Cells 6-8 builders for rollouts and MLX export."""

from __future__ import annotations

CELL_5_EVAL_SOURCE = """
def score_predictions_verifiable(
    refs: list[str],
    preds: list[str],
    finish_reasons: list[str],
    raw_refs: list[str] | None = None,
) -> dict:
    sw = sW = sc = sC = lw = lW = lc = lC = 0
    nl_edits = nl_chars = lb_tp = lb_p = lb_r = ot_edits = ot_chars = 0
    rewards, loops, budget_capped, non_terms, leaks, mods = [], 0, 0, 0, 0, 0
    for ref, pred, fr in zip(refs, preds, finish_reasons):
        pair = score_pair(pred, ref)
        st, ln = pair['strict'], pair['lenient']
        sw += st.word_edits; sW += st.ref_words
        sc += st.char_edits; sC += st.ref_chars
        lw += ln.word_edits; lW += ln.ref_words
        lc += ln.char_edits; lC += ln.ref_chars
        lb = score_line_breaks(pred, ref)
        nl_edits += int(lb['newline_char_edits'])
        nl_chars += int(lb['newline_ref_chars'])
        lb_tp += int(lb['matched_boundaries'])
        lb_p += int(lb['pred_lines'])
        lb_r += int(lb['ref_lines'])
        ot_edits += int(lb['order_tolerant_char_edits'])
        ot_chars += int(lb['order_tolerant_ref_chars'])
        ver = score_transcription(pred, ref, finish_reason=fr)
        rewards.append(ver.reward)
        loops += int(ver.loop_detected)
        if fr == 'length':
            budget_capped += 1
        non_terms += int(ver.non_terminated)
        leaks += ver.editorial_leak_count
        mods += ver.modernizations

    assert sw == sum(Levenshtein.distance(normalize_strict(r).split(), normalize_strict(p).split()) for r, p in zip(refs, preds))
    assert sc == sum(Levenshtein.distance(normalize_strict(r), normalize_strict(p)) for r, p in zip(refs, preds))
    lb_f1 = (2.0 * lb_tp / (lb_p + lb_r)) if (lb_p + lb_r) else 0.0
    out = {
        'strict_word_edits': sw,
        'strict_words': sW,
        'strict_wer_pct': round(100.0 * sw / sW, 2) if sW else 0.0,
        'strict_char_edits': sc,
        'strict_chars': sC,
        'strict_cer_pct': round(100.0 * sc / sC, 2) if sC else 0.0,
        'lenient_wer_pct': round(100.0 * lw / lW, 2) if lW else 0.0,
        'lenient_cer_pct': round(100.0 * lc / lC, 2) if lC else 0.0,
        'newline_cer_pct': round(100.0 * nl_edits / nl_chars, 2) if nl_chars else 0.0,
        'line_boundary_f1_pct': round(100.0 * lb_f1, 2),
        'order_tolerant_cer_pct': round(100.0 * ot_edits / ot_chars, 2) if ot_chars else 0.0,
        'mean_verifiable_reward': round(sum(rewards) / max(1, len(rewards)), 4),
        'loop_pages': loops,
        'budget_capped_pages': budget_capped,
        'non_terminated_pages': non_terms,
        'editorial_leaks': leaks,
        'modernizations': mods,
        'independent_recount_verified': True,
    }
    if raw_refs:
        rc = sum(Levenshtein.distance(normalize_strict(rr), normalize_strict(p)) for rr, p in zip(raw_refs, preds))
        rC = sum(len(normalize_strict(rr)) for rr in raw_refs)
        out['raw_strict_cer_pct'] = round(100.0 * rc / rC, 2) if rC else 0.0
    return out


def _detect_thinking_leak(raw_generated: str) -> bool:
    return any(tag in raw_generated for tag in ('<|channel>thought', '<|content_thinking|>', '<think>'))


@torch.no_grad()
def run_eval_panel(
    model,
    processor,
    rows: list[dict],
    panel_name: str,
    max_new_tokens: int | None = None,
    *,
    swap_images: bool = False,
) -> dict:
    model.eval()
    preds, refs, raw_refs, finish_reasons, per_side = [], [], [], [], []
    thinking_leak_pages = 0
    n_rows = len(rows)
    for idx, row in enumerate(tqdm(rows, desc=f'Eval {panel_name}', leave=False)):
        img_path = rows[(idx + 1) % n_rows]['image_path'] if (swap_images and n_rows > 1) else row['image_path']
        image = Image.open(img_path).convert('RGB')
        messages = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': DIPLOMATIC_PROMPT}]}]
        prompt = _apply_chat_nothink(processor, messages, add_generation_prompt=True)
        inputs = processor(text=[prompt], images=[image], return_tensors='pt').to(model.device)
        cap = int(max_new_tokens or MAX_EVAL_NEW_TOKENS)
        t0 = time.time()
        out_ids = model.generate(**inputs, max_new_tokens=cap, do_sample=False)
        elapsed = round(time.time() - t0, 2)
        gen_ids = out_ids[:, inputs['input_ids'].shape[1]:]
        fr = 'length' if int(gen_ids.shape[1]) >= cap else 'stop'
        raw_gen = processor.batch_decode(gen_ids, skip_special_tokens=False)[0]
        if _detect_thinking_leak(raw_gen):
            thinking_leak_pages += 1
        gen = processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
        ver = score_transcription(gen, row['text'], finish_reason=fr)
        lb = score_line_breaks(gen, row['text'])
        preds.append(gen)
        refs.append(row['text'])
        raw_refs.append(row.get('raw_text', row['text']))
        finish_reasons.append(fr)
        per_side.append({
            'record_id': row['record_id'],
            'document_id': row['document_id'],
            'elapsed_s': elapsed,
            'max_new_tokens': cap,
            'finish_reason': fr,
            'reward': ver.reward,
            'strict_cer': ver.cer_strict,
            'newline_cer': lb['newline_cer'],
            'order_tolerant_cer': lb['order_tolerant_cer'],
            'prediction': gen,
            'reference': row['text'],
        })
    scores = score_predictions_verifiable(refs, preds, finish_reasons, raw_refs=raw_refs)
    scores['thinking_leak_pages'] = thinking_leak_pages
    return {'panel': panel_name, 'sides': len(rows), 'scores': scores, 'predictions': per_side}


def _assert_highres_processor_config(model_dir: Path, hf_repo: str) -> None:
    jinja_path = model_dir / 'chat_template.jinja'
    tok_cfg_path = model_dir / 'tokenizer_config.json'
    tpl = jinja_path.read_text(encoding='utf-8') if jinja_path.is_file() else ''
    if not tpl.strip() and tok_cfg_path.is_file():
        tok_cfg = json.loads(tok_cfg_path.read_text(encoding='utf-8'))
        raw = tok_cfg.get('chat_template')
        tpl = raw if isinstance(raw, str) else ''
    assert tpl.strip(), f'{model_dir}: missing non-empty chat_template in merged export'
    assert 'else 0.9' not in tpl, f'{model_dir}: Inkling template defaults to 0.9 thinking effort'
    if 'gemma-4' not in hf_repo.lower():
        return
    cfg_path = model_dir / 'processor_config.json'
    assert cfg_path.is_file(), f'Missing required {cfg_path} in merged Gemma 4 export'
    data = json.loads(cfg_path.read_text(encoding='utf-8'))
    assert data.get('image_seq_length') == 1120, f'{cfg_path}: expected image_seq_length==1120'
    ip = data.get('image_processor', {})
    assert ip.get('max_soft_tokens') == 1120, f'{cfg_path}: expected max_soft_tokens==1120'
    if 'image_seq_length' in ip:
        assert ip['image_seq_length'] == 1120, f'{cfg_path}: expected image_processor.image_seq_length==1120'

"""

CELL_6_RUN_SOURCE = """
# === Step 3: Dev Validation Only (test_public stays frozen until final eval) ===
val_slice = val_rows[:VAL_EVAL_MAX_PAGES] if VAL_EVAL_MAX_PAGES else val_rows
pilot_train_rows = train_rows[:SHAM_MAX_PRESENTATIONS]
sham_rows = make_sham_derangement(pilot_train_rows, seed=SEED)
dev_test_slice = test_public_rows if FINAL_FREEZE_EVAL_TEST_PUBLIC else []
eval_slice = dev_test_slice if (FINAL_FREEZE_EVAL_TEST_PUBLIC and dev_test_slice) else val_slice

ALL_RESULTS = {}
for cand_key in CANDIDATE_KEYS:
    ALL_RESULTS[cand_key] = {}
    for arm in ARMS:
        arm_rows = sham_rows if arm == 'sham_short' else (
            pilot_train_rows if (arm == 'real' and SMOKE_TEST_MODE) else train_rows
        )
        ALL_RESULTS[cand_key][arm] = run_candidate_arm(cand_key, arm, arm_rows, val_slice, dev_test_slice)

summary = []
for cand_key, arms in ALL_RESULTS.items():
    z_res = arms.get('baseline_zeroshot', {})
    r_res = arms.get('real', {})
    s_res = arms.get('sham_short', {})
    z, r, s = z_res.get('scores', {}), r_res.get('scores', {}), s_res.get('scores', {})
    ci_vs_base = None
    if z_res.get('per_side_cer') and r_res.get('per_side_cer'):
        diffs = [rv - zv for rv, zv in zip(r_res['per_side_cer'], z_res['per_side_cer'])]
        eval_sub = eval_slice[:len(diffs)]
        cids = [row['document_id'] for row in eval_sub]
        wts = [max(1, len(normalize_strict(row['text']))) for row in eval_sub]
        ci_vs_base = paired_bootstrap_ci(diffs, cids, weights=wts, seed=SEED)
    summary.append({
        'candidate': cand_key,
        'eval_split': 'test_public' if FINAL_FREEZE_EVAL_TEST_PUBLIC else 'val',
        'zeroshot_wer_cer': f"{z.get('strict_wer_pct')}% / {z.get('strict_cer_pct')}%" if z else 'n/a',
        'zeroshot_raw_cer': z.get('raw_strict_cer_pct'),
        'sham_wer_cer': f"{s.get('strict_wer_pct')}% / {s.get('strict_cer_pct')}%" if s else 'n/a',
        'real_wer_cer': f"{r.get('strict_wer_pct')}% / {r.get('strict_cer_pct')}%" if r else 'n/a',
        'real_newline_f1': r.get('line_boundary_f1_pct'),
        'real_order_tol_cer': r.get('order_tolerant_cer_pct'),
        'real_swap_cer_delta': r_res.get('swap_cer_degradation'),
        'ci_real_minus_zeroshot_cer': ci_vs_base,
        'real_reward': r.get('mean_verifiable_reward'),
        'real_loops': r.get('loop_pages'),
    })
(OUTPUT_ROOT / 'colab_val_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
print(json.dumps(summary, indent=2))
"""

CELL_7_GRPO_SOURCE = """
# === Optional: Generate & Score Stage 2 Rollouts for Halo VLM SMPO (chosen_sft_ratio: 1.0) ===
if GENERATE_OFFLINE_GRPO_ROLLOUTS:
    cand_key = CANDIDATE_KEYS[0]
    spec = COMPETITION_CANDIDATES[cand_key]
    real_res = ALL_RESULTS.get(cand_key, {}).get('real', {}) if 'ALL_RESULTS' in globals() else {}
    merged_dir = Path(real_res['merged_hf_dir']) if real_res.get('merged_hf_dir') else (OUTPUT_ROOT / f'{cand_key}-real' / 'merged_hf')
    adapter_dir = Path(real_res['adapter_dir']) if real_res.get('adapter_dir') else (OUTPUT_ROOT / f'{cand_key}-real' / 'adapter_best')
    assert merged_dir.is_dir() or adapter_dir.is_dir(), f'Train Stage 1 real checkpoint first under {OUTPUT_ROOT}'

    processor = load_processor_fullres({'hf_repo': spec.hf_repo, 'max_pixels': spec.max_pixels, 'trust_remote_code': spec.trust_remote_code})
    rollout_4bit = should_use_4bit(spec, VRAM_GB, 'baseline_zeroshot')
    if rollout_4bit and spec.quant_method == 'modelopt':
        raise RuntimeError(f'{spec.hf_repo} uses {spec.quant_method} and cannot use 4-bit bitsandbytes rollouts.')
    load_target = str(merged_dir) if merged_dir.is_dir() else spec.hf_repo
    rollout_load_kwargs = {'device_map': 'auto', 'torch_dtype': torch.bfloat16, 'trust_remote_code': spec.trust_remote_code}
    if 'gemma-4' in spec.hf_repo.lower():
        rollout_load_kwargs['attn_implementation'] = 'sdpa'
    if rollout_4bit:
        rollout_load_kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            llm_int8_skip_modules=list(spec.vision_trainable_patterns),
        )
    base_model = load_vlm_model(load_target, rollout_load_kwargs)
    model = base_model if merged_dir.is_dir() else PeftModel.from_pretrained(base_model, str(adapter_dir))
    model.eval()

    ROLLOUT_TEMPS = (0.3, 0.7, 0.7, 1.0)
    rollout_rows = train_rows[:GRPO_ROLLOUT_PROMPTS]
    out_jsonl = OUTPUT_ROOT / f'{cand_key}_scored_offline_grpo_rollouts.jsonl'
    pref_jsonl = OUTPUT_ROOT / f'{cand_key}_vlm_preferences.jsonl'
    pref_records = []
    rollout_thinking_leaks = 0
    with out_jsonl.open('w', encoding='utf-8') as fout:
        for row in tqdm(rollout_rows, desc='VLM Stage-2 Rollouts'):
            image = Image.open(row['image_path']).convert('RGB')
            messages = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': DIPLOMATIC_PROMPT}]}]
            prompt = _apply_chat_nothink(processor, messages, add_generation_prompt=True)
            inputs = processor(text=[prompt], images=[image], return_tensors='pt').to(model.device)
            tok = getattr(processor, 'tokenizer', processor)
            cap = recommended_max_new_tokens(len(tok(row['text'], add_special_tokens=False)['input_ids']), min_tokens=MAX_EVAL_NEW_TOKENS)
            completions, finish_reasons = [], []
            for gen_idx in range(GRPO_NUM_GENERATIONS):
                temp = ROLLOUT_TEMPS[gen_idx % len(ROLLOUT_TEMPS)]
                with torch.no_grad():
                    out_ids = model.generate(**inputs, max_new_tokens=cap, do_sample=True, temperature=temp, top_p=0.9)
                gen_ids = out_ids[:, inputs['input_ids'].shape[1]:]
                fr = 'length' if int(gen_ids.shape[1]) >= cap else 'stop'
                raw_gen = processor.batch_decode(gen_ids, skip_special_tokens=False)[0]
                if _detect_thinking_leak(raw_gen):
                    rollout_thinking_leaks += 1
                completions.append(processor.batch_decode(gen_ids, skip_special_tokens=True)[0])
                finish_reasons.append(fr)
            scored = score_group_completions(completions, row['text'], finish_reasons)
            fout.write(json.dumps({
                'record_id': row['record_id'],
                'image_path': row['image_path'],
                'prompt': DIPLOMATIC_PROMPT,
                'completions': completions,
                'rewards': [float(item['reward']) for item in scored],
                'scored_completions': scored,
            }, ensure_ascii=False) + '\\n')
            pref_records.append({
                'record_id': row['record_id'],
                'document_id': row['document_id'],
                'writer_key': row['writer_key'],
                'image_sha256': row['image_sha256'],
                'image_path': row['image_path'],
                'prompt': DIPLOMATIC_PROMPT,
                'text': row['text'],
                'completions': completions,
                'finish_reasons': finish_reasons,
                'max_new_tokens_cap': cap,
                'split': 'train',
                'source_checkpoint': str(merged_dir if merged_dir.is_dir() else adapter_dir),
            })

    pref_pairs = build_preference_pairs(pref_records, min_reward_margin=0.05, prefer_gold_chosen=True)
    pref_jsonl.write_text(
        '\\n'.join(json.dumps(pair, ensure_ascii=False) for pair in pref_pairs) + ('\\n' if pref_pairs else ''),
        encoding='utf-8',
    )
    print(f'Wrote {len(pref_pairs)} gold-anchored Halo VLM preference pairs -> {pref_jsonl} (leaks={rollout_thinking_leaks})')
"""

CELL_8_MLX_SOURCE = """
# === Optional Local Quantized MLX Export (<= 24 GB Workstation Budget — 12B Only) ===
for cand_key in CANDIDATE_KEYS:
    spec = COMPETITION_CANDIDATES[cand_key]
    merged = ALL_RESULTS.get(cand_key, {}).get('real', {}).get('merged_hf_dir') or f'<merged_{cand_key}_hf_dir>'
    if spec.params_b > 15.0:
        print(f'# SKIP MLX 24 GB gate for {cand_key} ({spec.params_b}B > single-host 24 GB unified memory budget).')
        continue
    print(f'# {cand_key} ({spec.hf_repo}, est MLX peak ~{spec.est_mlx_peak_gb} GB):')
    print(f'python -m mlx_vlm convert --hf-path {merged} --mlx-path ./mlx/{cand_key}-q{spec.mlx_quant_bits} -q --q-bits {spec.mlx_quant_bits}')
    print(f'python scripts/verify_mlx_export.py ./mlx/{cand_key}-q{spec.mlx_quant_bits} --reference-dir {merged}')
"""


def build_cell_5_eval_section() -> str:
    """Return evaluation and metrics helper functions for Cell 5."""
    return CELL_5_EVAL_SOURCE


def build_cell_6_run() -> str:
    """Return Cell 6 baseline/real/sham arm execution code."""
    return CELL_6_RUN_SOURCE


def build_cell_7_rollouts() -> str:
    """Return Cell 7 multi-temperature VLM preference and GRPO rollout code."""
    return CELL_7_GRPO_SOURCE


def build_cell_8_mlx() -> str:
    """Return Cell 8 optional local quantized MLX export command generator."""
    return CELL_8_MLX_SOURCE
