"""Evaluation helpers and Cells 6-8 builders for rollouts and MLX export."""

from __future__ import annotations

CELL_5_EVAL_SOURCE = """
def score_predictions_verifiable(refs: list[str], preds: list[str], finish_reasons: list[str]) -> dict:
    sw = sW = sc = sC = lw = lW = lc = lC = 0
    rewards, loops, non_terms, leaks, mods = [], 0, 0, 0, 0
    for ref, pred, fr in zip(refs, preds, finish_reasons):
        pair = score_pair(pred, ref)
        st, ln = pair['strict'], pair['lenient']
        sw += st.word_edits; sW += st.ref_words
        sc += st.char_edits; sC += st.ref_chars
        lw += ln.word_edits; lW += ln.ref_words
        lc += ln.char_edits; lC += ln.ref_chars
        ver = score_transcription(pred, ref, finish_reason=fr)
        rewards.append(ver.reward)
        loops += int(ver.loop_detected)
        non_terms += int(ver.non_terminated)
        leaks += ver.editorial_leak_count
        mods += ver.modernizations

    # Independent rapidfuzz recount verification
    assert sw == sum(Levenshtein.distance(normalize_strict(r).split(), normalize_strict(p).split()) for r, p in zip(refs, preds))
    assert sc == sum(Levenshtein.distance(normalize_strict(r), normalize_strict(p)) for r, p in zip(refs, preds))
    return {
        'strict_word_edits': sw,
        'strict_words': sW,
        'strict_wer_pct': round(100.0 * sw / sW, 2) if sW else 0.0,
        'strict_char_edits': sc,
        'strict_chars': sC,
        'strict_cer_pct': round(100.0 * sc / sC, 2) if sC else 0.0,
        'lenient_word_edits': lw,
        'lenient_words': lW,
        'lenient_wer_pct': round(100.0 * lw / lW, 2) if lW else 0.0,
        'lenient_char_edits': lc,
        'lenient_chars': lC,
        'lenient_cer_pct': round(100.0 * lc / lC, 2) if lC else 0.0,
        'mean_verifiable_reward': round(sum(rewards) / max(1, len(rewards)), 4),
        'loop_pages': loops,
        'non_terminated_pages': non_terms,
        'editorial_leaks': leaks,
        'modernizations': mods,
        'independent_recount_verified': True,
    }


def _detect_thinking_leak(raw_generated: str) -> bool:
    return any(tag in raw_generated for tag in ('<|channel>thought', '<|content_thinking|>', '<think>'))


@torch.no_grad()
def run_eval_panel(model, processor, rows: list[dict], panel_name: str, max_new_tokens: int = MAX_EVAL_NEW_TOKENS) -> dict:
    model.eval()
    preds, refs, finish_reasons, per_side = [], [], [], []
    thinking_leak_pages = 0
    for row in tqdm(rows, desc=f'Eval {panel_name}', leave=False):
        image = Image.open(row['image_path']).convert('RGB')
        messages = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': DIPLOMATIC_PROMPT}]}]
        prompt = _apply_chat_nothink(processor, messages, add_generation_prompt=True)
        inputs = processor(text=[prompt], images=[image], return_tensors='pt').to(model.device)
        t0 = time.time()
        out_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        elapsed = round(time.time() - t0, 2)
        gen_ids = out_ids[:, inputs['input_ids'].shape[1]:]
        fr = 'length' if int(gen_ids.shape[1]) >= int(max_new_tokens) else 'stop'
        raw_gen = processor.batch_decode(gen_ids, skip_special_tokens=False)[0]
        if _detect_thinking_leak(raw_gen):
            thinking_leak_pages += 1
        gen = processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
        ver = score_transcription(gen, row['text'], finish_reason=fr)
        preds.append(gen)
        refs.append(row['text'])
        finish_reasons.append(fr)
        per_side.append({
            'record_id': row['record_id'],
            'document_id': row['document_id'],
            'elapsed_s': elapsed,
            'finish_reason': fr,
            'reward': ver.reward,
            'strict_cer': ver.cer_strict,
            'prediction': gen,
            'reference': row['text'],
        })
    scores = score_predictions_verifiable(refs, preds, finish_reasons)
    scores['thinking_leak_pages'] = thinking_leak_pages
    return {'panel': panel_name, 'sides': len(rows), 'scores': scores, 'predictions': per_side}


def _assert_highres_processor_config(model_dir: Path, hf_repo: str) -> None:
    if 'gemma-4' not in hf_repo.lower():
        return
    cfg_path = model_dir / 'processor_config.json'
    if not cfg_path.is_file():
        return
    data = json.loads(cfg_path.read_text(encoding='utf-8'))
    assert data.get('image_seq_length') == 1120, f'{cfg_path}: expected image_seq_length==1120'
    ip = data.get('image_processor', {})
    assert ip.get('max_soft_tokens') == 1120, f'{cfg_path}: expected max_soft_tokens==1120'
    if 'image_seq_length' in ip:
        assert ip['image_seq_length'] == 1120, f'{cfg_path}: expected image_processor.image_seq_length==1120'

"""

CELL_6_RUN_SOURCE = """
# === Step 3: Execute Baseline, Real Train, and Matched-Compute Sham Control ===
val_slice = val_rows[:VAL_EVAL_MAX_PAGES] if VAL_EVAL_MAX_PAGES else val_rows
test_slice = test_public_rows[:TEST_EVAL_MAX_PAGES] if TEST_EVAL_MAX_PAGES else test_public_rows
sham_rows = make_sham_derangement(train_rows, seed=SEED)[:SHAM_MAX_PRESENTATIONS]

ALL_RESULTS = {}
for cand_key in CANDIDATE_KEYS:
    ALL_RESULTS[cand_key] = {}
    for arm in ARMS:
        arm_rows = sham_rows if arm == 'sham_short' else train_rows
        ALL_RESULTS[cand_key][arm] = run_candidate_arm(cand_key, arm, arm_rows, val_slice, test_slice)

summary = []
for cand_key, arms in ALL_RESULTS.items():
    z = arms.get('baseline_zeroshot', {}).get('scores', {})
    r = arms.get('real', {}).get('scores', {})
    s = arms.get('sham_short', {}).get('scores', {})
    summary.append({
        'candidate': cand_key,
        'zeroshot_wer_cer': f"{z.get('strict_wer_pct')}% / {z.get('strict_cer_pct')}%" if z else 'n/a',
        'sham_wer_cer': f"{s.get('strict_wer_pct')}% / {s.get('strict_cer_pct')}%" if s else 'n/a',
        'real_wer_cer': f"{r.get('strict_wer_pct')}% / {r.get('strict_cer_pct')}%" if r else 'n/a',
        'real_reward': r.get('mean_verifiable_reward'),
        'real_loops': r.get('loop_pages'),
    })
(OUTPUT_ROOT / 'colab_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
print(json.dumps(summary, indent=2))
"""

CELL_7_GRPO_SOURCE = """
# === Optional: Generate & Score Stage 2 Rollouts for Halo VLM SMPO/DPO & Offline GRPO ===
if GENERATE_OFFLINE_GRPO_ROLLOUTS:
    cand_key = CANDIDATE_KEYS[0]
    spec = COMPETITION_CANDIDATES[cand_key]
    adapter_dir = OUTPUT_ROOT / f'{cand_key}-real' / 'adapter_best'
    assert adapter_dir.is_dir(), f'Train real adapter first: {adapter_dir}'

    processor = load_processor_fullres({'hf_repo': spec.hf_repo, 'max_pixels': spec.max_pixels, 'trust_remote_code': spec.trust_remote_code})
    rollout_4bit = should_use_4bit(spec, VRAM_GB, 'baseline_zeroshot')
    if rollout_4bit and spec.quant_method == 'modelopt':
        raise RuntimeError(f'{spec.hf_repo} uses {spec.quant_method} and cannot use 4-bit bitsandbytes rollouts.')
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
    base_model = load_vlm_model(spec.hf_repo, rollout_load_kwargs)
    model = PeftModel.from_pretrained(base_model, str(adapter_dir))
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
            completions, finish_reasons = [], []
            for gen_idx in range(GRPO_NUM_GENERATIONS):
                temp = ROLLOUT_TEMPS[gen_idx % len(ROLLOUT_TEMPS)]
                with torch.no_grad():
                    out_ids = model.generate(**inputs, max_new_tokens=MAX_EVAL_NEW_TOKENS, do_sample=True, temperature=temp, top_p=0.9)
                gen_ids = out_ids[:, inputs['input_ids'].shape[1]:]
                fr = 'length' if int(gen_ids.shape[1]) >= int(MAX_EVAL_NEW_TOKENS) else 'stop'
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
                'image_path': row['image_path'],
                'prompt': DIPLOMATIC_PROMPT,
                'text': row['text'],
                'completions': completions,
                'finish_reasons': finish_reasons,
            })

    pref_pairs = build_preference_pairs(pref_records, min_reward_margin=0.05)
    pref_jsonl.write_text(
        '\\n'.join(json.dumps(pair, ensure_ascii=False) for pair in pref_pairs) + ('\\n' if pref_pairs else ''),
        encoding='utf-8',
    )
    print(f'Wrote {len(pref_pairs)} Halo VLM preference pairs -> {pref_jsonl} (leaks={rollout_thinking_leaks})')
    print('Wrote grouped Halo Offline GRPO rollouts ->', out_jsonl)
"""

CELL_8_MLX_SOURCE = """
# === Apple Silicon (M3 Max <= 24 GB) MLX Conversion Commands ===
for cand_key in CANDIDATE_KEYS:
    spec = COMPETITION_CANDIDATES[cand_key]
    merged = ALL_RESULTS.get(cand_key, {}).get('real', {}).get('merged_hf_dir') or f'<merged_{cand_key}_hf_dir>'
    awq_note = ' --quant-method awq --calibration multimodal --calibration-data <calib_dir>' if spec.mlx_quant_bits == 4 else ''
    print(f'# {cand_key} ({spec.hf_repo}, est MLX peak ~{spec.est_mlx_peak_gb} GB):')
    print(f'python -m mlx_vlm convert --hf-path {merged} --mlx-path ./mlx/{cand_key}-q{spec.mlx_quant_bits} -q --q-bits {spec.mlx_quant_bits}{awq_note}')
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
    """Return Cell 8 Apple Silicon MLX conversion command generator."""
    return CELL_8_MLX_SOURCE
