"""Cell 5 training-helper assembly (processor loader, batch builder, arm runner)."""

from __future__ import annotations

from .colab_cells_eval import build_cell_5_eval_section

CELL_5_LOADER_SOURCE = """
# === Training, Vision-Grad Verification & Verifiable Scoring Loop ===
def load_processor_fullres(spec_dict: dict):
    hf_repo = spec_dict['hf_repo']
    max_pixels = spec_dict['max_pixels']
    trust = bool(spec_dict.get('trust_remote_code', False))
    rev = spec_dict.get('hf_revision')
    proc_kw = {'trust_remote_code': trust}
    if rev:
        proc_kw['revision'] = rev
    try:
        proc = AutoProcessor.from_pretrained(hf_repo, max_pixels=max_pixels, **proc_kw)
    except TypeError:
        proc = AutoProcessor.from_pretrained(hf_repo, **proc_kw)
    ip = getattr(proc, 'image_processor', None)
    if ip is not None:
        if hasattr(ip, 'max_soft_tokens'):
            ip.max_soft_tokens = 1120
        if hasattr(ip, 'image_seq_length'):
            ip.image_seq_length = 1120
        if hasattr(ip, 'max_pixels'):
            ip.max_pixels = max_pixels
    if hasattr(proc, 'image_seq_length'):
        proc.image_seq_length = 1120
    if 'gemma-4' in hf_repo.lower():
        tpl_path = REPO_DIR / 'configs' / 'halo' / 'templates' / 'gemma4_htr_nothink.jinja'
        if tpl_path.is_file():
            htr_tpl = tpl_path.read_text(encoding='utf-8')
            proc.chat_template = htr_tpl
            if hasattr(proc, 'tokenizer') and proc.tokenizer is not None:
                proc.tokenizer.chat_template = htr_tpl
    return proc


def load_vlm_model(hf_repo: str, load_kwargs: dict):
    # Load VLM cleanly; Gemma 4 global_head_dim=512 requires sdpa (FlashAttention rejects it).
    resolved_kwargs = dict(load_kwargs)
    if 'gemma-4' in hf_repo.lower() and 'attn_implementation' not in resolved_kwargs:
        resolved_kwargs['attn_implementation'] = 'sdpa'
    prefer_multimodal = 'inkling' in hf_repo.lower() and AutoModelForMultimodalLM is not None
    loaders = (
        [AutoModelForMultimodalLM, AutoModelForImageTextToText]
        if prefer_multimodal
        else [cls for cls in (AutoModelForImageTextToText, AutoModelForMultimodalLM) if cls is not None]
    )
    last_err = None
    for cls in loaders:
        try:
            return cls.from_pretrained(hf_repo, **resolved_kwargs)
        except (ValueError, KeyError, TypeError) as exc:
            last_err = exc
    raise last_err


def _apply_chat_nothink(processor, messages: list[dict], *, add_generation_prompt: bool) -> str:
    try:
        return processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=False,
        )
    except TypeError:
        return processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )


def _resolve_turn_terminator(tokenizer) -> str:
    eot = getattr(tokenizer, 'eot_token', None)
    if isinstance(eot, str) and eot:
        return eot
    vocab = tokenizer.get_vocab() if hasattr(tokenizer, 'get_vocab') else {}
    if '<turn|>' in vocab:
        return '<turn|>'
    return getattr(tokenizer, 'eos_token', None) or ''


def build_training_batch(processor, image: Image.Image, target_text: str, device):
    tok = getattr(processor, 'tokenizer', processor)
    user_msgs = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': DIPLOMATIC_PROMPT}]}]
    prompt_str = _apply_chat_nothink(processor, user_msgs, add_generation_prompt=True)
    eot_token = _resolve_turn_terminator(tok)
    full_str = prompt_str + target_text + eot_token

    prompt_inputs = processor(text=[prompt_str], images=[image], return_tensors='pt')
    full_inputs = processor(text=[full_str], images=[image], return_tensors='pt')
    prompt_len = prompt_inputs['input_ids'].shape[1]

    # Never truncate expanded VLM tensors mid-sample (desyncs mm_token_type_ids / token_type_ids).
    if full_inputs['input_ids'].shape[1] > MAX_SEQ_LEN:
        return None

    assert full_inputs['input_ids'].shape[1] >= prompt_len and torch.equal(
        full_inputs['input_ids'][:, :prompt_len], prompt_inputs['input_ids']
    ), 'Tokenized prompt/full prefix mismatch detected; refusing misaligned training batch.'

    labels = full_inputs['input_ids'].clone()
    labels[:, :prompt_len] = -100
    pad_id = getattr(tok, 'pad_token_id', None)
    if pad_id is not None:
        labels[labels == pad_id] = -100
    img_tid = getattr(tok, 'image_token_id', None)
    if img_tid is not None:
        labels[labels == img_tid] = -100
    assert int((labels[:, prompt_len:] != -100).sum().item()) > 0, (
        'No unmasked completion tokens after prompt boundary.'
    )
    full_inputs['labels'] = labels
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in full_inputs.items()}

"""

CELL_5_ARM_SOURCE = """
def _build_arm_cache_key(cand_key: str, arm: str, spec, train_rows: list[dict], eval_rows: list[dict], use_4bit: bool) -> str:
    tpl_file = REPO_DIR / 'configs' / 'halo' / 'templates' / 'gemma4_htr_nothink.jinja'
    tpl_sha = hashlib.sha256(tpl_file.read_bytes()).hexdigest()[:12] if tpl_file.is_file() else 'hub'
    prompt_sha = hashlib.sha256(DIPLOMATIC_PROMPT.encode('utf-8')).hexdigest()[:12]
    payload = {
        'cand': cand_key, 'hf_repo': spec.hf_repo, 'hf_rev': spec.hf_revision, 'arm': arm, 'smoke': SMOKE_TEST_MODE,
        'ep': EPOCHS, 'r': LORA_RANK, 'alpha': LORA_ALPHA, 'drop': LORA_DROPOUT, 'wd': WEIGHT_DECAY,
        'warmup': WARMUP_RATIO, 'lr': LEARNING_RATE, 'vlr': VISION_LR, 'ga': GRAD_ACCUM_STEPS, 'seq': MAX_SEQ_LEN,
        'max_new': MAX_EVAL_NEW_TOKENS, 'seed': SEED,
        'oof': [globals().get('OOF_NUM_FOLDS', 1), globals().get('OOF_FOLD_IDX', 0)],
        'use_4bit': use_4bit, 'git_sha': globals().get('REPO_GIT_SHA', 'unknown'), 'tpl': tpl_sha, 'prompt': prompt_sha,
        'max_px': spec.max_pixels, 'lora_targets': list(spec.lora_target_modules), 'vis_pats': list(spec.vision_trainable_patterns),
        'train_sig': [[r.get('record_id', ''), r.get('image_sha256', ''), hashlib.sha256(r.get('text', '').encode()).hexdigest()[:12], r.get('sham_donor_record_id', '')] for r in train_rows],
        'eval_sig': [[r.get('record_id', ''), r.get('image_sha256', ''), hashlib.sha256(r.get('text', '').encode()).hexdigest()[:12]] for r in eval_rows],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode('utf-8')).hexdigest()[:16]


def _merge_saved_adapter(spec, processor, adapter_dir: Path, merged_dir: Path, use_4bit: bool, live_model=None) -> str | None:
    merged_dir.parent.mkdir(parents=True, exist_ok=True)
    if not use_4bit and live_model is not None:
        merged_model = live_model.merge_and_unload()
        merged_model.save_pretrained(merged_dir, safe_serialization=True)
        processor.save_pretrained(merged_dir)
        del merged_model
        _assert_highres_processor_config(merged_dir, spec.hf_repo, reference_dir=adapter_dir)
        return str(merged_dir)
    try:
        host_ram_gb = (os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')) / (1024 ** 3)
    except Exception:
        host_ram_gb = None
    if host_ram_gb is not None and spec.params_b * 3.0 > host_ram_gb:
        print(f'[SKIP MERGE] Host RAM {host_ram_gb:.1f} GB < ~{spec.params_b * 3.0:.0f} GB needed; adapter saved at {adapter_dir}.')
        return None
    cpu_kw = {'device_map': 'cpu', 'torch_dtype': torch.bfloat16, 'trust_remote_code': spec.trust_remote_code}
    if spec.hf_revision:
        cpu_kw['revision'] = spec.hf_revision
    cpu_base = load_vlm_model(spec.hf_repo, cpu_kw)
    merged_model = PeftModel.from_pretrained(cpu_base, str(adapter_dir)).merge_and_unload()
    merged_model.save_pretrained(merged_dir, safe_serialization=True)
    processor.save_pretrained(merged_dir)
    del merged_model, cpu_base
    _assert_highres_processor_config(merged_dir, spec.hf_repo, reference_dir=adapter_dir)
    return str(merged_dir)


def run_candidate_arm(cand_key: str, arm: str, arm_train_rows: list[dict], val_slice: list[dict], test_slice: list[dict]) -> dict:
    spec = COMPETITION_CANDIDATES[cand_key]
    use_4bit = should_use_4bit(spec, VRAM_GB, arm)
    eval_target = test_slice if (FINAL_FREEZE_EVAL_TEST_PUBLIC and test_slice) else val_slice
    eval_split_name = 'public_test' if (FINAL_FREEZE_EVAL_TEST_PUBLIC and test_slice) else 'val'
    out_dir = OUTPUT_ROOT / f'{cand_key}-{arm}'
    cache_key = _build_arm_cache_key(cand_key, arm, spec, arm_train_rows, eval_target, use_4bit)
    run_dir = out_dir / cache_key
    merged_dir = WORK_ROOT / 'merged' / f'{cand_key}-{arm}-{cache_key}'
    out_dir.mkdir(parents=True, exist_ok=True); run_dir.mkdir(parents=True, exist_ok=True)
    score_file = out_dir / f'{eval_split_name}_eval_scores_{cache_key}.json'
    if score_file.is_file():
        cached = json.loads(score_file.read_text(encoding='utf-8'))
        if arm == 'baseline_zeroshot':
            print(f'[SKIP] Cache hit {cand_key}-{arm} ({cache_key}): {score_file}')
            return cached
        a_dir = cached.get('adapter_dir')
        if not a_dir or not Path(a_dir).is_dir():
            print(f'[WARN] Cached adapter_dir missing on disk; invalidating stale cache and re-running.')
        else:
            m_dir = cached.get('merged_hf_dir')
            if not m_dir or not Path(m_dir).is_dir():
                print(f'[RECOVER] Rebuilding missing merged_hf_dir from {a_dir} -> {merged_dir}')
                proc = load_processor_fullres({'hf_repo': spec.hf_repo, 'hf_revision': spec.hf_revision, 'max_pixels': spec.max_pixels, 'trust_remote_code': spec.trust_remote_code})
                try:
                    cached['merged_hf_dir'] = _merge_saved_adapter(spec, proc, Path(a_dir), merged_dir, use_4bit)
                    score_file.write_text(json.dumps(cached, indent=2), encoding='utf-8')
                except Exception as rec_exc:
                    cached['merged_hf_dir'] = None
                    print(f'[WARN] Cached merge recovery failed: {rec_exc}')
            print(f'[SKIP] Cache hit {cand_key}-{arm} ({cache_key}): {score_file}')
            return cached

    print(f'\\n================ {cand_key} ({spec.hf_repo}@{spec.hf_revision}) | ARM={arm} | EVAL={eval_split_name} ================')
    processor = load_processor_fullres({'hf_repo': spec.hf_repo, 'hf_revision': spec.hf_revision, 'max_pixels': spec.max_pixels, 'trust_remote_code': spec.trust_remote_code})

    if use_4bit and spec.quant_method == 'modelopt':
        raise RuntimeError(f'Candidate {spec.hf_repo} uses {spec.quant_method} quantization; cannot use 4-bit NF4 QLoRA.')

    load_kwargs = {'device_map': {'': 0} if use_4bit else 'auto', 'torch_dtype': torch.bfloat16, 'trust_remote_code': spec.trust_remote_code}
    if spec.hf_revision:
        load_kwargs['revision'] = spec.hf_revision
    if 'gemma-4' in spec.hf_repo.lower():
        load_kwargs['attn_implementation'] = 'sdpa'
    if use_4bit:
        print(f'VRAM {VRAM_GB} GB: loading {cand_key} ({arm}) in 4-bit NF4 (BF16 compute/storage)')
        bnb_kw = dict(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_storage=torch.bfloat16)
        if 'gemma-4' not in spec.hf_repo.lower():
            bnb_kw['llm_int8_skip_modules'] = list(spec.vision_trainable_patterns)
        load_kwargs['quantization_config'] = BitsAndBytesConfig(**bnb_kw)

    model = load_vlm_model(spec.hf_repo, load_kwargs)

    if arm == 'baseline_zeroshot':
        eval_res = run_eval_panel(model, processor, eval_target, panel_name=f'{cand_key}-zeroshot-{eval_split_name}')
        payload = {
            'candidate': cand_key, 'hf_repo': spec.hf_repo, 'arm': arm, 'eval_split': eval_split_name,
            'cache_key': cache_key, 'used_4bit': use_4bit, 'per_side_cer': [p['strict_cer'] for p in eval_res['predictions']], 'scores': eval_res['scores'],
        }
        (out_dir / f'predictions_{eval_split_name}_{cache_key}.json').write_text(json.dumps(eval_res['predictions'], ensure_ascii=False, indent=2), encoding='utf-8')
        score_file.write_text(json.dumps(payload, indent=2), encoding='utf-8')
        print(f'[ZERO-SHOT {cand_key} | {eval_split_name}]', payload['scores'])
        del model; gc.collect(); torch.cuda.empty_cache()
        return payload

    gc_kwargs = {'use_reentrant': False}
    if use_4bit:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True, gradient_checkpointing_kwargs=gc_kwargs
        )
    else:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gc_kwargs)
        model.enable_input_require_grads()

    try:
        vision_modules = resolve_vision_modules_to_save(
            (n for n, _ in model.named_modules()),
            spec.vision_trainable_patterns,
        )
    except RuntimeError as exc:
        top_names = sorted({n.split('.')[0] for n, _ in model.named_modules() if n})[:20]
        print('Module inspection failed. Top-level modules:', top_names)
        raise exc
    print('Resolved trainable vision modules:', vision_modules)
    lora_targets = 'all-linear' if spec.lora_target_modules == ('all-linear',) else list(spec.lora_target_modules)
    lora_kw = {
        'r': LORA_RANK, 'lora_alpha': LORA_ALPHA, 'lora_dropout': LORA_DROPOUT,
        'target_modules': lora_targets, 'bias': 'none', 'task_type': 'CAUSAL_LM',
    }
    if not use_4bit:
        lora_kw.update(exclude_modules=vision_modules, modules_to_save=vision_modules)
    model = get_peft_model(model, LoraConfig(**lora_kw))

    vision_params, lora_params = [], []
    for pname, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(vp in pname for vp in spec.vision_trainable_patterns) or 'modules_to_save' in pname:
            vision_params.append(param)
        else:
            lora_params.append(param)
    n_vision_trainable = sum(p.numel() for p in vision_params)
    n_lora_trainable = sum(p.numel() for p in lora_params)
    print(f'Verified trainable params -> vision: {n_vision_trainable:,} | LoRA: {n_lora_trainable:,}')
    assert n_vision_trainable > 0, 'FATAL: 0 vision parameters are trainable!'

    optimizer = torch.optim.AdamW([
        {'params': lora_params, 'lr': LEARNING_RATE, 'weight_decay': WEIGHT_DECAY},
        {'params': vision_params, 'lr': VISION_LR, 'weight_decay': WEIGHT_DECAY},
    ])
    total_opt_steps = max(1, math.ceil(len(arm_train_rows) * EPOCHS / GRAD_ACCUM_STEPS))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(total_opt_steps * WARMUP_RATIO)),
        num_training_steps=total_opt_steps,
    )

    rng = random.Random(SEED)
    best_val_cer = float('inf')
    best_epoch = 0
    best_trainable_state = None
    train_log = []
    skipped_oom = 0
    discarded_oom_microbatches = 0
    skipped_overlength = 0
    verified_step1_grads = False
    realized_image_tokens = None
    opt_step = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        order = list(arm_train_rows)
        rng.shuffle(order)
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        n_batches = 0
        accum_count = 0

        for row in tqdm(order, desc=f'{cand_key}-{arm} ep{epoch}/{EPOCHS}'):
            try:
                image = Image.open(row['image_path']).convert('RGB')
                batch = build_training_batch(processor, image, row['text'], model.device)
                if batch is None:
                    skipped_overlength += 1
                    continue
                if realized_image_tokens is None:
                    tok = getattr(processor, 'tokenizer', processor)
                    img_tid = getattr(tok, 'image_token_id', None)
                    if img_tid is not None:
                        realized_image_tokens = int((batch['input_ids'] == img_tid).sum().item())
                        print(f'Batch 0 realized image_tokens_per_page={realized_image_tokens}')
                        if 'gemma-4' in spec.hf_repo.lower() and realized_image_tokens <= 280:
                            raise RuntimeError(f'Gemma 4 vision tokens collapsed to {realized_image_tokens} (<= 280).')
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    out = model(**batch)
                    loss = out.loss / GRAD_ACCUM_STEPS
                loss.backward()
                if not verified_step1_grads:
                    has_live_vision_grad = any(
                        p.grad is not None and p.grad.abs().sum().item() > 0 for p in vision_params
                    )
                    if not has_live_vision_grad:
                        raise RuntimeError('Live step-1 vision gradient check failed: vision grad norm is zero.')
                    verified_step1_grads = True
                running_loss += float(out.loss.detach().cpu())
                n_batches += 1
                accum_count += 1
            except torch.cuda.OutOfMemoryError:
                skipped_oom += 1
                discarded_oom_microbatches += accum_count
                optimizer.zero_grad(set_to_none=True)
                accum_count = 0
                torch.cuda.empty_cache()
                continue

            if accum_count >= GRAD_ACCUM_STEPS:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                accum_count = 0
                opt_step += 1
                if globals().get('CHECKPOINT_EVERY_OPT_STEPS', 0) > 0 and opt_step % int(CHECKPOINT_EVERY_OPT_STEPS) == 0:
                    # Manual salvage snapshot only (weights + processor; no optimizer/scheduler state or auto-resume)
                    model.save_pretrained(run_dir / 'adapter_last')
                    processor.save_pretrained(run_dir / 'adapter_last')

        if accum_count > 0:
            scale = GRAD_ACCUM_STEPS / accum_count
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.mul_(scale)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        if n_batches == 0:
            raise RuntimeError(f'{cand_key}-{arm} epoch {epoch} completed 0 valid batches (oom={skipped_oom}, overlength={skipped_overlength}).')
        if skipped_oom > max(3, int(0.1 * len(order))):
            raise RuntimeError(f'{cand_key}-{arm} exceeded OOM threshold: skipped_oom={skipped_oom}/{len(order)}.')

        avg_loss = round(running_loss / max(1, n_batches), 4)
        val_eval = run_eval_panel(model, processor, val_slice, panel_name=f'{cand_key}-{arm}-val-ep{epoch}')
        val_cer = val_eval['scores']['strict_cer_pct']
        train_log.append({
            'epoch': epoch,
            'train_loss': avg_loss,
            'skipped_oom': skipped_oom,
            'skipped_overlength': skipped_overlength,
            'val_scores': val_eval['scores'],
        })
        print(f'Epoch {epoch} | loss={avg_loss} | val={val_eval["scores"]}')

        if val_cer <= best_val_cer:
            best_val_cer = val_cer
            best_epoch = epoch
            best_trainable_state = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}
            adapter_dir = run_dir / 'adapter_best'
            model.save_pretrained(adapter_dir)
            processor.save_pretrained(adapter_dir)

    if best_trainable_state is not None:
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in best_trainable_state:
                    p.copy_(best_trainable_state[n].to(device=p.device, dtype=p.dtype))
        print(f'Restored best-val checkpoint from epoch {best_epoch} (val strict CER={best_val_cer}%)')

    final_eval = run_eval_panel(model, processor, eval_target, panel_name=f'{cand_key}-{arm}-{eval_split_name}')
    pred_bytes = json.dumps(final_eval['predictions'], ensure_ascii=False, indent=2).encode('utf-8')
    (out_dir / f'predictions_{eval_split_name}_{cache_key}.json').write_bytes(pred_bytes)
    swap_delta = None
    if arm == 'real' and len(eval_target) >= 4:
        swap_subset = eval_target[:min(16, len(eval_target))]
        swap_eval = run_eval_panel(model, processor, swap_subset, panel_name=f'{cand_key}-real-swap-check', swap_images=True)
        matched_preds = final_eval['predictions'][:len(swap_subset)]
        matched_sub = score_predictions_verifiable(
            [p['reference'] for p in matched_preds],
            [p['prediction'] for p in matched_preds],
            [p['finish_reason'] for p in matched_preds],
        )['strict_cer_pct']
        swap_delta = round(swap_eval['scores']['strict_cer_pct'] - matched_sub, 2)
        print(f'Image-swap check ({len(swap_subset)} pages): swap CER={swap_eval["scores"]["strict_cer_pct"]}% (delta={swap_delta:+.2f} pp)')

    payload = {
        'candidate': cand_key, 'hf_repo': spec.hf_repo, 'arm': arm, 'eval_split': eval_split_name, 'cache_key': cache_key,
        'train_sides': len(arm_train_rows), 'vision_trainable_params': n_vision_trainable, 'lora_trainable_params': n_lora_trainable,
        'image_tokens_per_page': realized_image_tokens, 'used_4bit_qlora': use_4bit, 'skipped_overlength': skipped_overlength,
        'discarded_oom_microbatches': discarded_oom_microbatches, 'best_epoch': best_epoch, 'train_log': train_log,
        'predictions_sha256': hashlib.sha256(pred_bytes).hexdigest(), 'per_side_cer': [p['strict_cer'] for p in final_eval['predictions']],
        'swap_cer_degradation': swap_delta, 'scores': final_eval['scores'], 'adapter_dir': str(run_dir / 'adapter_best'), 'merged_hf_dir': None,
    }
    score_file.write_text(json.dumps(payload, indent=2), encoding='utf-8')

    try:
        if use_4bit:
            del model, optimizer, scheduler; gc.collect(); torch.cuda.empty_cache()
            model = optimizer = scheduler = None
        payload['merged_hf_dir'] = _merge_saved_adapter(
            spec, processor, run_dir / 'adapter_best', merged_dir, use_4bit, live_model=(None if use_4bit else model)
        )
        score_file.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    except Exception as merge_exc:
        print(f'[WARN] Post-eval merge failed (scores & adapter already saved): {merge_exc}')
    print(f'[DONE {cand_key}-{arm} | {eval_split_name}]', payload['scores'])

    del model, optimizer, scheduler
    gc.collect()
    torch.cuda.empty_cache()
    return payload
"""


def build_cell_5_train_helpers() -> str:
    """Assemble the complete Cell 5 training, evaluation, and merge source."""
    return CELL_5_LOADER_SOURCE + build_cell_5_eval_section() + CELL_5_ARM_SOURCE
