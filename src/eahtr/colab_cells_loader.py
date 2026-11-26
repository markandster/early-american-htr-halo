"""Cell 5 model/processor loader and multimodal training-batch helper source."""

from __future__ import annotations

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
