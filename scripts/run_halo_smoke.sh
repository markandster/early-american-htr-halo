#!/usr/bin/env bash
set -euo pipefail

# Clean-room 1x GPU 5-step Halo VLM SFT + merge + continuity + SMPO smoke harness.
# Bootstraps Python 3.12 via uv and installs PyTorch 2.11.0 matching the active NVIDIA driver.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HALO_DIR="${HALO_DIR:-/tmp/halo}"
HALO_COMMIT="${HALO_COMMIT:-53fe34a}"
GEMMA4_12B_REV="${GEMMA4_12B_REV:-43c9d014b231}"
SMOKE_VENV="${SMOKE_VENV:-/tmp/halo-gpu-smoke-venv}"
WORK_DIR="${WORK_DIR:-/tmp/eahtr_halo_smoke}"
mkdir -p "${WORK_DIR}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi not found; 1x CUDA GPU required for live Halo smoke run." >&2
  exit 1
fi

DRIVER_VER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1 | xargs)"
VRAM_MIB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1 | xargs)"
VRAM_GB="$(python3 -c "print(round(${VRAM_MIB} / 1024.0, 2))")"
echo "Detected NVIDIA driver=${DRIVER_VER}, VRAM=${VRAM_GB} GiB"

if [[ ! -d "${HALO_DIR}/halo" ]]; then
  git clone --depth 50 https://github.com/whitecircle/halo.git "${HALO_DIR}"
fi
git -C "${HALO_DIR}" fetch --tags --quiet || true
git -C "${HALO_DIR}" checkout "${HALO_COMMIT}"

command -v uv >/dev/null 2>&1 || pip install -q uv
uv venv --clear --python 3.12 "${SMOKE_VENV}"
uv pip install --python "${SMOKE_VENV}/bin/python" -e "${REPO_ROOT}"
TORCH_INDEX="$("${SMOKE_VENV}/bin/python" -c "from eahtr.halo_smoke import select_pytorch_cuda_index; print(select_pytorch_cuda_index('${DRIVER_VER}'))")" || exit 1
echo "Using PyTorch wheel index: ${TORCH_INDEX}"

uv pip install --python "${SMOKE_VENV}/bin/python" --index-url "${TORCH_INDEX}" "torch==2.11.0" "torchvision==0.26.0"
uv pip install --python "${SMOKE_VENV}/bin/python" \
  --extra-index-url "${TORCH_INDEX}" --index-strategy unsafe-best-match \
  "transformers==5.23.0" "trl==1.14.2" "peft==0.18.1" "accelerate>=1.11.0,<1.12.0" \
  "datasets>=4.8.5,<5.0.0" "ruamel.yaml>=0.19.1,<0.20.0" "pydantic>=2.6.4,<3.0.0" \
  "huggingface-hub>=1.0.0,<2.0.0" "pyarrow>=24.0.0,<25.0.0" "tokenizers>=0.23.1,<0.24.0" \
  "bitsandbytes>=0.49.2,<0.50.0" "liger-kernel>=0.8.4,<0.9.0" "flash-linear-attention>=0.6.0,<0.7.0" \
  "multiprocess>=0.70.19,<0.71.0" "pandas>=2.2.1,<3.0.0" pillow "wandb>=0.25.0,<0.26.0" \
  "clearml>=2.0.2,<3.0.0" "tabulate>=0.9.0,<0.10.0" "boto3>=1.35.0,<2.0.0" \
  "s3fs>=2024.6.0" "openai>=1.44.1,<2.0.0" "backoff>=2.2.1,<3.0.0" \
  "typer>=0.25.1,<0.26.0" "click>=8.1.0,<9.0.0" einops
uv pip install --python "${SMOKE_VENV}/bin/python" --no-deps \
  "gram-newton-schulz @ git+https://github.com/Dao-AILab/gram-newton-schulz.git@244a451d684a39d68f593aef03b6fe7cfa6a31d1"
uv pip install --python "${SMOKE_VENV}/bin/python" --no-deps -e "${HALO_DIR}"
"${SMOKE_VENV}/bin/python" -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable after PyTorch install'"
"${SMOKE_VENV}/bin/python" -c "from halo.trainers.sft import DistributedSFTTrainer; from halo.trainers.preference.smpo import SMPOTrainer; from halo.checkpoint.tools.adapter_merge import merge_peft_adapter; print('Halo trainer imports OK')"
"${SMOKE_VENV}/bin/python" "${REPO_ROOT}/scripts/validate_halo_configs.py" --halo-repo "${HALO_DIR}"

FLEX_ENV="1"
if python3 -c "import sys; sys.exit(0 if float('${VRAM_GB}') < 32.0 else 1)"; then
  FLEX_ENV="0"
fi

echo "Staging pinned Gemma 4 12B (${GEMMA4_12B_REV}), fetching RevCity sample via HTTP Range, and checking Halo VLMDataCollator..."
"${SMOKE_VENV}/bin/python" - <<PY
from pathlib import Path
from datasets import load_from_disk
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoProcessor, AutoTokenizer, GenerationConfig
from halo.data.collators.vlm import VLMDataCollator
from halo.data.pipeline.vlm_rows import build_vlm_row_processor
from eahtr.candidates import REVCITY_ZENODO_URL
from eahtr.halo_smoke import (
    build_smoke_yaml_configs,
    export_halo_vlm_smoke_datasets,
    fetch_revcity_smoke_sample,
    prepare_gemma4_highres_dir,
    verify_processor_image_tokens,
)

work = Path("${WORK_DIR}")
repo = Path("${REPO_ROOT}")
vram_gb = float("${VRAM_GB}")
records = fetch_revcity_smoke_sample(REVCITY_ZENODO_URL, work / "images", max_pages=32)
assert len(records) >= 16, f"Expected >=16 smoke records via Range read, got {len(records)}"

cache_dir = Path(snapshot_download(
    "google/gemma-4-12B-it",
    revision="${GEMMA4_12B_REV}",
    allow_patterns=["*.json", "*.jinja", "*.model", "*.safetensors*"],
))
staged_dir = prepare_gemma4_highres_dir(
    cache_dir,
    work / "gemma4_12b_highres",
    chat_template_path=repo / "configs" / "halo" / "templates" / "gemma4_htr_nothink.jinja",
)

proc = AutoProcessor.from_pretrained(str(staged_dir))
realized_toks = verify_processor_image_tokens(proc, Path(records[0]["image_path"]))
tok = getattr(proc, "tokenizer", None) or AutoTokenizer.from_pretrained(str(staged_dir))
htr_tpl = (repo / "configs" / "halo" / "templates" / "gemma4_htr_nothink.jinja").read_text(encoding="utf-8")
proc.chat_template = htr_tpl
tok.chat_template = htr_tpl
max_len = 2048 if vram_gb < 32.0 else 4096
max_prompt = 1350 if vram_gb < 32.0 else 1600
sft_ds, smpo_ds, dropped = export_halo_vlm_smoke_datasets(
    records, work / "datasets", lambda s: len(tok(s, add_special_tokens=False)["input_ids"]),
    max_length=max_len, max_prompt_length=max_prompt,
)

ds = load_from_disk(str(sft_ds))
collator = VLMDataCollator(
    proc, tok, max_length=max_len, train_on_completions_only=True,
    response_prompt_template="<|turn>model\\n<|channel>thought\\n<channel|>",
    model_config=AutoConfig.from_pretrained(str(staged_dir)),
    generation_config=GenerationConfig.from_pretrained(str(staged_dir)),
)
row0_norm = build_vlm_row_processor("messages", images_field="images")(ds["train"][0])
batch0 = collator([row0_norm])
img_tid = getattr(tok, "image_token_id", None) or tok.get_vocab()["<|image|>"]
b0_img = int((batch0["input_ids"][0] == img_tid).sum().item())
assert 281 <= b0_img <= 1120, f"Collated batch-0 image token collapse: {b0_img}"
assert int((batch0["labels"][0] != -100).sum().item()) > 0, "Collated batch-0 has no unmasked completion tokens"
eot_id = tok.convert_tokens_to_ids("<turn|>")
assert eot_id in batch0["labels"][0].tolist(), "Collated batch-0 labels missing <turn|> terminator"
print(f"Verified Halo VLMDataCollator batch-0 (proc={realized_toks}, batch={b0_img}, eot_id={eot_id})")

sft_yaml, smpo_yaml = build_smoke_yaml_configs(
    repo / "configs" / "halo" / "gemma4_12b_sft.yaml",
    repo / "configs" / "halo" / "gemma4_12b_vlm_smpo.yaml",
    work / "configs", model_dir=staged_dir, smpo_model_dir=work / "configs" / "sft_merged",
    sft_dataset_dir=sft_ds, smpo_dataset_dir=smpo_ds, vram_gb=vram_gb, repo_root=repo,
)
print(f"Prepared smoke YAMLs (dropped={dropped}): {sft_yaml} | {smpo_yaml}")
PY

cd "${REPO_ROOT}"
"${SMOKE_VENV}/bin/python" "${REPO_ROOT}/scripts/validate_halo_configs.py" --config-dir "${WORK_DIR}/configs" --halo-repo "${HALO_DIR}"

HALO_FLEX_SLIDING="${FLEX_ENV}" "${SMOKE_VENV}/bin/python" "${HALO_DIR}/scripts/training/sft.py" "${WORK_DIR}/configs/smoke_sft.yaml" 2>&1 | tee "${WORK_DIR}/configs/sft_run.log"
"${SMOKE_VENV}/bin/python" -c "from pathlib import Path; from eahtr.halo_smoke import verify_halo_run_log_finite_loss; print('SFT losses:', verify_halo_run_log_finite_loss(Path('${WORK_DIR}/configs/sft_run.log'), min_steps=5))"

RESOLVED_SFT_ADAPTER="$("${SMOKE_VENV}/bin/python" -c "from pathlib import Path; d=Path('${WORK_DIR}/configs/sft_out'); cands=[d]+[p for p in sorted(d.glob('checkpoint-*'), key=lambda p: int(p.name.split('-')[-1]) if p.name.split('-')[-1].isdigit() else -1, reverse=True) if p.is_dir()]; print(next(str(p) for p in cands if (p/'adapter_model.safetensors').is_file()))")"
echo "Merging Stage 1 SFT adapter (${RESOLVED_SFT_ADAPTER}) via Halo merge_peft_adapters.py..."
"${SMOKE_VENV}/bin/python" "${HALO_DIR}/scripts/after_training/merge_peft_adapters.py" \
  --adapter_dir "${RESOLVED_SFT_ADAPTER}" --output_dir "${WORK_DIR}/configs/sft_merged"
"${SMOKE_VENV}/bin/python" "${REPO_ROOT}/scripts/verify_mlx_export.py" "${WORK_DIR}/configs/sft_merged" --reference-dir "${RESOLVED_SFT_ADAPTER}"

echo "Checking Stage 1 -> Stage 2 fresh-process log-prob continuity and stop termination..."
HALO_FLEX_SLIDING="${FLEX_ENV}" "${SMOKE_VENV}/bin/python" - <<PY
import json, subprocess, sys
from pathlib import Path
import torch
from datasets import load_from_disk
from peft import PeftModel
from safetensors.torch import load_file
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from eahtr.candidates import DIPLOMATIC_PROMPT
from eahtr.halo_smoke import verify_stage_continuity_logprobs

work = Path("${WORK_DIR}")
sft_ds_path = work / "datasets" / "halo_vlm_sft_smoke"
ds = load_from_disk(str(sft_ds_path))["train"]
sample_rows = [ds[0], ds[min(1, len(ds) - 1)]]
shortest_row = min(list(ds), key=lambda r: len(r["messages"][1]["content"][0]["text"]))

adapter_st = load_file(str(Path("${RESOLVED_SFT_ADAPTER}") / "adapter_model.safetensors"))
vision_deltas = [
    v.float().abs().max().item()
    for k, v in adapter_st.items()
    if any(t in k for t in ("vision", "embed_vision", "projector")) and ("lora_B." in k or "modules_to_save" in k)
]
assert vision_deltas and max(vision_deltas) > 0.0, "Stage 1 Halo SFT recorded zero vision lora_B/modules_to_save updates"

vram_gb = float("${VRAM_GB}")
bnb_cfg = (
    BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    if vram_gb < 56.0
    else None
)

def _stop_ids(tok, gen_cfg=None) -> set[int]:
    ids: set[int] = set()
    for owner in (tok, gen_cfg):
        if owner is None:
            continue
        eos = getattr(owner, "eos_token_id", None)
        if isinstance(eos, int) and eos >= 0:
            ids.add(eos)
        elif isinstance(eos, (list, tuple)):
            ids.update(int(x) for x in eos if isinstance(x, int) and x >= 0)
    for token in ("<turn|>", "<eos>"):
        tid = tok.convert_tokens_to_ids(token)
        if isinstance(tid, int) and tid >= 0:
            ids.add(tid)
    return ids

def _score_examples(model_path: Path, *, adapter_dir: Path | None = None, check_generate: bool = False) -> list[float]:
    proc = AutoProcessor.from_pretrained(str(model_path))
    load_kw = {"torch_dtype": torch.bfloat16, "device_map": "auto", "attn_implementation": "sdpa"}
    if bnb_cfg is not None:
        load_kw["quantization_config"] = bnb_cfg
    mdl = AutoModelForImageTextToText.from_pretrained(str(model_path), **load_kw)
    if adapter_dir is not None:
        mdl = PeftModel.from_pretrained(mdl, str(adapter_dir))
    mdl.eval()
    collected: list[float] = []
    tok = getattr(proc, "tokenizer", proc)
    with torch.no_grad():
        for row in sample_rows:
            img = row["images"][0].convert("RGB")
            ref_text = row["messages"][1]["content"][0]["text"]
            prompt = proc.apply_chat_template(
                [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": DIPLOMATIC_PROMPT}]}],
                tokenize=False, add_generation_prompt=True,
            )
            p_in = proc(text=[prompt], images=[img], return_tensors="pt")
            f_in = proc(text=[prompt + ref_text + "<turn|>"], images=[img], return_tensors="pt")
            dev_in = {k: v.to(mdl.device) if torch.is_tensor(v) else v for k, v in f_in.items()}
            logits = mdl(**dev_in).logits[0]
            p_len = p_in["input_ids"].shape[1]
            tgt = dev_in["input_ids"][0, p_len:]
            lps = torch.log_softmax(logits[p_len - 1 : -1].float(), dim=-1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            collected.extend(float(x) for x in lps[:16].cpu().tolist())
        if check_generate:
            s_img = shortest_row["images"][0].convert("RGB")
            s_ref = shortest_row["messages"][1]["content"][0]["text"]
            ref_toks = len(tok(s_ref, add_special_tokens=False)["input_ids"])
            stop_cap = max(512, min(1024, ref_toks * 2 + 64))
            s_prompt = proc.apply_chat_template(
                [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": DIPLOMATIC_PROMPT}]}],
                tokenize=False, add_generation_prompt=True,
            )
            sp_in = proc(text=[s_prompt], images=[s_img], return_tensors="pt")
            out = mdl.generate(**{k: v.to(mdl.device) if torch.is_tensor(v) else v for k, v in sp_in.items()}, max_new_tokens=stop_cap, do_sample=False)
            gen_ids = out[:, sp_in["input_ids"].shape[1]:]
            decoded = proc.batch_decode(gen_ids, skip_special_tokens=False)[0]
            assert "<|channel>thought" not in decoded, f"Thinking leak in Stage 1 smoke generate: {decoded[:120]}"
            sids = _stop_ids(tok, getattr(mdl, "generation_config", None))
            last_id = int(gen_ids[0, -1].item())
            assert last_id in sids, f"Stage 1 smoke failed natural stop (last_id={last_id}, len={int(gen_ids.shape[1])}, cap={stop_cap})"
    del mdl
    torch.cuda.empty_cache()
    return collected

base_lps = _score_examples(work / "gemma4_12b_highres")
live_sft_lps = _score_examples(work / "gemma4_12b_highres", adapter_dir=Path("${RESOLVED_SFT_ADAPTER}"))
sft_lps = _score_examples(work / "configs" / "sft_merged", check_generate=True)
sft_shift = max(abs(a - b) for a, b in zip(base_lps, sft_lps))
merge_diff = max(abs(a - b) for a, b in zip(live_sft_lps, sft_lps))
if merge_diff > 0.05:
    print(f"[WARN] Live PEFT vs merged SFT max logit diff={merge_diff:.6f} > 0.05 (base->SFT shift={sft_shift:.6f})")
merge_abort_tol = max(0.15, 0.5 * sft_shift) if vram_gb >= 56.0 else max(0.35, 0.75 * sft_shift)
assert merge_diff <= merge_abort_tol, f"Merged SFT diverged excessively from live PEFT adapter (diff={merge_diff:.6f} > {merge_abort_tol:.6f})"

stage2_probe = """
import json, sys
from pathlib import Path
import torch
from datasets import load_from_disk
from halo.configs.parallelism_arguments import ParallelismArguments
from halo.configs.preference import SMPOConfig, SMPOScriptArguments
from halo.configs.script_arguments import ModelArguments
from halo.entrypoints.data_setup import apply_vision_options, load_script_datasets, prepare_script_preference_data, resolve_preference_vlm_run
from halo.entrypoints.model_setup import apply_max_length, apply_text_path_padding_side, default_padded_attn_implementation, install_resolved_tokenizer, load_policy_model
from halo.entrypoints.script import configure_balancing_and_build_callbacks, distributed_trainer_kwargs, init_training_script, parse_script_args
from halo.model_loading.parallel.peft_setup import configure_adapter_training
from halo.trainers.preference.smpo import SMPOTrainer
from eahtr.candidates import DIPLOMATIC_PROMPT

work = Path(sys.argv[1])
probe_out = Path(sys.argv[2])
sys.argv = ["smpo.py", str(work / "configs" / "smoke_vlm_smpo.yaml")]
s_args, s_cfg, m_cfg, d_args = parse_script_args(SMPOScriptArguments, SMPOConfig, ModelArguments, ParallelismArguments)
ctx = init_training_script(s_args, s_cfg, m_cfg, d_args, script_prefix="smpo", trainer_cls=SMPOTrainer)
ds_s, ds_presharded = load_script_datasets(s_args, ctx.parallelism_config)
is_vlm_s = resolve_preference_vlm_run(s_args, m_cfg, ds_s, text_only_model=s_args.text_only_model, method="SMPO")
policy = load_policy_model(ctx, s_cfg, m_cfg, s_args, vlm_run=is_vlm_s, attn_default=default_padded_attn_implementation(m_cfg, sinks_reset=s_args.reset_sinks, context_parallel=ctx.parallelism_config.is_cp_mode))
tok = apply_max_length(s_cfg, s_args, policy.model, policy.tokenizer)
proc = install_resolved_tokenizer(policy.processing_class, tok)
apply_text_path_padding_side(tok, is_vlm_s)
peft_cfg = configure_adapter_training(s_args, policy.model, m_cfg, "CAUSAL_LM")
apply_vision_options(s_args, policy.model, m_cfg, proc, is_vlm=is_vlm_s)
train_ds, eval_ds, gen_cb = prepare_script_preference_data(s_args, s_cfg, ds_s, tok, is_vlm_data=is_vlm_s, generation_max_prompt_length=s_cfg.max_prompt_length)
cbs = configure_balancing_and_build_callbacks(s_args, s_cfg, policy.model, ctx, generate_callback=gen_cb)
trainer = SMPOTrainer(
    model=policy.model, args=s_cfg, train_dataset=train_ds, eval_dataset=eval_ds,
    processing_class=proc, peft_config=peft_cfg, callbacks=cbs,
    **distributed_trainer_kwargs(s_args, ctx.parallelism_config, dataset_presharded=ds_presharded), is_vlm=is_vlm_s,
)
mdl = trainer.model
ds_rows = load_from_disk(str(work / "datasets" / "halo_vlm_sft_smoke"))["train"]
sample_rows = [ds_rows[0], ds_rows[min(1, len(ds_rows) - 1)]]
collected = []
mdl.eval()
with torch.no_grad():
    for row in sample_rows:
        img = row["images"][0].convert("RGB")
        ref_text = row["messages"][1]["content"][0]["text"]
        prompt = proc.apply_chat_template([{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": DIPLOMATIC_PROMPT}]}], tokenize=False, add_generation_prompt=True)
        p_in = proc(text=[prompt], images=[img], return_tensors="pt")
        f_in = proc(text=[prompt + ref_text + "<turn|>"], images=[img], return_tensors="pt")
        dev_in = {k: v.to(mdl.device) if torch.is_tensor(v) else v for k, v in f_in.items()}
        logits = mdl(**dev_in).logits[0]
        p_len = p_in["input_ids"].shape[1]
        lps = torch.log_softmax(logits[p_len - 1 : -1].float(), dim=-1).gather(-1, dev_in["input_ids"][0, p_len:].unsqueeze(-1)).squeeze(-1)
        collected.extend(float(x) for x in lps[:16].cpu().tolist())
probe_out.write_text(json.dumps(collected), encoding="utf-8")
"""
probe_json = work / "configs" / "stage2_probe_lps.json"
subprocess.check_call([sys.executable, "-c", stage2_probe, str(work), str(probe_json)])
stage2_lps = json.loads(probe_json.read_text(encoding="utf-8"))
base_tol = 0.05 if vram_gb >= 56.0 else 0.15
s2_tol = min(0.15 if vram_gb >= 56.0 else 0.35, max(base_tol, 0.4 * sft_shift))
stage2_vs_base_diff = max(abs(a - b) for a, b in zip(base_lps, stage2_lps))
report = verify_stage_continuity_logprobs(base_lps, sft_lps, stage2_lps, tol=s2_tol)
if sft_shift < 2.0 * s2_tol:
    print(f"[WARN] Low Stage 2 continuity separation power: sft_shift={sft_shift:.6f} < 2*s2_tol ({2.0 * s2_tol:.6f})")
if report["max_step0_diff"] > 0.05:
    print(f"[WARN] Stage 2 step-0 vs merged SFT max logit diff={report['max_step0_diff']:.6f} > 0.05 (tol={s2_tol:.6f}, sft_shift={sft_shift:.6f})")
assert report["max_step0_diff"] < 0.5 * sft_shift and stage2_vs_base_diff >= 0.5 * sft_shift, (
    f"Stage 2 continuity lacks power / fell back toward base (step0_diff={report['max_step0_diff']:.6f}, stage2_vs_base={stage2_vs_base_diff:.6f}, sft_shift={sft_shift:.6f})"
)
report["s2_tol"] = round(s2_tol, 6)
report["stage2_vs_base_diff"] = round(stage2_vs_base_diff, 6)
report["live_vs_merged_diff"] = merge_diff
print("Verified Stage 1->2 continuity (via Halo SMPOTrainer):", report)
PY

HALO_FLEX_SLIDING="${FLEX_ENV}" "${SMOKE_VENV}/bin/python" "${HALO_DIR}/scripts/training/preference/smpo.py" "${WORK_DIR}/configs/smoke_vlm_smpo.yaml" 2>&1 | tee "${WORK_DIR}/configs/smpo_run.log"
"${SMOKE_VENV}/bin/python" -c "from pathlib import Path; from eahtr.halo_smoke import verify_halo_run_log_finite_loss; print('SMPO losses:', verify_halo_run_log_finite_loss(Path('${WORK_DIR}/configs/smpo_run.log'), min_steps=5))"
