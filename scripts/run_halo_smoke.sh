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
uv pip install --python "${SMOKE_VENV}/bin/python" --no-deps -e "${HALO_DIR}"
"${SMOKE_VENV}/bin/python" -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable after PyTorch install'"
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
staged_dir = prepare_gemma4_highres_dir(cache_dir, work / "gemma4_12b_highres")

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
"${SMOKE_VENV}/bin/python" "${REPO_ROOT}/scripts/verify_mlx_export.py" "${WORK_DIR}/configs/sft_merged" --reference-dir "${WORK_DIR}/gemma4_12b_highres"

echo "Checking Stage 1 -> Stage 2 fresh-process log-prob continuity and stop termination..."
"${SMOKE_VENV}/bin/python" - <<PY
import json, subprocess, sys
from pathlib import Path
import torch
from datasets import load_from_disk
from safetensors.torch import load_file
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from eahtr.candidates import DIPLOMATIC_PROMPT
from eahtr.halo_smoke import verify_stage_continuity_logprobs

work = Path("${WORK_DIR}")
sft_ds_path = work / "datasets" / "halo_vlm_sft_smoke"
ds = load_from_disk(str(sft_ds_path))["train"]
row0 = ds[0]
img = row0["images"][0].convert("RGB")
ref_text = row0["messages"][1]["content"][0]["text"]

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

def _score_mean_lp(model_path: Path, check_generate: bool = False) -> list[float]:
    proc = AutoProcessor.from_pretrained(str(model_path))
    prompt = proc.apply_chat_template(
        [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": DIPLOMATIC_PROMPT}]}],
        tokenize=False, add_generation_prompt=True,
    )
    full = prompt + ref_text + "<turn|>"
    p_in = proc(text=[prompt], images=[img], return_tensors="pt")
    f_in = proc(text=[full], images=[img], return_tensors="pt")
    load_kw = {"torch_dtype": torch.bfloat16, "device_map": "auto", "attn_implementation": "sdpa"}
    if bnb_cfg is not None:
        load_kw["quantization_config"] = bnb_cfg
    mdl = AutoModelForImageTextToText.from_pretrained(str(model_path), **load_kw)
    mdl.eval()
    with torch.no_grad():
        dev_in = {k: v.to(mdl.device) if torch.is_tensor(v) else v for k, v in f_in.items()}
        logits = mdl(**dev_in).logits[0]
        p_len = p_in["input_ids"].shape[1]
        tgt = dev_in["input_ids"][0, p_len:]
        lps = torch.log_softmax(logits[p_len - 1 : -1].float(), dim=-1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        if check_generate:
            out = mdl.generate(**{k: v.to(mdl.device) if torch.is_tensor(v) else v for k, v in p_in.items()}, max_new_tokens=128, do_sample=False)
            gen_ids = out[:, p_len:]
            decoded = proc.batch_decode(gen_ids, skip_special_tokens=False)[0]
            assert "<|channel>thought" not in decoded, f"Thinking leak in Stage 1 smoke generate: {decoded[:120]}"
            tok = getattr(proc, "tokenizer", proc)
            eot_id = tok.convert_tokens_to_ids("<turn|>")
            assert int(gen_ids.shape[1]) < 128 or eot_id in gen_ids[0].tolist(), "Stage 1 smoke generate hit 128-token cap without stopping"
    del mdl
    torch.cuda.empty_cache()
    return [float(x) for x in lps[:16].cpu().tolist()]

base_lps = _score_mean_lp(work / "gemma4_12b_highres")
sft_lps = _score_mean_lp(work / "configs" / "sft_merged", check_generate=True)

stage2_probe = """
import json, sys
from pathlib import Path
import torch
from datasets import load_from_disk
from peft import get_peft_model
from halo.configs.parallelism_arguments import ParallelismArguments
from halo.configs.preference import SMPOConfig, SMPOScriptArguments
from halo.configs.script_arguments import ModelArguments
from halo.entrypoints.data_setup import apply_vision_options, load_script_datasets, resolve_preference_vlm_run
from halo.entrypoints.model_setup import default_padded_attn_implementation, load_policy_model
from halo.entrypoints.script import init_training_script, parse_script_args
from halo.model_loading.parallel.peft_setup import configure_adapter_training
from halo.trainers.preference.smpo import SMPOTrainer
from eahtr.candidates import DIPLOMATIC_PROMPT

work = Path(sys.argv[1])
sys.argv = ["smpo.py", str(work / "configs" / "smoke_vlm_smpo.yaml")]
s_args, s_cfg, m_cfg, d_args = parse_script_args(SMPOScriptArguments, SMPOConfig, ModelArguments, ParallelismArguments)
ctx = init_training_script(s_args, s_cfg, m_cfg, d_args, script_prefix="smpo", trainer_cls=SMPOTrainer)
ds_s, _ = load_script_datasets(s_args, ctx.parallelism_config)
is_vlm_s = resolve_preference_vlm_run(s_args, m_cfg, ds_s, text_only_model=s_args.text_only_model, method="SMPO")
policy = load_policy_model(ctx, s_cfg, m_cfg, s_args, vlm_run=is_vlm_s, attn_default=default_padded_attn_implementation(m_cfg, sinks_reset=s_args.reset_sinks, context_parallel=False))
mdl = policy.model
peft_cfg = configure_adapter_training(s_args, mdl, m_cfg, "CAUSAL_LM")
if peft_cfg is not None:
    mdl = get_peft_model(mdl, peft_cfg)
apply_vision_options(s_args, mdl, m_cfg, policy.processing_class, is_vlm=is_vlm_s)
proc = policy.processing_class
row0 = load_from_disk(str(work / "datasets" / "halo_vlm_sft_smoke"))["train"][0]
img = row0["images"][0].convert("RGB")
ref_text = row0["messages"][1]["content"][0]["text"]
prompt = proc.apply_chat_template([{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": DIPLOMATIC_PROMPT}]}], tokenize=False, add_generation_prompt=True)
p_in = proc(text=[prompt], images=[img], return_tensors="pt")
f_in = proc(text=[prompt + ref_text + "<turn|>"], images=[img], return_tensors="pt")
mdl.eval()
with torch.no_grad():
    dev_in = {k: v.to(mdl.device) if torch.is_tensor(v) else v for k, v in f_in.items()}
    logits = mdl(**dev_in).logits[0]
    p_len = p_in["input_ids"].shape[1]
    lps = torch.log_softmax(logits[p_len - 1 : -1].float(), dim=-1).gather(-1, dev_in["input_ids"][0, p_len:].unsqueeze(-1)).squeeze(-1)
print(json.dumps([float(x) for x in lps[:16].cpu().tolist()]))
"""
raw_s2 = subprocess.check_output([sys.executable, "-c", stage2_probe, str(work)], text=True).strip().splitlines()[-1]
stage2_lps = json.loads(raw_s2)
report = verify_stage_continuity_logprobs(base_lps, sft_lps, stage2_lps)
print("Verified Stage 1->2 continuity (via Halo SMPO loader):", report)
PY

HALO_FLEX_SLIDING="${FLEX_ENV}" "${SMOKE_VENV}/bin/python" "${HALO_DIR}/scripts/training/preference/smpo.py" "${WORK_DIR}/configs/smoke_vlm_smpo.yaml" 2>&1 | tee "${WORK_DIR}/configs/smpo_run.log"
"${SMOKE_VENV}/bin/python" -c "from pathlib import Path; from eahtr.halo_smoke import verify_halo_run_log_finite_loss; print('SMPO losses:', verify_halo_run_log_finite_loss(Path('${WORK_DIR}/configs/smpo_run.log'), min_steps=5))"
