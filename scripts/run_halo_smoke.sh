#!/usr/bin/env bash
set -euo pipefail

# Clean-room 1x GPU 5-step Halo VLM SFT + SMPO smoke harness.
# Bootstraps Python 3.12 via uv and installs PyTorch 2.11.0 matching the active NVIDIA driver.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HALO_DIR="${HALO_DIR:-/tmp/halo}"
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

echo "Staging high-resolution Gemma 4 processor dir and smoke Arrow datasets..."
"${SMOKE_VENV}/bin/python" - <<PY
import hashlib, io, json, zipfile
from pathlib import Path
from urllib.request import Request, urlopen
from huggingface_hub import snapshot_download
from transformers import AutoProcessor, AutoTokenizer
from eahtr.corpus import filter_corpus_record, parse_alto_xml_lines
from eahtr.halo_smoke import (
    build_smoke_yaml_configs,
    export_halo_vlm_smoke_datasets,
    prepare_gemma4_highres_dir,
    verify_processor_image_tokens,
)
from eahtr.hygiene import clean_diplomatic_target

work = Path("${WORK_DIR}")
repo = Path("${REPO_ROOT}")
vram_gb = float("${VRAM_GB}")
cache_dir = Path(snapshot_download("google/gemma-4-12B-it", allow_patterns=["*.json", "*.jinja", "*.model", "*.safetensors*"]))
staged_dir = prepare_gemma4_highres_dir(cache_dir, work / "gemma4_12b_highres")

zip_path = work / "revcity-htr.zip"
if not zip_path.exists():
    req = Request("https://zenodo.org/api/records/15776323/files/revcity-htr.zip/content", headers={"User-Agent": "eahtr-smoke/0.1"})
    zip_path.write_bytes(urlopen(req, timeout=120).read())

records = []
with zipfile.ZipFile(zip_path) as zf:
    names = set(zf.namelist())
    img_dir = work / "images"
    img_dir.mkdir(exist_ok=True)
    for stem in sorted(n[:-4] for n in names if n.endswith(".xml") and (n[:-4] + ".jpg") in names):
        lines = parse_alto_xml_lines(zf.read(stem + ".xml").decode("utf-8", errors="replace"))
        raw = "\n".join(lines)
        ok, _ = filter_corpus_record({"source_name": "Revolutionary City", "license": "CC-BY-4.0", "text": raw})
        if not ok:
            continue
        cleaned = clean_diplomatic_target(raw)
        if len(cleaned.split()) < 3:
            continue
        out_img = img_dir / f"{Path(stem).name}.jpg"
        if not out_img.exists():
            out_img.write_bytes(zf.read(stem + ".jpg"))
        records.append({"image_path": str(out_img), "text": cleaned})
        if len(records) >= 40:
            break

proc = AutoProcessor.from_pretrained(str(staged_dir))
realized_toks = verify_processor_image_tokens(proc, Path(records[0]["image_path"]))
print(f"Verified realized Gemma 4 processor image tokens on sample page: {realized_toks}")

tok = getattr(proc, "tokenizer", None) or AutoTokenizer.from_pretrained(str(staged_dir))
max_len = 2048 if vram_gb < 32.0 else 4096
max_prompt = 1350 if vram_gb < 32.0 else 1600
sft_ds, smpo_ds, dropped = export_halo_vlm_smoke_datasets(
    records, work / "datasets", lambda s: len(tok(s, add_special_tokens=False)["input_ids"]),
    max_length=max_len, max_prompt_length=max_prompt,
)
sft_yaml, smpo_yaml = build_smoke_yaml_configs(
    repo / "configs" / "halo" / "gemma4_12b_sft.yaml",
    repo / "configs" / "halo" / "gemma4_12b_vlm_smpo.yaml",
    work / "configs", model_dir=staged_dir, sft_dataset_dir=sft_ds, smpo_dataset_dir=smpo_ds, vram_gb=vram_gb, repo_root=repo,
)
print(f"Prepared smoke YAMLs (dropped={dropped}): {sft_yaml} | {smpo_yaml}")
PY

cd "${REPO_ROOT}"
"${SMOKE_VENV}/bin/python" "${REPO_ROOT}/scripts/validate_halo_configs.py" --config-dir "${WORK_DIR}/configs" --halo-repo "${HALO_DIR}"
HALO_FLEX_SLIDING="${FLEX_ENV}" "${SMOKE_VENV}/bin/python" "${HALO_DIR}/scripts/training/sft.py" "${WORK_DIR}/configs/smoke_sft.yaml" 2>&1 | tee "${WORK_DIR}/configs/sft_run.log"
"${SMOKE_VENV}/bin/python" -c "from pathlib import Path; from eahtr.halo_smoke import verify_halo_run_log_finite_loss; print('Verified SFT finite losses:', verify_halo_run_log_finite_loss(Path('${WORK_DIR}/configs/sft_run.log')))"
HALO_FLEX_SLIDING="${FLEX_ENV}" "${SMOKE_VENV}/bin/python" "${HALO_DIR}/scripts/training/preference/smpo.py" "${WORK_DIR}/configs/smoke_vlm_smpo.yaml" 2>&1 | tee "${WORK_DIR}/configs/smpo_run.log"
"${SMOKE_VENV}/bin/python" -c "from pathlib import Path; from eahtr.halo_smoke import verify_halo_run_log_finite_loss; print('Verified SMPO finite losses:', verify_halo_run_log_finite_loss(Path('${WORK_DIR}/configs/smpo_run.log')))"
