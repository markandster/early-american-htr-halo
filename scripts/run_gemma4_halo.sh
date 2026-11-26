#!/usr/bin/env bash
set -euo pipefail

# Canonical Halo Gemma 4 launcher: auto-stages 1120 vision soft tokens at pinned HF revisions,
# validates YAML, runs SFT or VLM SMPO, and merges PEFT adapters with export parity checks.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HALO_DIR="${HALO_DIR:-/tmp/halo}"
STAGE="${STAGE:-sft}"                     # sft | smpo
MODEL_SIZE="${MODEL_SIZE:-12b}"           # 12b | 26b
WORK_DIR="${WORK_DIR:-${REPO_ROOT}/checkpoints/halo_runs}"
DATASET_DIR="${DATASET_DIR:?Set DATASET_DIR to exported Halo Arrow DatasetDict}"
ADAPTER_DIR="${ADAPTER_DIR:-${WORK_DIR}/gemma4-${MODEL_SIZE}-eahtr-${STAGE}}"
MERGED_DIR="${MERGED_DIR:-${REPO_ROOT}/checkpoints/gemma4-${MODEL_SIZE}-eahtr-${STAGE}-merged}"
NUM_GPUS="${NUM_GPUS:-1}"

if [[ "${MODEL_SIZE}" == "12b" ]]; then
  HF_REPO="google/gemma-4-12B-it"
  HF_REV="43c9d014b231"
  BASE_YAML="${REPO_ROOT}/configs/halo/gemma4_12b_${STAGE/smpo/vlm_smpo}.yaml"
  DEFAULT_SFT_MERGED="${REPO_ROOT}/checkpoints/gemma4-12b-eahtr-sft-merged"
elif [[ "${MODEL_SIZE}" == "26b" ]]; then
  HF_REPO="google/gemma-4-26B-A4B-it"
  HF_REV="4d7ae4984b7d"
  BASE_YAML="${REPO_ROOT}/configs/halo/gemma4_26b_a4b_${STAGE/smpo/vlm_smpo}.yaml"
  DEFAULT_SFT_MERGED="${REPO_ROOT}/checkpoints/gemma4-26b-a4b-eahtr-sft-merged"
else
  echo "ERROR: MODEL_SIZE must be '12b' or '26b' (got ${MODEL_SIZE})" >&2
  exit 1
fi

STAGED_BASE="${WORK_DIR}/gemma4-${MODEL_SIZE}-highres-1120"
mkdir -p "${WORK_DIR}"
python3 "${REPO_ROOT}/scripts/stage_gemma4_highres.py" \
  --model "${HF_REPO}" --revision "${HF_REV}" --out-dir "${STAGED_BASE}" --max-soft-tokens 1120

RUN_YAML="${WORK_DIR}/run_${MODEL_SIZE}_${STAGE}.yaml"
SFT_MERGED_DIR="${SFT_MERGED_DIR:-${DEFAULT_SFT_MERGED}}"
python3 - <<PY
from pathlib import Path
import yaml

cfg = yaml.safe_load(Path("${BASE_YAML}").read_text(encoding="utf-8"))
cfg["model_name_or_path"] = "${STAGED_BASE}" if "${STAGE}" == "sft" else "${SFT_MERGED_DIR}"
cfg["dataset"] = ["${DATASET_DIR}"]
cfg["output_dir"] = "${ADAPTER_DIR}"
if isinstance(cfg.get("chat_template"), str) and not Path(cfg["chat_template"]).is_absolute():
    cfg["chat_template"] = str((Path("${REPO_ROOT}") / cfg["chat_template"]).resolve())
Path("${RUN_YAML}").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
PY

python3 "${REPO_ROOT}/scripts/validate_halo_configs.py" --config-dir "${WORK_DIR}" --halo-repo "${HALO_DIR}"
TRAIN_SCRIPT="${HALO_DIR}/scripts/training/sft.py"
[[ "${STAGE}" == "smpo" ]] && TRAIN_SCRIPT="${HALO_DIR}/scripts/training/preference/smpo.py"

LOG_FILE="${WORK_DIR}/run_${MODEL_SIZE}_${STAGE}.log"
if [[ "${NUM_GPUS}" -gt 1 ]]; then
  accelerate launch --num_processes "${NUM_GPUS}" "${TRAIN_SCRIPT}" "${RUN_YAML}" 2>&1 | tee "${LOG_FILE}"
else
  python3 "${TRAIN_SCRIPT}" "${RUN_YAML}" 2>&1 | tee "${LOG_FILE}"
fi

python3 -c "from pathlib import Path; from eahtr.halo_smoke import verify_halo_run_log_finite_loss; print('Losses verified:', verify_halo_run_log_finite_loss(Path('${LOG_FILE}'), min_steps=5))"
RESOLVED_ADAPTER="$(python3 -c "from pathlib import Path; d=Path('${ADAPTER_DIR}'); cands=[d]+[p for p in sorted(d.glob('checkpoint-*'), key=lambda p: int(p.name.split('-')[-1]) if p.name.split('-')[-1].isdigit() else -1, reverse=True) if p.is_dir()]; print(next(str(p) for p in cands if (p/'adapter_model.safetensors').is_file()))")"
python3 "${HALO_DIR}/scripts/after_training/merge_peft_adapters.py" \
  --adapter_dir "${RESOLVED_ADAPTER}" --output_dir "${MERGED_DIR}"
python3 "${REPO_ROOT}/scripts/verify_mlx_export.py" "${MERGED_DIR}" --reference-dir "${STAGED_BASE}"
