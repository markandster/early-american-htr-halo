#!/usr/bin/env bash
set -euo pipefail

# Canonical Halo Gemma 4 launcher: auto-stages 1120 vision soft tokens at pinned HF revisions,
# enforces EP/nproc parity and fresh non-colliding output dirs, validates in an isolated dir,
# runs SFT or VLM SMPO, and converts/merges outputs with export parity verification.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HALO_DIR="${HALO_DIR:-/tmp/halo}"
STAGE="${STAGE:-sft}"                     # sft | smpo
MODEL_SIZE="${MODEL_SIZE:-12b}"           # 12b | 26b_a4b | 31b
WORK_DIR="${WORK_DIR:-${REPO_ROOT}/checkpoints/halo_runs}"
DATASET_DIR="${DATASET_DIR:?Set DATASET_DIR to exported Halo Arrow DatasetDict}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
RESUME="${RESUME:-0}"
EP_SIZE="${EP_SIZE:-}"
MAX_STEPS="${MAX_STEPS:-}"
PILOT_MODE="${PILOT_MODE:-0}"

if [[ "${MODEL_SIZE}" == "12b" ]]; then
  HF_REPO="google/gemma-4-12B-it"
  HF_REV="43c9d014b231"
  DEFAULT_NUM_GPUS=1
  BASE_YAML="${REPO_ROOT}/configs/halo/gemma4_12b_${STAGE/smpo/vlm_smpo}.yaml"
  DEFAULT_SFT_MERGED="${REPO_ROOT}/checkpoints/gemma4-12b-eahtr-sft-merged"
elif [[ "${MODEL_SIZE}" == "26b_a4b" ]]; then
  HF_REPO="google/gemma-4-26B-A4B-it"
  HF_REV="4d7ae4984b7db7de8f8457170b3f1a419ee76d52"
  DEFAULT_NUM_GPUS=8
  BASE_YAML="${REPO_ROOT}/configs/halo/gemma4_26b_a4b_${STAGE/smpo/vlm_smpo}.yaml"
  DEFAULT_SFT_MERGED="${REPO_ROOT}/checkpoints/gemma4-26b-a4b-eahtr-sft-bf16"
elif [[ "${MODEL_SIZE}" == "31b" ]]; then
  HF_REPO="google/gemma-4-31B-it"
  HF_REV="842da3794eaa0b77d5f08bae87a17459d91ff475"
  DEFAULT_NUM_GPUS=2
  BASE_YAML="${REPO_ROOT}/configs/halo/gemma4_31b_${STAGE/smpo/vlm_smpo}.yaml"
  DEFAULT_SFT_MERGED="${REPO_ROOT}/checkpoints/gemma4-31b-eahtr-sft-merged"
else
  echo "ERROR: MODEL_SIZE must be one of 12b, 26b_a4b, 31b (got ${MODEL_SIZE})" >&2
  exit 1
fi

if [[ "${STAGE}" != "sft" && "${STAGE}" != "smpo" ]]; then
  echo "ERROR: STAGE must be 'sft' or 'smpo' (got ${STAGE})" >&2
  exit 1
fi

NUM_GPUS="${NUM_GPUS:-${DEFAULT_NUM_GPUS}}"
ADAPTER_DIR="${ADAPTER_DIR:-${WORK_DIR}/gemma4-${MODEL_SIZE}-eahtr-${STAGE}-${RUN_ID}}"
MERGED_DIR="${MERGED_DIR:-${REPO_ROOT}/checkpoints/gemma4-${MODEL_SIZE}-eahtr-${STAGE}-merged-${RUN_ID}}"

if [[ "${RESUME}" != "1" && -d "${ADAPTER_DIR}" ]] && [[ -n "$(ls -A "${ADAPTER_DIR}" 2>/dev/null)" ]]; then
  echo "ERROR: Output dir ${ADAPTER_DIR} is non-empty (set RESUME=1 or pass a fresh ADAPTER_DIR/RUN_ID)." >&2
  exit 1
fi
if [[ -d "${MERGED_DIR}" ]] && [[ -n "$(ls -A "${MERGED_DIR}" 2>/dev/null)" ]]; then
  echo "ERROR: Final export dir ${MERGED_DIR} already exists and is non-empty (pass a fresh MERGED_DIR/RUN_ID)." >&2
  exit 1
fi

if [[ "${PILOT_MODE}" == "1" && -z "${MAX_STEPS}" ]]; then
  MAX_STEPS=5
fi

STAGED_BASE="${WORK_DIR}/gemma4-${MODEL_SIZE}-highres-1120"
RUN_CFG_DIR="${WORK_DIR}/run_cfg_${MODEL_SIZE}_${STAGE}_${RUN_ID}"
mkdir -p "${WORK_DIR}" "${RUN_CFG_DIR}"
python3 "${REPO_ROOT}/scripts/stage_gemma4_highres.py" \
  --model "${HF_REPO}" --revision "${HF_REV}" --out-dir "${STAGED_BASE}" --max-soft-tokens 1120 \
  --chat-template "${REPO_ROOT}/configs/halo/templates/gemma4_htr_nothink.jinja"

RUN_YAML="${RUN_CFG_DIR}/$(basename "${BASE_YAML}")"
SFT_MERGED_DIR="${SFT_MERGED_DIR:-${DEFAULT_SFT_MERGED}}"
python3 - <<PY
from pathlib import Path
import yaml

cfg = yaml.safe_load(Path("${BASE_YAML}").read_text(encoding="utf-8"))
cfg["model_name_or_path"] = "${STAGED_BASE}" if "${STAGE}" == "sft" else "${SFT_MERGED_DIR}"
cfg["dataset"] = ["${DATASET_DIR}"]
cfg["output_dir"] = "${ADAPTER_DIR}"
if "${EP_SIZE}":
    cfg["expert_parallel_size"] = int("${EP_SIZE}")
if "${MAX_STEPS}":
    ms = int("${MAX_STEPS}")
    cfg["max_steps"] = ms
    cfg["save_strategy"] = "steps"
    cfg["save_steps"] = min(int(cfg.get("save_steps", ms)), ms)
    cfg["eval_steps"] = min(int(cfg.get("eval_steps", ms)), ms)
if isinstance(cfg.get("chat_template"), str) and not Path(cfg["chat_template"]).is_absolute():
    cfg["chat_template"] = str((Path("${REPO_ROOT}") / cfg["chat_template"]).resolve())
Path("${RUN_YAML}").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
PY

read -r RESOLVED_EP USE_PEFT < <(python3 -c "import yaml; c=yaml.safe_load(open('${RUN_YAML}')); print(int(c.get('expert_parallel_size', 1) or 1), 'true' if c.get('use_peft') else 'false')")
if [[ "${RESOLVED_EP}" -gt 1 ]]; then
  if [[ "${NUM_GPUS}" -ne "${RESOLVED_EP}" ]] && ! ( [[ "${RESOLVED_EP}" -eq 2 && "${NUM_GPUS}" -ge 2 && $((NUM_GPUS % 2)) -eq 0 ]] ); then
    echo "ERROR: Invalid GPU count NUM_GPUS=${NUM_GPUS} for expert_parallel_size=${RESOLVED_EP} (Halo requires nproc == EP, or EP == 2 with even nproc)." >&2
    exit 1
  fi
fi

python3 "${REPO_ROOT}/scripts/validate_halo_configs.py" --config-dir "${RUN_CFG_DIR}" --halo-repo "${HALO_DIR}"
TRAIN_SCRIPT="${HALO_DIR}/scripts/training/sft.py"
[[ "${STAGE}" == "smpo" ]] && TRAIN_SCRIPT="${HALO_DIR}/scripts/training/preference/smpo.py"

LOG_FILE="${WORK_DIR}/run_${MODEL_SIZE}_${STAGE}_${RUN_ID}.log"
if [[ "${RESOLVED_EP}" -gt 1 || "${NUM_GPUS}" -gt 1 ]]; then
  torchrun --nproc_per_node="${NUM_GPUS}" "${TRAIN_SCRIPT}" "${RUN_YAML}" 2>&1 | tee "${LOG_FILE}"
else
  python3 "${TRAIN_SCRIPT}" "${RUN_YAML}" 2>&1 | tee "${LOG_FILE}"
fi

MIN_LOG_STEPS="${MAX_STEPS:-5}"
[[ "${MIN_LOG_STEPS}" -gt 5 ]] && MIN_LOG_STEPS=5
python3 -c "from pathlib import Path; from eahtr.halo_smoke import verify_halo_run_log_finite_loss; print('Losses verified:', verify_halo_run_log_finite_loss(Path('${LOG_FILE}'), min_steps=${MIN_LOG_STEPS}))"

if [[ "${USE_PEFT}" == "true" ]]; then
  RESOLVED_ADAPTER="$(python3 -c "from pathlib import Path; d=Path('${ADAPTER_DIR}'); cands=[d]+[p for p in sorted(d.glob('checkpoint-*'), key=lambda p: int(p.name.split('-')[-1]) if p.name.split('-')[-1].isdigit() else -1, reverse=True) if p.is_dir()]; print(next(str(p) for p in cands if (p/'adapter_model.safetensors').is_file()))")"
  python3 "${HALO_DIR}/scripts/after_training/merge_peft_adapters.py" \
    --adapter_dir "${RESOLVED_ADAPTER}" --output_dir "${MERGED_DIR}"
  python3 "${REPO_ROOT}/scripts/verify_mlx_export.py" "${MERGED_DIR}" --reference-dir "${RESOLVED_ADAPTER}"
else
  RESOLVED_CKPT="$(python3 -c "from pathlib import Path; d=Path('${ADAPTER_DIR}'); cands=[d]+[p for p in sorted(d.glob('checkpoint-*'), key=lambda p: int(p.name.split('-')[-1]) if p.name.split('-')[-1].isdigit() else -1, reverse=True) if p.is_dir()]; print(next(str(p) for p in cands if (p/'config.json').is_file() and ((p/'model.safetensors').is_file() or (p/'model.safetensors.index.json').is_file())))")"
  python3 "${HALO_DIR}/scripts/after_training/convert_to_bf16.py" \
    --input_dir "${RESOLVED_CKPT}" --output_dir "${MERGED_DIR}" --verify
  python3 - <<PY
import shutil
from pathlib import Path
src = Path("${STAGED_BASE}")
dst = Path("${MERGED_DIR}")
for fname in ("processor_config.json", "chat_template.jinja"):
    if (src / fname).is_file():
        shutil.copy2(src / fname, dst / fname)
PY
  python3 "${REPO_ROOT}/scripts/verify_mlx_export.py" "${MERGED_DIR}" --reference-dir "${STAGED_BASE}"
fi
