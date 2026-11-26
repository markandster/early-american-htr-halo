#!/usr/bin/env bash
set -euo pipefail

# Canonical Halo Inkling-2-Small launcher: enforces EP/nproc parity, fresh non-colliding
# output directories, isolated Tier-1/Tier-2 config validation, finite-loss log checks,
# vLLM engine-adapter export (must not pre-exist), strict allow-list validation, and
# optional vLLM server evaluation (with --swap-images visual-grounding check).

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
HALO_DIR="${HALO_DIR:-/tmp/halo}"
STAGE="${STAGE:-sft}"                              # sft | smpo
WORK_DIR="${WORK_DIR:-${REPO_ROOT}/checkpoints/halo_runs}"
DATASET_DIR="${DATASET_DIR:?Set DATASET_DIR to exported Inkling Halo Arrow DatasetDict}"
MODEL_DIR="${MODEL_DIR:-thinkingmachines/Inkling-2-Small}"
HF_REV="${HF_REV:-885d78bfca61a7e9a498e33cfa7ad35850533b8c}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
RESUME="${RESUME:-0}"
EP_SIZE="${EP_SIZE:-}"
NUM_GPUS="${NUM_GPUS:-8}"
MAX_STEPS="${MAX_STEPS:-}"
PILOT_MODE="${PILOT_MODE:-0}"
VLLM_BASE_URL="${VLLM_BASE_URL:-}"
VLLM_MODEL="${VLLM_MODEL:-inkling2-sft}"
EVAL_JSONL="${EVAL_JSONL:-}"
IMAGE_ROOT="${IMAGE_ROOT:-}"

if [[ "${STAGE}" != "sft" && "${STAGE}" != "smpo" ]]; then
  echo "ERROR: STAGE must be 'sft' or 'smpo' (got ${STAGE})" >&2
  exit 1
fi

BASE_YAML="${REPO_ROOT}/configs/halo/inkling2_small_${STAGE/smpo/vlm_smpo}.yaml"
ADAPTER_DIR="${ADAPTER_DIR:-${WORK_DIR}/inkling2-small-eahtr-${STAGE}-${RUN_ID}}"
ENGINE_ADAPTER_DIR="${ENGINE_ADAPTER_DIR:-${REPO_ROOT}/checkpoints/inkling2-small-eahtr-${STAGE}-engine-${RUN_ID}}"

if [[ "${RESUME}" != "1" && -d "${ADAPTER_DIR}" ]] && [[ -n "$(ls -A "${ADAPTER_DIR}" 2>/dev/null)" ]]; then
  echo "ERROR: Output dir ${ADAPTER_DIR} is non-empty (set RESUME=1 or pass a fresh ADAPTER_DIR/RUN_ID)." >&2
  exit 1
fi
if [[ -e "${ENGINE_ADAPTER_DIR}" ]]; then
  echo "ERROR: Engine adapter output dir ${ENGINE_ADAPTER_DIR} already exists (export_engine_adapter.py requires a non-existent path)." >&2
  exit 1
fi

if [[ "${PILOT_MODE}" == "1" && -z "${MAX_STEPS}" ]]; then
  MAX_STEPS=5
fi

RUN_CFG_DIR="${WORK_DIR}/run_cfg_inkling2_${STAGE}_${RUN_ID}"
mkdir -p "${WORK_DIR}" "${RUN_CFG_DIR}" "$(dirname "${ENGINE_ADAPTER_DIR}")"

# Preserve YAML basename so family-keyed Tier-1 and Tier-2 rules inspect inkling2_small_*.yaml
RUN_YAML="${RUN_CFG_DIR}/$(basename "${BASE_YAML}")"
"${PYTHON_BIN}" - <<PY
from pathlib import Path
import yaml

cfg = yaml.safe_load(Path("${BASE_YAML}").read_text(encoding="utf-8"))
cfg["model_name_or_path"] = "${MODEL_DIR}"
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

RESOLVED_EP="$("${PYTHON_BIN}" -c "import yaml; c=yaml.safe_load(open('${RUN_YAML}')); print(int(c.get('expert_parallel_size', 1) or 1))")"
if [[ "${RESOLVED_EP}" -gt 1 ]]; then
  if [[ "${NUM_GPUS}" -ne "${RESOLVED_EP}" ]] && ! ( [[ "${RESOLVED_EP}" -eq 2 && "${NUM_GPUS}" -ge 2 && $((NUM_GPUS % 2)) -eq 0 ]] ); then
    echo "ERROR: Invalid GPU count NUM_GPUS=${NUM_GPUS} for expert_parallel_size=${RESOLVED_EP} (Halo requires nproc == EP, or EP == 2 with even nproc)." >&2
    exit 1
  fi
fi

"${PYTHON_BIN}" "${REPO_ROOT}/scripts/validate_halo_configs.py" --config-dir "${RUN_CFG_DIR}" --halo-repo "${HALO_DIR}"
TRAIN_SCRIPT="${HALO_DIR}/scripts/training/sft.py"
[[ "${STAGE}" == "smpo" ]] && TRAIN_SCRIPT="${HALO_DIR}/scripts/training/preference/smpo.py"

LOG_FILE="${WORK_DIR}/run_inkling2_${STAGE}_${RUN_ID}.log"
if [[ "${RESOLVED_EP}" -gt 1 || "${NUM_GPUS}" -gt 1 ]]; then
  torchrun --nproc_per_node="${NUM_GPUS}" "${TRAIN_SCRIPT}" "${RUN_YAML}" 2>&1 | tee "${LOG_FILE}"
else
  "${PYTHON_BIN}" "${TRAIN_SCRIPT}" "${RUN_YAML}" 2>&1 | tee "${LOG_FILE}"
fi

MIN_LOG_STEPS="${MAX_STEPS:-5}"
[[ "${MIN_LOG_STEPS}" -gt 5 ]] && MIN_LOG_STEPS=5
"${PYTHON_BIN}" -c "from pathlib import Path; from eahtr.halo_smoke import verify_halo_run_log_finite_loss; print('Losses verified:', verify_halo_run_log_finite_loss(Path('${LOG_FILE}'), min_steps=${MIN_LOG_STEPS}))"

RESOLVED_ADAPTER="$("${PYTHON_BIN}" -c "from pathlib import Path; d=Path('${ADAPTER_DIR}'); cands=[d]+[p for p in sorted(d.glob('checkpoint-*'), key=lambda p: int(p.name.split('-')[-1]) if p.name.split('-')[-1].isdigit() else -1, reverse=True) if p.is_dir()]; print(next(str(p) for p in cands if (p/'adapter_model.safetensors').is_file()))")"
EXTRA_REV_ARGS=()
if [[ "${MODEL_DIR}" == "thinkingmachines/Inkling-2-Small" ]]; then
  EXTRA_REV_ARGS=(--base_revision "${HF_REV}")
fi
"${PYTHON_BIN}" "${HALO_DIR}/scripts/after_training/export_engine_adapter.py" \
  --adapter_dir "${RESOLVED_ADAPTER}" \
  --output_dir "${ENGINE_ADAPTER_DIR}" \
  "${EXTRA_REV_ARGS[@]}"

"${PYTHON_BIN}" -c "from pathlib import Path; from eahtr.vllm_client import validate_vllm_engine_adapter_dir; print('Validated engine adapter:', validate_vllm_engine_adapter_dir(Path('${ENGINE_ADAPTER_DIR}')))"

if [[ -n "${VLLM_BASE_URL}" && -n "${EVAL_JSONL}" ]]; then
  IMG_ROOT_ARGS=()
  [[ -n "${IMAGE_ROOT}" ]] && IMG_ROOT_ARGS=(--image-root "${IMAGE_ROOT}")
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/eval_vllm_server.py" \
    --base-url "${VLLM_BASE_URL}" --model "${VLLM_MODEL}" \
    --eval-jsonl "${EVAL_JSONL}" "${IMG_ROOT_ARGS[@]}" \
    --engine-adapter-dir "${ENGINE_ADAPTER_DIR}" \
    --tokenizer "${MODEL_DIR}" \
    --max-image-tokens 2000 \
    --out-json "${WORK_DIR}/val_inkling2_${STAGE}_${RUN_ID}.json" \
    --dual-convention
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/eval_vllm_server.py" \
    --base-url "${VLLM_BASE_URL}" --model "${VLLM_MODEL}" \
    --eval-jsonl "${EVAL_JSONL}" "${IMG_ROOT_ARGS[@]}" \
    --engine-adapter-dir "${ENGINE_ADAPTER_DIR}" \
    --tokenizer "${MODEL_DIR}" \
    --max-image-tokens 2000 \
    --swap-images \
    --out-json "${WORK_DIR}/val_inkling2_${STAGE}_${RUN_ID}_swap.json"
fi
