"""Cells 1-3 builders for the competition Colab notebook (plus Cell 4 re-export)."""

from __future__ import annotations

from .colab_cells_corpus import build_cell_4_corpus

CELL_1_MD_SOURCE = """
# Early American HTR — Public Competition Training & Eval (Colab Pro)

**Competition-cleared public data only (`CC-BY-4.0` and `Public Domain`).**

This notebook mirrors our high-resolution, trainable-vision, sham-controlled training harness while strictly restricting datasets and models to competition-eligible open resources:

1. **Data (100% Public & Rights-Cleared):**
   - **Revolutionary City HTR Corpus** (Zenodo `15776323`, **CC-BY-4.0**) downloaded directly to Colab SSD, parsed from ALTO-XML, cleaned via `eahtr.hygiene.clean_diplomatic_target` (stripping `[...]` editorial brackets and malformed standalone `-` lines), filtered for duplicate-line slips, and split deterministically by series (`train` / `val` / `test_public`) with zero document, writer, image SHA-256, or 32-word n-gram leakage.
   - **Optional Library of Congress *By the People*** (`LCCN 2020446971`, **Public Domain**) behind `INCLUDE_LOC_BY_THE_PEOPLE = False` (skipping tabular leader-dot pages).
   - **Candidate Apache-2.0 Models:**
   - `gemma4_12b` (`google/gemma-4-12B-it`, default, ungated) — encoder-free dense VLM
   - `gemma4_26b_a4b` (`google/gemma-4-26B-A4B-it`, ungated) — 26B/4B-active MoE VLM
   - `inkling2_mini` (`thinkingmachines/Inkling-2-Mini`) — 26.4B `modelopt` NVFP4 (`inkling2_loop`, gated on upstream loader support; Halo cluster track uses `thinkingmachines/Inkling-2-Small`)
   - `qwen36_35b_a3b` (`Qwen/Qwen3.6-35B-A3B`) — 35B/3B-active MoE VLM
3. **Evaluation & Stage 2 VLM SMPO/DPO Rollout Prep:**
   - Evaluates strict/lenient WER & CER (with independent `rapidfuzz` recount), repetition loops, non-termination, editorial bracket leakage, modernization counts, silent thinking leaks, and mean verifiable `eahtr.rewards` score, plus a matched-compute `sham_short` control.
   - Optional Cell 7 generates and scores $K=4$ multi-temperature rollouts for Halo Stage 2 **VLM SMPO/DPO** (`*_vlm_preferences.jsonl`) and text-only GRPO contingency (`*_scored_offline_grpo_rollouts.jsonl`).

> **Halo Competition Rule Note (Terms §§ 4.2 & 5):** Pre-Dec-1 Colab runs establish baseline benchmarks, data hygiene, vision-gradient flow, and VLM preference rollouts on the competition corpus. Final competition submissions after Dec 1 must be reproduced by the official **Halo** training recipe (`github.com/whitecircle/halo`) under published Track Rules.
"""

CELL_2_SETUP_SOURCE = """
!pip install -q --upgrade transformers peft accelerate bitsandbytes rapidfuzz pyyaml tqdm

import csv
import gc
import hashlib
import io
import json
import math
import os
import random
import re
import subprocess
import tarfile
import time
import zipfile
from collections import Counter
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import torch
import transformers
from PIL import Image
from rapidfuzz.distance import Levenshtein
from tqdm.auto import tqdm
from huggingface_hub import login as hf_login
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
)
AutoModelForMultimodalLM = getattr(transformers, 'AutoModelForMultimodalLM', None)
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

USE_GOOGLE_DRIVE = True
if USE_GOOGLE_DRIVE:
    try:
        from google.colab import drive
        drive.mount('/content/drive', force_remount=False)
    except Exception as exc:
        raise RuntimeError(f'Google Drive mount failed while USE_GOOGLE_DRIVE=True: {exc}') from exc
    if not Path('/content/drive/MyDrive').is_dir():
        raise RuntimeError('Google Drive mount did not expose /content/drive/MyDrive while USE_GOOGLE_DRIVE=True')

REPO_URL = 'https://github.com/markandster/early-american-htr-halo.git'
REPO_DIR = Path('/content/early-american-htr-halo')
if not REPO_DIR.exists():
    subprocess.run(['git', 'clone', '--depth', '1', REPO_URL, str(REPO_DIR)], check=True)
else:
    subprocess.run(['git', '-C', str(REPO_DIR), 'pull', '--ff-only'], check=True)
subprocess.run(['pip', 'install', '-q', '--no-deps', '-e', str(REPO_DIR)], check=True)

from eahtr.colab_pipeline import (
    BENTHAM_ICFHR2016_ZENODO_URL,
    COMPETITION_CANDIDATES,
    DIPLOMATIC_PROMPT,
    LOC_WASHINGTON_ITEM_JSON_URL,
    REVCITY_ZENODO_URL,
    assign_deterministic_split,
    make_sham_derangement,
    resolve_vision_modules_to_save,
    should_use_4bit,
    validate_competition_candidate,
)
from eahtr.corpus import filter_corpus_record, parse_alto_xml_lines, parse_page_xml_lines
from eahtr.hf_dataset import push_splits_to_hf_hub
from eahtr.hygiene import audit_split_disjointness, clean_diplomatic_target
from eahtr.metrics import normalize_strict, score_pair
from eahtr.rewards import build_preference_pairs, score_group_completions, score_transcription

assert torch.cuda.is_available(), 'CUDA GPU required (select L4 24GB, A100 40GB, or A100 80GB in Colab Pro)'
GPU_NAME = torch.cuda.get_device_name(0)
VRAM_GB = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
print(f'GPU: {GPU_NAME} | VRAM: {VRAM_GB} GB | PyTorch: {torch.__version__}')

hf_token = os.environ.get('HF_TOKEN')
if not hf_token:
    try:
        from google.colab import userdata
        hf_token = userdata.get('HF_TOKEN')
    except Exception:
        hf_token = None
if hf_token:
    os.environ['HF_TOKEN'] = hf_token
    hf_login(token=hf_token, add_to_git_credential=False)
    print('Hugging Face authenticated via HF_TOKEN.')
else:
    print(
        'NOTE: HF_TOKEN not set. All active Apache-2.0 base checkpoints '
        '(google/gemma-4-12B-it, google/gemma-4-26B-A4B-it, thinkingmachines/Inkling-2-Small) are ungated; '
        'HF_TOKEN is only required when PUBLISH_TO_HF_DATASET=True.'
    )
"""

CELL_3_CONFIG_SOURCE = """
# === Run Configuration (100% Public Competition Data Only) ===
OUTPUT_ROOT = (
    Path('/content/drive/MyDrive/early-american-htr-halo-runs')
    if USE_GOOGLE_DRIVE
    else Path('/content/eahtr_runs')
)
WORK_ROOT = Path('/content/eahtr_work')
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
WORK_ROOT.mkdir(parents=True, exist_ok=True)

# Set SMOKE_TEST_MODE = True for a ~15-25 min end-to-end validation run first;
# switch to False for the full ~3,200-page public dataset run.
SMOKE_TEST_MODE = True

# Select which Apache-2.0 competition candidates to run:
# Available keys: ['gemma4_12b', 'gemma4_26b_a4b', 'inkling2_mini', 'qwen36_35b_a3b']
CANDIDATE_KEYS = ['gemma4_12b']
ARMS = ['baseline_zeroshot', 'real', 'sham_short']
SHAM_MAX_PRESENTATIONS = 128 if SMOKE_TEST_MODE else 504

# Public rights-cleared datasets (CC-BY-4.0 / Public Domain ONLY)
INCLUDE_REVCITY = True
REVCITY_MAX_PAGES = 400 if SMOKE_TEST_MODE else None
VAL_FRACTION = 0.075
TEST_FRACTION = 0.075

# Optional Public Domain LOC By the People (LCCN 2020446971) polite fetcher
INCLUDE_LOC_BY_THE_PEOPLE = False
LOC_MAX_PAGES = 150  # Polite cap when enabled

# Optional CC-BY-4.0 Bentham ICFHR-2016 HTR corpus (Zenodo 1164045; train-only to keep eval writers disjoint)
INCLUDE_BENTHAM = False
BENTHAM_MAX_PAGES = 200  # Cap when enabled

# Training hyperparameters (full-resolution trainable-vision SFT setup)
SEED = 20261125
EPOCHS = 1 if SMOKE_TEST_MODE else 2
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LEARNING_RATE = 1e-4
VISION_LR = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.05
GRAD_ACCUM_STEPS = 8
MAX_SEQ_LEN = 4096
MAX_EVAL_NEW_TOKENS = 1536
VAL_EVAL_MAX_PAGES = 24 if SMOKE_TEST_MODE else 48
TEST_EVAL_MAX_PAGES = 32 if SMOKE_TEST_MODE else 64  # Set None for entire held-out public test split

# Optional Hugging Face Hub Dataset export (strictly public/CC-BY-4.0 splits only)
PUBLISH_TO_HF_DATASET = False
HF_DATASET_REPO_ID = 'markandster/early-american-htr-public'

# Offline GRPO rollout generation (Cell 7)
GENERATE_OFFLINE_GRPO_ROLLOUTS = False
GRPO_ROLLOUT_PROMPTS = 64
GRPO_NUM_GENERATIONS = 4

for k in CANDIDATE_KEYS:
    validate_competition_candidate(COMPETITION_CANDIDATES[k])
print('SMOKE_TEST_MODE:', SMOKE_TEST_MODE, '| candidates:', CANDIDATE_KEYS, '| arms:', ARMS)
"""


def build_cell_1_markdown() -> str:
    """Return Cell 1 markdown documentation."""
    return CELL_1_MD_SOURCE


def build_cell_2_setup() -> str:
    """Return Cell 2 environment setup and authentication code."""
    return CELL_2_SETUP_SOURCE


def build_cell_3_config() -> str:
    """Return Cell 3 hyperparameter and corpus flags code."""
    return CELL_3_CONFIG_SOURCE
