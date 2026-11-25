"""Public-data-only Colab Pro training pipeline & notebook generator for Halo competition prep.

All datasets referenced here are strictly rights-cleared (CC-BY-4.0 or Public Domain),
and all candidate base models are Apache-2.0.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

DIPLOMATIC_PROMPT = (
    "Transcribe every word on this manuscript page using only the visible ink. "
    "Preserve the writer's original spelling, capitalization, punctuation, "
    "abbreviations, and line breaks. Do not modernize spelling or add commentary."
)

REVCITY_ZENODO_URL = "https://zenodo.org/api/records/15776323/files/revcity-htr.zip/content"
LOC_WASHINGTON_ITEM_JSON_URL = "https://www.loc.gov/item/2020446971/?fo=json"


@dataclass(frozen=True, slots=True)
class CompetitionCandidateSpec:
    """Configuration for a competition-eligible Apache-2.0 base VLM."""

    key: str
    hf_repo: str
    license: str
    params_b: float
    active_params_b: float
    lora_target_modules: tuple[str, ...]
    vision_trainable_patterns: tuple[str, ...]
    max_pixels: int
    mlx_quant_bits: int
    est_mlx_peak_gb: float
    min_bf16_vram_gb: float
    trust_remote_code: bool = False


COMPETITION_CANDIDATES: dict[str, CompetitionCandidateSpec] = {
    "gemma4_12b": CompetitionCandidateSpec(
        key="gemma4_12b",
        hf_repo="google/gemma-4-12B-it",
        license="Apache-2.0",
        params_b=12.0,
        active_params_b=12.0,
        lora_target_modules=("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
        vision_trainable_patterns=(
            "embed_vision",
            "multimodal_embedder",
            "patch_dense",
            "multi_modal_projector",
            "vision_projection",
        ),
        max_pixels=2048 * 2048,
        mlx_quant_bits=8,
        est_mlx_peak_gb=12.5,
        min_bf16_vram_gb=56.0,
        trust_remote_code=False,
    ),
    "gemma4_26b_a4b": CompetitionCandidateSpec(
        key="gemma4_26b_a4b",
        hf_repo="google/gemma-4-26B-A4B-it",
        license="Apache-2.0",
        params_b=26.0,
        active_params_b=4.0,
        lora_target_modules=("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
        vision_trainable_patterns=(
            "vision_tower",
            "multi_modal_projector",
            "embed_vision",
            "vision_projection",
            "mm_projector",
            "merger",
        ),
        max_pixels=2048 * 2048,
        mlx_quant_bits=4,
        est_mlx_peak_gb=15.5,
        min_bf16_vram_gb=72.0,
        trust_remote_code=False,
    ),
    "inkling2_mini": CompetitionCandidateSpec(
        key="inkling2_mini",
        hf_repo="thinkingmachines/Inkling-2-Mini",
        license="Apache-2.0",
        params_b=26.4,
        active_params_b=5.4,
        lora_target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
        vision_trainable_patterns=(
            "vision_projection",
            "multi_modal_projector",
            "patch_embed",
            "embed_vision",
            "visual.merger",
            "mm_projector",
            "merger",
        ),
        max_pixels=2048 * 2048,
        mlx_quant_bits=4,
        est_mlx_peak_gb=20.0,
        min_bf16_vram_gb=72.0,
        trust_remote_code=True,
    ),
    "qwen36_35b_a3b": CompetitionCandidateSpec(
        key="qwen36_35b_a3b",
        hf_repo="Qwen/Qwen3.6-35B-A3B",
        license="Apache-2.0",
        params_b=35.0,
        active_params_b=3.0,
        lora_target_modules=("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
        vision_trainable_patterns=("visual.merger", "visual.deepstack_merger_list", "merger"),
        max_pixels=2048 * 2048,
        mlx_quant_bits=4,
        est_mlx_peak_gb=21.5,
        min_bf16_vram_gb=78.0,
        trust_remote_code=False,
    ),
}


def validate_competition_candidate(spec: CompetitionCandidateSpec) -> None:
    """Enforce Apache-2.0 license, full-res input, trainable vision, and <= 24 GB MLX bound."""
    if spec.license != "Apache-2.0":
        raise ValueError(f"Competition candidate {spec.hf_repo} must be Apache-2.0, got {spec.license}")
    if spec.est_mlx_peak_gb > 24.0:
        raise ValueError(f"Estimated MLX peak RAM exceeds 24 GB bound: {spec.est_mlx_peak_gb}")
    if spec.max_pixels < 1280 * 1280:
        raise ValueError(f"Full-resolution training requires max_pixels >= 1280*1280, got {spec.max_pixels}")
    if not spec.vision_trainable_patterns:
        raise ValueError(f"Vision projector/embedder patterns must be non-empty for {spec.hf_repo}")


def should_use_4bit(spec: CompetitionCandidateSpec, vram_gb: float, arm: str) -> bool:
    """Decide whether to load a model in 4-bit NF4 based on GPU VRAM and arm.

    Zero-shot inference needs roughly ``params_b * 2.5`` GB in BF16 (weights + KV/vision buffers),
    whereas LoRA fine-tuning requires ``spec.min_bf16_vram_gb`` to stay in full BF16 cleanly.
    """
    if arm == "baseline_zeroshot":
        return vram_gb < min(spec.min_bf16_vram_gb, spec.params_b * 2.5)
    return vram_gb < spec.min_bf16_vram_gb


def resolve_vision_modules_to_save(
    module_names: Iterable[str],
    patterns: Sequence[str],
) -> list[str]:
    """Select shallowest vision projector/embedder ancestors; fail closed if 0 match."""
    all_names = [n for n in module_names if n]
    matched: list[str] = []
    for name in all_names:
        if any(name == pat or name.endswith("." + pat) for pat in patterns):
            matched.append(name)

    deduped: list[str] = []
    for name in sorted(set(matched), key=lambda s: (s.count("."), s)):
        if not any(name.startswith(parent + ".") for parent in deduped):
            deduped.append(name)

    if not deduped:
        sample_top = sorted({n.split(".")[0] for n in all_names})[:15]
        raise RuntimeError(
            f"No trainable vision modules matched patterns {list(patterns)}. "
            f"Sample top-level modules: {sample_top}. Refusing frozen-vision run."
        )
    return deduped


def make_sham_derangement(records: Sequence[dict], seed: int = 20261125) -> list[dict]:
    """Deterministically mismatch images to transcripts with zero self-matches."""
    n = len(records)
    if n < 2:
        raise ValueError("Sham derangement requires at least 2 records")
    rng = random.Random(seed)
    indices = list(range(n))
    for _ in range(128):
        rng.shuffle(indices)
        if all(indices[i] != i for i in range(n)):
            break
    else:
        indices = [(i + 1) % n for i in range(n)]

    sham: list[dict] = []
    for i, rec in enumerate(records):
        donor = records[indices[i]]
        sham.append(
            {
                **rec,
                "text": donor["text"],
                "sham_donor_record_id": donor.get("record_id", str(indices[i])),
                "arm": "sham",
            }
        )
    return sham


def assign_deterministic_split(
    group_key: str,
    *,
    val_fraction: float = 0.075,
    test_fraction: float = 0.075,
    salt: str = "eahtr-public-split-v1",
) -> str:
    """Assign 'test_public', 'val', or 'train' deterministically by SHA-256 of group_key."""
    if val_fraction < 0 or test_fraction < 0 or val_fraction + test_fraction >= 1.0:
        raise ValueError("val_fraction + test_fraction must lie in [0, 1)")
    digest = hashlib.sha256(f"{salt}:{group_key}".encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    if bucket < test_fraction:
        return "test_public"
    if bucket < test_fraction + val_fraction:
        return "val"
    return "train"


def _code_cell(code: str) -> dict:
    lines = [line + "\n" for line in code.strip("\n").splitlines()]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines,
    }


def _md_cell(md: str) -> dict:
    lines = [line + "\n" for line in md.strip("\n").splitlines()]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": lines,
    }


def write_competition_colab_notebook(out_path: Path) -> Path:
    """Generate a self-contained Colab Pro notebook strictly for public competition datasets."""
    for spec in COMPETITION_CANDIDATES.values():
        validate_competition_candidate(spec)

    cell_1_md = """
# Early American HTR — Public Competition Training & Eval (Colab Pro)

**Competition-cleared public data only (`CC-BY-4.0` and `Public Domain`).**

This notebook mirrors our high-resolution, trainable-vision, sham-controlled training harness while strictly restricting datasets and models to competition-eligible open resources:

1. **Data (100% Public & Rights-Cleared):**
   - **Revolutionary City HTR Corpus** (Zenodo `15776323`, **CC-BY-4.0**) downloaded directly to Colab SSD, parsed from ALTO-XML, cleaned via `eahtr.hygiene.clean_diplomatic_target` (stripping `[...]` editorial brackets and malformed standalone `-` lines), filtered for duplicate-line slips, and split deterministically by series (`train` / `val` / `test_public`) with zero document, writer, image SHA-256, or 32-word n-gram leakage.
   - **Optional Library of Congress *By the People*** (`LCCN 2020446971`, **Public Domain**) behind `INCLUDE_LOC_BY_THE_PEOPLE = False` (skipping tabular leader-dot pages).
2. **Candidate Apache-2.0 Models:**
   - `gemma4_12b` (`google/gemma-4-12B-it`, default) — encoder-free dense VLM
   - `gemma4_26b_a4b` (`google/gemma-4-26B-A4B-it`) — 26B/4B-active MoE VLM
   - `inkling2_mini` (`thinkingmachines/Inkling-2-Mini`) — 26.4B looped MoE VLM
   - `qwen36_35b_a3b` (`Qwen/Qwen3.6-35B-A3B`) — 35B/3B-active MoE VLM
3. **Evaluation & Stage 2 Offline GRPO Prep:**
   - Evaluates strict/lenient WER & CER (with independent `rapidfuzz` recount), repetition loops, non-termination, editorial bracket leakage, modernization counts, and mean verifiable `eahtr.rewards` score, plus a matched-compute `sham_short` control.
   - Optional Cell 7 generates and scores $K=4$ rollouts for Halo `grpo/offline` (`configs/halo/archival_grpo_offline.yaml`).

> **Halo Competition Rule Note (Terms §§ 4.2 & 5):** Pre-Dec-1 Colab runs establish baseline benchmarks, data hygiene, vision-gradient flow, and offline-GRPO rollout data on the competition corpus. Final competition submissions after Dec 1 must be reproduced by the official **Halo** training recipe (`github.com/whitecircle/halo`) under published Track Rules.
"""

    cell_2_setup = """
!pip install -q --upgrade transformers peft accelerate bitsandbytes rapidfuzz pyyaml pillow tqdm

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
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from google.colab import drive
drive.mount('/content/drive', force_remount=False)

REPO_URL = 'https://github.com/markandster/early-american-htr-halo.git'
REPO_DIR = Path('/content/early-american-htr-halo')
if not REPO_DIR.exists():
    subprocess.run(['git', 'clone', '--depth', '1', REPO_URL, str(REPO_DIR)], check=True)
else:
    subprocess.run(['git', '-C', str(REPO_DIR), 'pull', '--ff-only'], check=True)
subprocess.run(['pip', 'install', '-q', '--no-deps', '-e', str(REPO_DIR)], check=True)

from eahtr.colab_pipeline import (
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
from eahtr.corpus import filter_corpus_record, parse_alto_xml_lines
from eahtr.hf_dataset import push_splits_to_hf_hub
from eahtr.hygiene import audit_split_disjointness, clean_diplomatic_target
from eahtr.metrics import normalize_strict, score_pair
from eahtr.rewards import score_group_completions, score_transcription

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
        'NOTE: HF_TOKEN not set in env or Colab Secrets. '
        'Gemma 4 checkpoints (e.g. https://huggingface.co/google/gemma-4-12B-it) are gated and require accepting terms + HF_TOKEN; '
        'thinkingmachines/Inkling-2-Mini is ungated.'
    )
"""

    cell_3_config = """
# === Run Configuration (100% Public Competition Data Only) ===
OUTPUT_ROOT = Path('/content/drive/MyDrive/early-american-htr-halo-runs')
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

# Training hyperparameters (mirrors proven Phase 4B full-res setup)
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
MAX_SEQ_LEN = 2048
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

    cell_4_corpus = """
# === Step 1: Download, Clean & Split Rights-Cleared Public Datasets ===
def download_with_retry(url: str, dest: Path, max_attempts: int = 4, min_existing_bytes: int = 1_000_000_000) -> None:
    if dest.exists() and dest.stat().st_size >= min_existing_bytes:
        print(f'Reusing cached file: {dest} ({dest.stat().st_size / 1e6:.1f} MB)')
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + '.part')
    last_err = None
    for attempt in range(1, max_attempts + 1):
        try:
            existing = part.stat().st_size if part.exists() else 0
            headers = {'User-Agent': 'early-american-htr-halo/0.1 (public-corpus-research)'}
            if existing > 0:
                headers['Range'] = f'bytes={existing}-'
            req = Request(url, headers=headers)
            with urlopen(req, timeout=120) as resp:
                status = getattr(resp, 'status', 200)
                mode = 'ab' if (existing > 0 and status == 206) else 'wb'
                total = int(resp.headers.get('Content-Length') or 0) + (existing if mode == 'ab' else 0)
                with part.open(mode) as fh, tqdm(
                    total=total or None,
                    initial=existing if mode == 'ab' else 0,
                    unit='B',
                    unit_scale=True,
                    desc=dest.name,
                ) as pbar:
                    while True:
                        chunk = resp.read(4 << 20)
                        if not chunk:
                            break
                        fh.write(chunk)
                        pbar.update(len(chunk))
            part.replace(dest)
            return
        except (HTTPError, URLError, OSError, TimeoutError) as exc:
            last_err = exc
            wait_s = min(60, 5 * attempt)
            print(f'[Attempt {attempt}/{max_attempts}] Download error ({type(exc).__name__}: {exc}); retrying in {wait_s}s...')
            time.sleep(wait_s)
    raise RuntimeError(f'Failed to download {url}: {last_err}')


def fetch_bytes_polite(url: str, max_attempts: int = 3, delay_s: float = 0.5) -> bytes:
    last_err = None
    for attempt in range(1, max_attempts + 1):
        try:
            time.sleep(delay_s)
            req = Request(url, headers={'User-Agent': 'early-american-htr-halo/0.1 (public-corpus-research)'})
            with urlopen(req, timeout=60) as resp:
                return resp.read()
        except (HTTPError, URLError, OSError, TimeoutError) as exc:
            last_err = exc
            time.sleep(min(30, 3 * attempt))
    raise RuntimeError(f'Failed GET {url}: {last_err}')


all_records_by_split = {'train': [], 'val': [], 'test_public': []}
corpus_report = Counter()

if INCLUDE_REVCITY:
    rc_dir = WORK_ROOT / 'revcity'
    rc_zip = rc_dir / 'revcity-htr.zip'
    rc_pages = rc_dir / 'pages'
    rc_pages.mkdir(parents=True, exist_ok=True)
    download_with_retry(REVCITY_ZENODO_URL, rc_zip, min_existing_bytes=1_000_000_000)

    with zipfile.ZipFile(rc_zip) as zf:
        names = set(zf.namelist())
        stems = sorted(n[:-4] for n in names if n.endswith('.xml') and (n[:-4] + '.jpg') in names)
        if REVCITY_MAX_PAGES is not None:
            stems = stems[:REVCITY_MAX_PAGES]
        print(f'Scanning {len(stems)} paired (JPG, ALTO-XML) pages in Zenodo 15776323 (CC-BY-4.0)')

        for stem in tqdm(stems, desc='RevCity ALTO + Hygiene'):
            safe_id = Path(stem).name.replace('%3A', '_')
            try:
                lines = parse_alto_xml_lines(zf.read(stem + '.xml').decode('utf-8', errors='replace'))
            except Exception:
                corpus_report['revcity_xml_parse_error'] += 1
                continue
            raw_text = '\\n'.join(lines)
            rec_probe = {'source_name': 'Revolutionary City', 'license': 'CC-BY-4.0', 'text': raw_text}
            accepted, reason = filter_corpus_record(rec_probe)
            if not accepted:
                corpus_report[f'revcity_{reason}'] += 1
                continue

            cleaned_text = clean_diplomatic_target(raw_text)
            if len(cleaned_text.split()) < 3:
                corpus_report['revcity_dropped_too_short'] += 1
                continue

            jpg_bytes = zf.read(stem + '.jpg')
            img_sha = hashlib.sha256(jpg_bytes).hexdigest()
            img_out = rc_pages / f'{safe_id}.jpg'
            if not img_out.exists():
                img_out.write_bytes(jpg_bytes)

            # Group multi-page letters together by stripping trailing -NNN
            group_key = re.sub(r'[-_]\\d+$', '', safe_id)
            split = assign_deterministic_split(
                f'revcity:{group_key}', val_fraction=VAL_FRACTION, test_fraction=TEST_FRACTION
            )
            all_records_by_split[split].append({
                'record_id': f'revcity-{safe_id}',
                'collection': 'revcity-zenodo-15776323',
                'document_id': f'revcity:{group_key}',
                'writer_key': f'revcity_series:{group_key}',
                'image_path': str(img_out),
                'image_sha256': img_sha,
                'text': cleaned_text,
                'word_count': len(cleaned_text.split()),
                'license': 'CC-BY-4.0',
                'split': split,
            })
            corpus_report[f'revcity_admitted_{split}'] += 1

if INCLUDE_LOC_BY_THE_PEOPLE:
    loc_dir = WORK_ROOT / 'loc_btp'
    loc_pages = loc_dir / 'pages'
    loc_pages.mkdir(parents=True, exist_ok=True)
    item_meta = json.loads(fetch_bytes_polite(LOC_WASHINGTON_ITEM_JSON_URL, delay_s=0.2).decode('utf-8'))
    zip_urls = []
    for res in item_meta.get('resources', []):
        for f_entry in res.get('files', []):
            for sub in (f_entry if isinstance(f_entry, list) else [f_entry]):
                u = sub.get('url', '') if isinstance(sub, dict) else ''
                if u.endswith('.zip'):
                    zip_urls.append(u)
    assert zip_urls, 'Could not find Public Domain dataset .zip URL in LOC item metadata'
    loc_zip = loc_dir / '2020446971.zip'
    download_with_retry(zip_urls[0], loc_zip, min_existing_bytes=50_000)

    loc_added = 0
    with zipfile.ZipFile(loc_zip) as zf:
        csv_names = [n for n in zf.namelist() if n.lower().endswith('.csv')]
        assert csv_names, 'No CSV found inside LOC By the People zip'
        csv_text = zf.read(csv_names[0]).decode('utf-8', errors='replace')
        reader = csv.DictReader(io.StringIO(csv_text))
        for row_idx, row in enumerate(reader):
            if loc_added >= LOC_MAX_PAGES:
                break
            item_title = (row.get('Item') or row.get('Title') or '').lower()
            if 'farm report' in item_title:
                corpus_report['loc_skipped_farm_report'] += 1
                continue
            raw_text = row.get('Transcription') or row.get('Text') or ''
            img_url = row.get('DownloadUrl') or row.get('Asset') or ''
            if not raw_text.strip() or not img_url.startswith('http'):
                corpus_report['loc_skipped_missing_fields'] += 1
                continue
            accepted, reason = filter_corpus_record({
                'source_name': 'LOC By the People',
                'license': 'Public Domain',
                'text': raw_text,
            })
            if not accepted:
                corpus_report[f'loc_{reason}'] += 1
                continue
            cleaned_text = clean_diplomatic_target(raw_text)
            if len(cleaned_text.split()) < 5:
                corpus_report['loc_dropped_too_short'] += 1
                continue
            try:
                jpg_bytes = fetch_bytes_polite(img_url, delay_s=0.4)
            except Exception:
                corpus_report['loc_image_fetch_error'] += 1
                continue
            img_sha = hashlib.sha256(jpg_bytes).hexdigest()
            rec_id = f'loc-btp-{row_idx:05d}'
            img_out = loc_pages / f'{rec_id}.jpg'
            img_out.write_bytes(jpg_bytes)
            doc_group = str(row.get('ItemId') or row.get('Item') or rec_id)
            split = assign_deterministic_split(
                f'loc:{doc_group}', val_fraction=VAL_FRACTION, test_fraction=TEST_FRACTION
            )
            all_records_by_split[split].append({
                'record_id': rec_id,
                'collection': 'loc-by-the-people-2020446971',
                'document_id': f'loc:{doc_group}',
                'writer_key': f'loc_item:{doc_group}',
                'image_path': str(img_out),
                'image_sha256': img_sha,
                'text': cleaned_text,
                'word_count': len(cleaned_text.split()),
                'license': 'Public Domain',
                'split': split,
            })
            corpus_report[f'loc_admitted_{split}'] += 1
            loc_added += 1

train_rows = all_records_by_split['train']
val_rows = all_records_by_split['val']
test_public_rows = all_records_by_split['test_public']

val_audit = audit_split_disjointness(train_rows, val_rows, ngram_size=32)
test_audit = audit_split_disjointness(train_rows, test_public_rows, ngram_size=32)
assert val_audit['passed'], f'Train/Val leakage detected: {val_audit}'
assert test_audit['passed'], f'Train/Test leakage detected: {test_audit}'

manifest_dir = OUTPUT_ROOT / 'manifests'
manifest_dir.mkdir(parents=True, exist_ok=True)
for split_name, rows in all_records_by_split.items():
    (manifest_dir / f'{split_name}.jsonl').write_text(
        '\\n'.join(json.dumps(r, ensure_ascii=False) for r in rows) + '\\n',
        encoding='utf-8',
    )

audit_summary = {
    'counts': dict(corpus_report),
    'train_sides': len(train_rows),
    'train_words': sum(r['word_count'] for r in train_rows),
    'val_sides': len(val_rows),
    'val_words': sum(r['word_count'] for r in val_rows),
    'test_public_sides': len(test_public_rows),
    'test_public_words': sum(r['word_count'] for r in test_public_rows),
    'val_disjoint_audit': val_audit,
    'test_disjoint_audit': test_audit,
}
(OUTPUT_ROOT / 'corpus_audit_report.json').write_text(json.dumps(audit_summary, indent=2), encoding='utf-8')
print(json.dumps({k: v for k, v in audit_summary.items() if 'audit' not in k}, indent=2))
print('Disjointness checks PASSED: train vs val and train vs test_public.')

if PUBLISH_TO_HF_DATASET:
    print(f'Publishing validated public splits to Hugging Face Hub: {HF_DATASET_REPO_ID} ...')
    !pip install -q datasets huggingface_hub
    ds_url = push_splits_to_hf_hub(all_records_by_split, HF_DATASET_REPO_ID, audit_summary, private=False)
    print(f'Published public dataset: {ds_url}')
"""

    cell_5_train_helpers = """
# === Step 2: Training, Vision-Grad Verification & Verifiable Scoring Loop ===
def load_processor_fullres(spec_dict: dict):
    hf_repo = spec_dict['hf_repo']
    max_pixels = spec_dict['max_pixels']
    trust = bool(spec_dict.get('trust_remote_code', False))
    try:
        return AutoProcessor.from_pretrained(hf_repo, max_pixels=max_pixels, trust_remote_code=trust)
    except TypeError:
        proc = AutoProcessor.from_pretrained(hf_repo, trust_remote_code=trust)
        ip = getattr(proc, 'image_processor', None)
        if ip is not None and hasattr(ip, 'max_pixels'):
            ip.max_pixels = max_pixels
        return proc


def load_vlm_model(hf_repo: str, load_kwargs: dict):
    # Load VLM via AutoModelForMultimodalLM or AutoModelForImageTextToText cleanly.
    prefer_multimodal = 'inkling' in hf_repo.lower() and AutoModelForMultimodalLM is not None
    loaders = (
        [AutoModelForMultimodalLM, AutoModelForImageTextToText]
        if prefer_multimodal
        else [cls for cls in (AutoModelForImageTextToText, AutoModelForMultimodalLM) if cls is not None]
    )
    last_err = None
    for cls in loaders:
        try:
            return cls.from_pretrained(hf_repo, **load_kwargs)
        except (ValueError, KeyError, TypeError) as exc:
            last_err = exc
    raise last_err


def build_training_batch(processor, image: Image.Image, target_text: str, device):
    user_msgs = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': DIPLOMATIC_PROMPT}]}]
    full_msgs = user_msgs + [{'role': 'assistant', 'content': [{'type': 'text', 'text': target_text}]}]
    prompt_str = processor.apply_chat_template(user_msgs, tokenize=False, add_generation_prompt=True)
    full_str = processor.apply_chat_template(full_msgs, tokenize=False, add_generation_prompt=False)

    prompt_inputs = processor(text=[prompt_str], images=[image], return_tensors='pt')
    full_inputs = processor(text=[full_str], images=[image], return_tensors='pt')
    prompt_len = prompt_inputs['input_ids'].shape[1]

    if full_inputs['input_ids'].shape[1] > MAX_SEQ_LEN:
        full_inputs['input_ids'] = full_inputs['input_ids'][:, :MAX_SEQ_LEN]
        full_inputs['attention_mask'] = full_inputs['attention_mask'][:, :MAX_SEQ_LEN]

    labels = full_inputs['input_ids'].clone()
    labels[:, :min(prompt_len, labels.shape[1])] = -100
    pad_id = getattr(processor.tokenizer, 'pad_token_id', None)
    if pad_id is not None:
        labels[labels == pad_id] = -100
    full_inputs['labels'] = labels
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in full_inputs.items()}


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


@torch.no_grad()
def run_eval_panel(model, processor, rows: list[dict], panel_name: str, max_new_tokens: int = MAX_EVAL_NEW_TOKENS) -> dict:
    model.eval()
    preds, refs, finish_reasons, per_side = [], [], [], []
    for row in tqdm(rows, desc=f'Eval {panel_name}', leave=False):
        image = Image.open(row['image_path']).convert('RGB')
        messages = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': DIPLOMATIC_PROMPT}]}]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[prompt], images=[image], return_tensors='pt').to(model.device)
        t0 = time.time()
        out_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        elapsed = round(time.time() - t0, 2)
        gen_ids = out_ids[:, inputs['input_ids'].shape[1]:]
        fr = 'length' if int(gen_ids.shape[1]) >= int(max_new_tokens) else 'stop'
        gen = processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
        preds.append(gen)
        refs.append(row['text'])
        finish_reasons.append(fr)
        per_side.append({
            'record_id': row['record_id'],
            'document_id': row['document_id'],
            'elapsed_s': elapsed,
            'finish_reason': fr,
            'prediction': gen,
        })
    scores = score_predictions_verifiable(refs, preds, finish_reasons)
    return {'panel': panel_name, 'sides': len(rows), 'scores': scores, 'predictions': per_side}


def run_candidate_arm(cand_key: str, arm: str, arm_train_rows: list[dict], val_slice: list[dict], test_slice: list[dict]) -> dict:
    spec = COMPETITION_CANDIDATES[cand_key]
    out_dir = OUTPUT_ROOT / f'{cand_key}-{arm}'
    out_dir.mkdir(parents=True, exist_ok=True)
    score_file = out_dir / 'public_eval_scores.json'
    if score_file.is_file():
        print(f'[SKIP] Already completed {cand_key}-{arm}: {score_file}')
        return json.loads(score_file.read_text(encoding='utf-8'))

    print(f'\\n================ {cand_key} ({spec.hf_repo}) | ARM={arm} ================')
    spec_dict = {
        'hf_repo': spec.hf_repo,
        'max_pixels': spec.max_pixels,
        'trust_remote_code': spec.trust_remote_code,
    }
    processor = load_processor_fullres(spec_dict)

    use_4bit = should_use_4bit(spec, VRAM_GB, arm)
    load_kwargs = {
        'device_map': 'auto',
        'torch_dtype': torch.bfloat16,
        'trust_remote_code': spec.trust_remote_code,
    }
    if use_4bit:
        mode_label = '4-bit NF4 inference' if arm == 'baseline_zeroshot' else '4-bit NF4 QLoRA'
        print(f'VRAM {VRAM_GB} GB: loading {cand_key} ({arm}) with {mode_label} (BF16 compute)')
        load_kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    model = load_vlm_model(spec.hf_repo, load_kwargs)

    if arm == 'baseline_zeroshot':
        eval_res = run_eval_panel(model, processor, test_slice, panel_name=f'{cand_key}-zeroshot-public-test')
        payload = {
            'candidate': cand_key,
            'hf_repo': spec.hf_repo,
            'arm': arm,
            'used_4bit': use_4bit,
            'scores': eval_res['scores'],
        }
        (out_dir / 'predictions_public_test.json').write_text(
            json.dumps(eval_res['predictions'], ensure_ascii=False, indent=2), encoding='utf-8'
        )
        score_file.write_text(json.dumps(payload, indent=2), encoding='utf-8')
        print(f'[ZERO-SHOT {cand_key}]', payload['scores'])
        del model; gc.collect(); torch.cuda.empty_cache()
        return payload

    if use_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model.gradient_checkpointing_enable()
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
    lora_cfg = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=list(spec.lora_target_modules),
        modules_to_save=vision_modules,
        bias='none',
        task_type='CAUSAL_LM',
    )
    model = get_peft_model(model, lora_cfg)

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

    for epoch in range(1, EPOCHS + 1):
        model.train()
        order = list(arm_train_rows)
        rng.shuffle(order)
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        n_batches = 0

        for step_idx, row in enumerate(tqdm(order, desc=f'{cand_key}-{arm} ep{epoch}/{EPOCHS}'), start=1):
            try:
                image = Image.open(row['image_path']).convert('RGB')
                batch = build_training_batch(processor, image, row['text'], model.device)
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    out = model(**batch)
                    loss = out.loss / GRAD_ACCUM_STEPS
                loss.backward()
                running_loss += float(out.loss.detach().cpu())
                n_batches += 1
            except torch.cuda.OutOfMemoryError:
                skipped_oom += 1
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                continue

            if (step_idx % GRAD_ACCUM_STEPS == 0) or (step_idx == len(order)):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        avg_loss = round(running_loss / max(1, n_batches), 4)
        val_eval = run_eval_panel(model, processor, val_slice, panel_name=f'{cand_key}-{arm}-val-ep{epoch}')
        val_cer = val_eval['scores']['strict_cer_pct']
        train_log.append({'epoch': epoch, 'train_loss': avg_loss, 'skipped_oom': skipped_oom, 'val_scores': val_eval['scores']})
        print(f'Epoch {epoch} | loss={avg_loss} | val={val_eval["scores"]}')

        if val_cer <= best_val_cer:
            best_val_cer = val_cer
            best_epoch = epoch
            best_trainable_state = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}
            adapter_dir = out_dir / 'adapter_best'
            model.save_pretrained(adapter_dir)
            processor.save_pretrained(adapter_dir)

    if best_trainable_state is not None:
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in best_trainable_state:
                    p.copy_(best_trainable_state[n].to(device=p.device, dtype=p.dtype))
        print(f'Restored best-val checkpoint from epoch {best_epoch} (val strict CER={best_val_cer}%)')

    test_eval = run_eval_panel(model, processor, test_slice, panel_name=f'{cand_key}-{arm}-public-test')
    pred_bytes = json.dumps(test_eval['predictions'], ensure_ascii=False, indent=2).encode('utf-8')
    (out_dir / 'predictions_public_test.json').write_bytes(pred_bytes)

    merged_dir = out_dir / 'merged_hf'
    if not use_4bit:
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(merged_dir, safe_serialization=True)
        processor.save_pretrained(merged_dir)
        del merged_model

    payload = {
        'candidate': cand_key,
        'hf_repo': spec.hf_repo,
        'arm': arm,
        'train_sides': len(arm_train_rows),
        'vision_trainable_params': n_vision_trainable,
        'lora_trainable_params': n_lora_trainable,
        'used_4bit_qlora': use_4bit,
        'best_epoch': best_epoch,
        'train_log': train_log,
        'predictions_sha256': hashlib.sha256(pred_bytes).hexdigest(),
        'scores': test_eval['scores'],
        'adapter_dir': str(out_dir / 'adapter_best'),
        'merged_hf_dir': str(merged_dir) if not use_4bit else None,
    }
    score_file.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(f'[DONE {cand_key}-{arm}] public test:', payload['scores'])

    del model, optimizer, scheduler
    gc.collect()
    torch.cuda.empty_cache()
    return payload
"""

    cell_6_run = """
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

    cell_7_grpo = """
# === Step 4 (Optional): Generate & Score Stage 2 Rollouts for Halo Offline GRPO ===
if GENERATE_OFFLINE_GRPO_ROLLOUTS:
    from peft import PeftModel
    cand_key = CANDIDATE_KEYS[0]
    spec = COMPETITION_CANDIDATES[cand_key]
    adapter_dir = OUTPUT_ROOT / f'{cand_key}-real' / 'adapter_best'
    assert adapter_dir.is_dir(), f'Train real adapter first: {adapter_dir}'

    processor = load_processor_fullres({'hf_repo': spec.hf_repo, 'max_pixels': spec.max_pixels, 'trust_remote_code': spec.trust_remote_code})
    base_model = load_vlm_model(spec.hf_repo, {'device_map': 'auto', 'torch_dtype': torch.bfloat16, 'trust_remote_code': spec.trust_remote_code})
    model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    model.eval()

    rollout_rows = train_rows[:GRPO_ROLLOUT_PROMPTS]
    out_jsonl = OUTPUT_ROOT / f'{cand_key}_scored_offline_grpo_rollouts.jsonl'
    with out_jsonl.open('w', encoding='utf-8') as fout:
        for row in tqdm(rollout_rows, desc='Offline GRPO Rollouts'):
            image = Image.open(row['image_path']).convert('RGB')
            messages = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': DIPLOMATIC_PROMPT}]}]
            prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[prompt], images=[image], return_tensors='pt').to(model.device)
            completions, finish_reasons = [], []
            for _ in range(GRPO_NUM_GENERATIONS):
                with torch.no_grad():
                    out_ids = model.generate(**inputs, max_new_tokens=MAX_EVAL_NEW_TOKENS, do_sample=True, temperature=0.7, top_p=0.9)
                gen_ids = out_ids[:, inputs['input_ids'].shape[1]:]
                fr = 'length' if int(gen_ids.shape[1]) >= int(MAX_EVAL_NEW_TOKENS) else 'stop'
                completions.append(processor.batch_decode(gen_ids, skip_special_tokens=True)[0])
                finish_reasons.append(fr)
            scored = score_group_completions(completions, row['text'], finish_reasons)
            for item in scored:
                fout.write(json.dumps({
                    'record_id': row['record_id'],
                    'image_path': row['image_path'],
                    'prompt': DIPLOMATIC_PROMPT,
                    'completion': item['completion'],
                    'reward': item['reward'],
                    'advantage': item['advantage'],
                }, ensure_ascii=False) + '\\n')
    print('Wrote Halo Offline GRPO dataset ->', out_jsonl)
"""

    cell_8_mlx = """
# === Step 5: Apple Silicon (M3 Max <= 24 GB) MLX Conversion Commands ===
for cand_key in CANDIDATE_KEYS:
    spec = COMPETITION_CANDIDATES[cand_key]
    merged = ALL_RESULTS.get(cand_key, {}).get('real', {}).get('merged_hf_dir') or f'<merged_{cand_key}_hf_dir>'
    print(f'# {cand_key} ({spec.hf_repo}, est MLX peak ~{spec.est_mlx_peak_gb} GB):')
    print(f'python -m mlx_vlm.convert --hf-path {merged} --mlx-path ./mlx/{cand_key}-q{spec.mlx_quant_bits} -q --q-bits {spec.mlx_quant_bits}')
"""

    cells = [
        _md_cell(cell_1_md),
        _code_cell(cell_2_setup),
        _code_cell(cell_3_config),
        _code_cell(cell_4_corpus),
        _code_cell(cell_5_train_helpers),
        _code_cell(cell_6_run),
        _code_cell(cell_7_grpo),
        _code_cell(cell_8_mlx),
    ]

    nb = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"gpuType": "A100", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".part")
    tmp.write_text(json.dumps(nb, indent=2) + "\n", encoding="utf-8")
    tmp.replace(out_path)
    return out_path


if __name__ == "__main__":
    default_out = Path(__file__).resolve().parents[2] / "notebooks" / "colab_competition_pretrain_sft.ipynb"
    written = write_competition_colab_notebook(default_out)
    print(f"Wrote Colab notebook -> {written}")
