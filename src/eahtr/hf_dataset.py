"""Opt-in Hugging Face Hub publisher for strictly rights-cleared public splits.

Every split passed to this module is gated on:
1. Allowed public-domain / CC-BY-4.0 / CC0-1.0 source licenses via
   :func:`eahtr.corpus.verify_source_license`.
2. Zero document, writer, image-SHA256, or 32-gram leakage between ``train`` and
   every evaluation split (``val``, ``test_public``, etc.) via
   :func:`eahtr.hygiene.audit_split_disjointness`.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import json
from typing import Any

from .corpus import verify_source_license
from .hygiene import audit_split_disjointness


def validate_records_for_public_publish(
    records_by_split: Mapping[str, list[dict[str, Any]]],
    *,
    ngram_size: int = 32,
) -> dict[str, Any]:
    """Verify public-license compliance and strict train/eval split disjointness."""
    train_rows = records_by_split.get("train", [])
    if not train_rows:
        raise ValueError("Cannot publish dataset: 'train' split is empty.")

    observed_licenses: set[str] = set()
    split_counts: dict[str, int] = {}

    for split_name, rows in records_by_split.items():
        split_counts[split_name] = len(rows)
        for rec in rows:
            source_name = str(rec.get("source_name") or rec.get("collection") or "unknown")
            license_id = str(rec.get("license") or "")
            verify_source_license(source_name, license_id)
            observed_licenses.add(license_id)

    audits: dict[str, dict[str, Any]] = {}
    for split_name, rows in records_by_split.items():
        if split_name == "train" or not rows:
            continue
        audit = audit_split_disjointness(train_rows, rows, ngram_size=ngram_size)
        audits[f"train_vs_{split_name}"] = audit
        if not audit["passed"]:
            raise ValueError(
                f"Cannot publish dataset: leakage detected between 'train' and '{split_name}': {audit}"
            )

    return {
        "passed": True,
        "splits": split_counts,
        "licenses": sorted(observed_licenses),
        "audits": audits,
    }


def _card_license_slug(licenses: Iterable[str]) -> str:
    lic_set = {lic.strip() for lic in licenses if lic and lic.strip()}
    if lic_set == {"CC0-1.0"}:
        return "cc0-1.0"
    if lic_set == {"Public Domain"}:
        return "cc0-1.0"
    return "cc-by-4.0"


def build_hf_dataset_card(
    repo_id: str,
    audit_summary: Mapping[str, Any],
    licenses: Iterable[str],
) -> str:
    """Build a Hugging Face dataset card with valid YAML frontmatter and attribution."""
    lic_list = sorted({lic.strip() for lic in licenses if lic and lic.strip()}) or ["CC-BY-4.0"]
    card_lic = _card_license_slug(lic_list)
    frontmatter_licenses = json.dumps(card_lic)
    summary_json = json.dumps(dict(audit_summary), indent=2, default=str)

    return f"""---
license: {frontmatter_licenses}
language:
  - en
task_categories:
  - image-to-text
tags:
  - ocr
  - htr
  - historical-manuscripts
  - paleography
  - early-american
pretty_name: Early American Handwritten Text Recognition (Public Split)
---

# {repo_id}

Cleaned, leakage-audited historical manuscript transcription splits built exclusively
from rights-cleared public sources for the WhiteCircle Halo Fine-Tuning Competition.

## Source Attribution & Licenses

Observed source licenses in this build: `{", ".join(lic_list)}`.

- **Revolutionary City Corpus (1758-1805)**: Zenodo record `15776323` (`CC-BY-4.0`),
  cleaned diplomatic line transcriptions extracted from paired ALTO-XML and page images.
- **Library of Congress *By the People*** (when included): Public Domain (`LCCN 2020446971`).

All non-commercial (`CC-BY-NC*`), restricted, or unverified third-party archives are
excluded by construction via `eahtr.corpus.verify_source_license`.

## Split Hygiene

Splits are deterministic at document-group granularity (`assign_deterministic_split`) and
verified via `audit_split_disjointness` for zero document ID, writer key, image SHA-256,
and 32-gram word overlap between `train` and every held-out evaluation split.

```json
{summary_json}
```
"""


def push_splits_to_hf_hub(
    records_by_split: Mapping[str, list[dict[str, Any]]],
    repo_id: str,
    audit_summary: Mapping[str, Any],
    *,
    private: bool = False,
) -> str:
    """Validate public splits and push a DatasetDict + README card to Hugging Face Hub."""
    validation = validate_records_for_public_publish(records_by_split)

    try:
        from datasets import Dataset, DatasetDict, Features, Image as HFImage, Value
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError(
            "Publishing requires optional 'datasets' and 'huggingface_hub' packages: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    features = Features(
        {
            "record_id": Value("string"),
            "collection": Value("string"),
            "document_id": Value("string"),
            "writer_key": Value("string"),
            "image": HFImage(),
            "image_sha256": Value("string"),
            "text": Value("string"),
            "word_count": Value("int32"),
            "license": Value("string"),
            "split": Value("string"),
        }
    )

    hf_splits: dict[str, Any] = {}
    for split_name, rows in records_by_split.items():
        if not rows:
            continue
        columns = {
            "record_id": [str(r["record_id"]) for r in rows],
            "collection": [str(r["collection"]) for r in rows],
            "document_id": [str(r["document_id"]) for r in rows],
            "writer_key": [str(r["writer_key"]) for r in rows],
            "image": [str(r["image_path"]) for r in rows],
            "image_sha256": [str(r["image_sha256"]) for r in rows],
            "text": [str(r["text"]) for r in rows],
            "word_count": [int(r["word_count"]) for r in rows],
            "license": [str(r["license"]) for r in rows],
            "split": [str(r.get("split", split_name)) for r in rows],
        }
        hf_splits[split_name] = Dataset.from_dict(columns, features=features)

    dataset_dict = DatasetDict(hf_splits)
    merged_audit = dict(audit_summary)
    merged_audit["publish_validation"] = {
        "splits": validation["splits"],
        "licenses": validation["licenses"],
    }
    card_md = build_hf_dataset_card(repo_id, merged_audit, validation["licenses"])

    try:
        api = HfApi()
        api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
        dataset_dict.push_to_hub(repo_id, private=private)
        api.upload_file(
            path_or_fileobj=card_md.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Upload validated public HTR splits and dataset card",
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed pushing public dataset to Hugging Face Hub ({repo_id}): "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    return f"https://huggingface.co/datasets/{repo_id}"
