"""Unit tests for the opt-in Hugging Face public dataset publisher."""

import pytest
import yaml

from eahtr.hf_dataset import build_hf_dataset_card, validate_records_for_public_publish


def _sample_record(idx: int, *, split: str, lic: str = "CC-BY-4.0") -> dict[str, object]:
    text = " ".join(f"tok{idx}_{w}" for w in range(40))
    return {
        "record_id": f"rec-{split}-{idx}",
        "collection": "revcity-zenodo-15776323",
        "document_id": f"doc-{split}-{idx}",
        "writer_key": f"writer-{split}-{idx}",
        "image_path": f"/tmp/{split}_{idx}.jpg",
        "image_sha256": f"sha-{split}-{idx}",
        "text": text,
        "word_count": len(text.split()),
        "license": lic,
        "split": split,
    }


def test_validate_records_for_public_publish_accepts_clean_splits() -> None:
    splits = {
        "train": [_sample_record(1, split="train", lic="CC-BY-4.0")],
        "val": [_sample_record(2, split="val", lic="Public Domain")],
        "test_public": [_sample_record(3, split="test_public", lic="CC0-1.0")],
    }
    res = validate_records_for_public_publish(splits)
    assert res["passed"] is True
    assert res["splits"] == {"train": 1, "val": 1, "test_public": 1}
    assert res["licenses"] == ["CC-BY-4.0", "CC0-1.0", "Public Domain"]


def test_validate_records_rejects_non_commercial_or_empty_train() -> None:
    with pytest.raises(ValueError, match="empty"):
        validate_records_for_public_publish({"train": [], "val": [_sample_record(1, split="val")]})

    nc_splits = {
        "train": [_sample_record(1, split="train", lic="CC-BY-NC-4.0")],
        "val": [_sample_record(2, split="val")],
    }
    with pytest.raises(ValueError, match="CC-BY-NC-4.0"):
        validate_records_for_public_publish(nc_splits)


def test_validate_records_rejects_writer_doc_and_ngram_overlap() -> None:
    train_rec = _sample_record(1, split="train")
    val_same_writer = _sample_record(2, split="val")
    val_same_writer["writer_key"] = train_rec["writer_key"]
    with pytest.raises(ValueError, match="leakage detected"):
        validate_records_for_public_publish({"train": [train_rec], "val": [val_same_writer]})

    val_same_ngram = _sample_record(3, split="val")
    val_same_ngram["text"] = train_rec["text"]
    with pytest.raises(ValueError, match="leakage detected"):
        validate_records_for_public_publish({"train": [train_rec], "val": [val_same_ngram]})


def test_build_hf_dataset_card_has_valid_yaml_and_attribution() -> None:
    card = build_hf_dataset_card(
        "markandster/early-american-htr-public",
        {"train_sides": 300, "val_sides": 25, "test_public_sides": 25},
        ["CC-BY-4.0", "Public Domain"],
    )
    parts = card.split("---", 2)
    assert len(parts) == 3
    frontmatter = yaml.safe_load(parts[1])
    assert frontmatter["license"] == "cc-by-4.0"
    assert "image-to-text" in frontmatter["task_categories"]
    assert "15776323" in parts[2]
    assert "By the People" in parts[2]
