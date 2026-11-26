"""Unit tests for the opt-in Hugging Face public dataset publisher."""

import sys
import types
from typing import Any

import pytest
import yaml

from eahtr.hf_dataset import (
    build_hf_dataset_card,
    push_splits_to_hf_hub,
    validate_records_for_public_publish,
)


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

    test_same_writer = _sample_record(4, split="test_public")
    val_clean = _sample_record(5, split="val")
    test_same_writer["writer_key"] = val_clean["writer_key"]
    with pytest.raises(ValueError, match="val.*test_public"):
        validate_records_for_public_publish(
            {"train": [train_rec], "val": [val_clean], "test_public": [test_same_writer]}
        )


def test_push_splits_to_hf_hub_emits_halo_schema_and_wraps_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    pushed_holders: list[dict[str, Any]] = []

    class FakeFeature:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.args = args
            self.kwargs = kwargs

    class FakeFeatures(dict):
        pass

    class FakeDataset:
        def __init__(self, data: dict[str, Any], features: FakeFeatures) -> None:
            self.data = data
            self.features = features

        @classmethod
        def from_dict(cls, data: dict[str, Any], features: FakeFeatures) -> "FakeDataset":
            return cls(data, features)

    class FakeDatasetDict(dict):
        def push_to_hub(self, repo_id: str, private: bool = False) -> None:
            pushed_holders.append({"repo_id": repo_id, "private": private, "splits": dict(self)})

    fake_datasets = types.ModuleType("datasets")
    fake_datasets.Dataset = FakeDataset  # type: ignore[attr-defined]
    fake_datasets.DatasetDict = FakeDatasetDict  # type: ignore[attr-defined]
    fake_datasets.Features = FakeFeatures  # type: ignore[attr-defined]
    fake_datasets.Image = FakeFeature  # type: ignore[attr-defined]
    fake_datasets.Sequence = lambda inner: ("Sequence", inner)  # type: ignore[attr-defined]
    fake_datasets.Value = lambda dtype: ("Value", dtype)  # type: ignore[attr-defined]

    uploaded_files: list[str] = []

    class WorkingHfApi:
        def create_repo(self, **kwargs: object) -> None:
            return None

        def upload_file(self, **kwargs: object) -> None:
            uploaded_files.append(str(kwargs["path_in_repo"]))

    fake_hf_hub = types.ModuleType("huggingface_hub")
    fake_hf_hub.HfApi = WorkingHfApi  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf_hub)

    splits = {
        "train": [_sample_record(1, split="train")],
        "val": [_sample_record(2, split="val")],
        "test_public": [_sample_record(3, split="test_public")],
    }
    url = push_splits_to_hf_hub(splits, "markandster/test-htr", {"train_sides": 1})
    assert url == "https://huggingface.co/datasets/markandster/test-htr"
    assert uploaded_files == ["README.md"]
    assert set(pushed_holders[0]["splits"]) == {"train", "validation", "test_public"}
    train_ds = pushed_holders[0]["splits"]["train"]
    assert train_ds.data["images"] == [["/tmp/train_1.jpg"]]
    assert train_ds.data["messages"][0][0]["content"][0]["type"] == "image"
    assert train_ds.features["images"][0] == "Sequence"

    class FailingHfApi:
        def create_repo(self, **kwargs: object) -> None:
            raise OSError("network unreachable")

    fake_hf_hub.HfApi = FailingHfApi  # type: ignore[attr-defined]
    with pytest.raises(RuntimeError, match="Failed pushing public dataset"):
        push_splits_to_hf_hub(splits, "markandster/test-htr", {"train_sides": 1})


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
