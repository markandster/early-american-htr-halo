"""Unit tests for the two-tier Halo validator and 1x GPU smoke harness."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import types

import pytest
import yaml

from eahtr.halo_export import (
    export_halo_vlm_preference_dataset,
    export_halo_vlm_sft_dataset,
    verify_exported_vlm_dir,
)
from eahtr.halo_smoke import (
    build_smoke_yaml_configs,
    filter_vlm_rows_post_expansion,
    prepare_gemma4_highres_dir,
    select_pytorch_cuda_index,
    verify_halo_run_log_finite_loss,
    verify_processor_image_tokens,
    verify_train_sample_image_tokens,
    write_halo_smoke_notebook,
)
from eahtr.halo_validate import validate_yaml_configs_tier1, validate_yaml_configs_tier2


def test_tier1_validator_passes_repo_configs() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    res = validate_yaml_configs_tier1(repo_root / "configs" / "halo", repo_root=repo_root)
    assert res["tier"] == 1
    assert res["count"] == 11


def test_tier1_validator_rejects_yaml11_boolean_and_ep_all_linear(tmp_path: Path) -> None:
    bad_bool = tmp_path / "bad_sft.yaml"
    bad_bool.write_text("bf16: true\npacking: no\n", encoding="utf-8")
    with pytest.raises(ValueError, match="YAML 1.1"):
        validate_yaml_configs_tier1(tmp_path, repo_root=tmp_path)

    bad_ep_dir = tmp_path / "ep"
    bad_ep_dir.mkdir()
    (bad_ep_dir / "ep_sft.yaml").write_text(
        "bf16: true\nexpert_parallel_size: 8\nuse_peft: true\nlora_target_modules: all-linear\nlora_r: 16\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="all-linear is forbidden"):
        validate_yaml_configs_tier1(bad_ep_dir, repo_root=tmp_path)


def test_tier2_validator_fails_closed_on_missing_repo_and_surfaces_mocked_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    with pytest.raises(FileNotFoundError, match="Invalid --halo-repo"):
        validate_yaml_configs_tier2(repo_root / "configs" / "halo", tmp_path / "missing_halo", repo_root=repo_root)

    fake_halo = tmp_path / "fake_halo"
    (fake_halo / "halo").mkdir(parents=True)
    monkeypatch.setattr("eahtr.halo_validate.sys.version_info", (3, 11, 9, "final", 0))
    with pytest.raises(RuntimeError, match="Python 3.12"):
        validate_yaml_configs_tier2(repo_root / "configs" / "halo", fake_halo, repo_root=repo_root)


def test_select_pytorch_cuda_index_driver_matrix() -> None:
    assert select_pytorch_cuda_index("580.65") == "https://download.pytorch.org/whl/cu130"
    assert select_pytorch_cuda_index("570.26") == "https://download.pytorch.org/whl/cu128"
    assert select_pytorch_cuda_index("560.28") == "https://download.pytorch.org/whl/cu126"
    with pytest.raises(ValueError, match="whitecircle/halo"):
        select_pytorch_cuda_index("535.104")


def test_prepare_gemma4_highres_dir_patches_all_resolution_fields(tmp_path: Path) -> None:
    src = tmp_path / "hf_cache"
    src.mkdir()
    (src / "config.json").write_text('{"model_type": "gemma4"}', encoding="utf-8")
    (src / "preprocessor_config.json").write_text('{"max_soft_tokens": 280}', encoding="utf-8")
    (src / "processor_config.json").write_text(
        json.dumps(
            {
                "image_seq_length": 280,
                "image_processor": {"max_soft_tokens": 280, "image_seq_length": 280},
            }
        ),
        encoding="utf-8",
    )

    staged = prepare_gemma4_highres_dir(src, tmp_path / "staged", max_soft_tokens=1120)
    assert (staged / "config.json").is_symlink()
    assert json.loads((staged / "preprocessor_config.json").read_text(encoding="utf-8"))["max_soft_tokens"] == 1120
    proc_cfg = json.loads((staged / "processor_config.json").read_text(encoding="utf-8"))
    assert proc_cfg["image_seq_length"] == 1120
    assert proc_cfg["image_processor"]["max_soft_tokens"] == 1120
    assert proc_cfg["image_processor"]["image_seq_length"] == 1120


def test_filter_vlm_rows_post_expansion_and_vram_smoke_configs(tmp_path: Path) -> None:
    records = [
        {"image_path": "/tmp/a.jpg", "text": "short diplomatic page"},
        {"image_path": "/tmp/b.jpg", "text": " ".join(["word"] * 900)},
    ]
    kept, dropped = filter_vlm_rows_post_expansion(
        records,
        lambda s: len(s.split()),
        max_length=2048,
        max_prompt_length=1350,
    )
    assert len(kept) == 1
    assert dropped == 1

    repo_root = Path(__file__).resolve().parents[1]
    sft_24, smpo_24 = build_smoke_yaml_configs(
        repo_root / "configs" / "halo" / "gemma4_12b_sft.yaml",
        repo_root / "configs" / "halo" / "gemma4_12b_vlm_smpo.yaml",
        tmp_path / "sm24",
        model_dir=tmp_path / "model",
        sft_dataset_dir=tmp_path / "sft_ds",
        smpo_dataset_dir=tmp_path / "smpo_ds",
        vram_gb=24.0,
        repo_root=repo_root,
    )
    sft_cfg = yaml.safe_load(sft_24.read_text(encoding="utf-8"))
    smpo_cfg = yaml.safe_load(smpo_24.read_text(encoding="utf-8"))
    assert sft_24.name == "smoke_sft.yaml"
    assert smpo_24.name == "smoke_vlm_smpo.yaml"
    assert validate_yaml_configs_tier1(tmp_path / "sm24", repo_root=repo_root)["count"] == 2
    assert sft_cfg["bf16"] is True
    assert sft_cfg["max_steps"] == 5
    assert sft_cfg["save_strategy"] == "no"
    assert smpo_cfg["max_length"] == 2048
    assert smpo_cfg["max_prompt_length"] == 1350

    _, smpo_80 = build_smoke_yaml_configs(
        repo_root / "configs" / "halo" / "gemma4_12b_sft.yaml",
        repo_root / "configs" / "halo" / "gemma4_12b_vlm_smpo.yaml",
        tmp_path / "sm80",
        model_dir=tmp_path / "model",
        sft_dataset_dir=tmp_path / "sft_ds",
        smpo_dataset_dir=tmp_path / "smpo_ds",
        vram_gb=80.0,
        repo_root=repo_root,
    )
    assert yaml.safe_load(smpo_80.read_text(encoding="utf-8"))["max_length"] == 4096


def test_verify_train_sample_image_tokens_and_smoke_notebook(tmp_path: Path) -> None:
    good_log = tmp_path / "train_sample.txt"
    good_log.write_text("<|image|>" * 560, encoding="utf-8")
    assert verify_train_sample_image_tokens(good_log) == 560

    bad_log = tmp_path / "bad_sample.txt"
    bad_log.write_text("<|image|>" * 280, encoding="utf-8")
    with pytest.raises(ValueError, match="outside"):
        verify_train_sample_image_tokens(bad_log)

    nb_path = write_halo_smoke_notebook(tmp_path / "smoke.ipynb")
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4
    assert "run_halo_smoke.sh" in "".join(nb["cells"][1]["source"])


def test_verify_processor_image_tokens_and_run_log_losses(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    img_path = tmp_path / "page.jpg"
    img_path.write_bytes(b"fake-jpg")

    class _FakeImg:
        def __enter__(self) -> "_FakeImg":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def convert(self, mode: str) -> "_FakeImg":
            return self

    pil_mod = types.ModuleType("PIL")
    img_mod = types.ModuleType("PIL.Image")
    img_mod.open = lambda p: _FakeImg()  # type: ignore[attr-defined]
    pil_mod.Image = img_mod  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "PIL", pil_mod)
    monkeypatch.setitem(sys.modules, "PIL.Image", img_mod)

    class _FakeProc:
        def __init__(self, tok_count: int) -> None:
            self.tok_count = tok_count
            self.image_token_id = 258880

        def apply_chat_template(self, msgs: object, **kw: object) -> str:
            return "<bos><|image|>Transcribe."

        def __call__(self, text: object, images: object, return_tensors: str = "pt") -> dict:
            return {"input_ids": [[2] + [258880] * self.tok_count + [10]]}

    assert verify_processor_image_tokens(_FakeProc(1120), img_path) == 1120
    with pytest.raises(ValueError, match="outside"):
        verify_processor_image_tokens(_FakeProc(280), img_path)

    log_ok = tmp_path / "ok.log"
    log_ok.write_text("{'loss': 1.42, 'grad_norm': 0.5}\n{'loss': 0.91}\n", encoding="utf-8")
    assert verify_halo_run_log_finite_loss(log_ok) == [1.42, 0.91]

    log_nan = tmp_path / "nan.log"
    log_nan.write_text("{'loss': nan}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Non-finite"):
        verify_halo_run_log_finite_loss(log_nan)

    log_empty = tmp_path / "empty.log"
    log_empty.write_text("training completed with no metrics\n", encoding="utf-8")
    with pytest.raises(ValueError, match="No training loss metrics"):
        verify_halo_run_log_finite_loss(log_empty)


def test_halo_export_helpers_and_dir_verifier(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    saved_dicts: list[tuple[str, dict]] = []

    class FakeDataset:
        def __init__(self, data: dict, features: object) -> None:
            self.data = data
            self.features = features

        @classmethod
        def from_dict(cls, data: dict, features: object) -> "FakeDataset":
            return cls(data, features)

    class FakeDatasetDict(dict):
        def save_to_disk(self, path: str) -> None:
            saved_dicts.append((path, dict(self)))

    fake_datasets = types.ModuleType("datasets")
    fake_datasets.Dataset = FakeDataset  # type: ignore[attr-defined]
    fake_datasets.DatasetDict = FakeDatasetDict  # type: ignore[attr-defined]
    fake_datasets.Features = dict  # type: ignore[attr-defined]
    fake_datasets.Image = lambda: "Image"  # type: ignore[attr-defined]
    fake_datasets.Sequence = lambda inner: ("Sequence", inner)  # type: ignore[attr-defined]
    fake_datasets.Value = lambda dtype: ("Value", dtype)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    train_rec = [
        {"record_id": "t1", "document_id": "d1", "writer_key": "w1", "image_sha256": "s1", "image_path": "/tmp/a.jpg", "text": "Dear Sir your letter arrived safely today."}
    ]
    test_rec = [
        {"record_id": "e1", "document_id": "d2", "writer_key": "w2", "image_sha256": "s2", "image_path": "/tmp/b.jpg", "text": "I remain your obedient humble servant."}
    ]
    sft_meta = export_halo_vlm_sft_dataset(train_rec, test_rec, tmp_path / "sft_out", lambda s: len(s.split()))
    assert sft_meta["train_rows"] == 1
    assert saved_dicts[-1][1]["train"].features["images"] == ("Sequence", "Image")

    with pytest.raises(ValueError, match="Split disjointness"):
        export_halo_vlm_sft_dataset(train_rec, train_rec, tmp_path / "bad_sft", lambda s: len(s.split()))

    pair_train = [{"images": ["/tmp/a.jpg"], "prompt": [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Transcribe."}]}], "chosen": [{"role": "assistant", "content": "Dear Sir"}], "rejected": [{"role": "assistant", "content": "Bad"}], "margin": 0.8}]
    pair_test = [{"images": ["/tmp/b.jpg"], "prompt": [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Transcribe."}]}], "chosen": [{"role": "assistant", "content": "Humble"}], "rejected": [{"role": "assistant", "content": "Bad"}], "margin": 0.5}]
    pref_meta = export_halo_vlm_preference_dataset(pair_train, pair_test, tmp_path / "pref_out", lambda s: len(s.split()))
    assert pref_meta["include_margin"] is True
    assert "margin" in saved_dicts[-1][1]["train"].data

    ok_dir = tmp_path / "ok_mlx"
    ok_dir.mkdir()
    (ok_dir / "processor_config.json").write_text(
        json.dumps({"image_seq_length": 1120, "image_processor": {"max_soft_tokens": 1120, "image_seq_length": 1120}}),
        encoding="utf-8",
    )
    (ok_dir / "chat_template.jinja").write_text("{%- set eff = 0.0 -%}", encoding="utf-8")
    assert verify_exported_vlm_dir(ok_dir)["image_seq_length"] == 1120

    bad_dir = tmp_path / "bad_mlx"
    bad_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="processor_config.json"):
        verify_exported_vlm_dir(bad_dir)

    (ok_dir / "chat_template.jinja").write_text("Thinking effort level: 0.9", encoding="utf-8")
    with pytest.raises(ValueError, match="0.9 reasoning-effort"):
        verify_exported_vlm_dir(ok_dir)
