"""Unit tests for the two-tier Halo validator and 1x GPU smoke harness."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from eahtr.halo_smoke import (
    build_smoke_yaml_configs,
    filter_vlm_rows_post_expansion,
    prepare_gemma4_highres_dir,
    select_pytorch_cuda_index,
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
