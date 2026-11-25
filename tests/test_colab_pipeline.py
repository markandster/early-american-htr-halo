"""Unit tests for the competition-only Colab Pro pipeline and notebook generator."""

import ast
import json
from pathlib import Path

import pytest
import yaml

from eahtr.colab_pipeline import (
    COMPETITION_CANDIDATES,
    assign_deterministic_split,
    make_sham_derangement,
    resolve_vision_modules_to_save,
    should_use_4bit,
    validate_competition_candidate,
    write_competition_colab_notebook,
)


def test_all_competition_candidates_pass_policy() -> None:
    assert set(COMPETITION_CANDIDATES) == {
        "gemma4_12b",
        "gemma4_26b_a4b",
        "inkling2_mini",
        "qwen36_35b_a3b",
    }
    assert "r_proj" in COMPETITION_CANDIDATES["inkling2_mini"].lora_target_modules
    for spec in COMPETITION_CANDIDATES.values():
        validate_competition_candidate(spec)


def test_resolve_vision_modules_across_model_families() -> None:
    gemma12b_modules = [
        "model",
        "model.embed_vision",
        "model.embed_vision.patch_dense",
        "model.embed_vision.multimodal_embedder",
        "model.embed_vision.multimodal_embedder.embedding_projection",
        "model.language_model.layers.0.self_attn.q_proj",
    ]
    assert resolve_vision_modules_to_save(
        gemma12b_modules, COMPETITION_CANDIDATES["gemma4_12b"].vision_trainable_patterns
    ) == ["model.embed_vision"]

    gemma26b_modules = [
        "model.vision_tower",
        "model.vision_tower.encoder",
        "model.multi_modal_projector",
        "model.multi_modal_projector.linear",
    ]
    assert resolve_vision_modules_to_save(
        gemma26b_modules, COMPETITION_CANDIDATES["gemma4_26b_a4b"].vision_trainable_patterns
    ) == ["model.multi_modal_projector", "model.vision_tower"]

    qwen_modules = [
        "model.visual.merger",
        "model.visual.merger.linear_fc1",
        "model.visual.deepstack_merger_list",
    ]
    assert resolve_vision_modules_to_save(
        qwen_modules, COMPETITION_CANDIDATES["qwen36_35b_a3b"].vision_trainable_patterns
    ) == ["model.visual.deepstack_merger_list", "model.visual.merger"]

    with pytest.raises(RuntimeError, match="Refusing frozen-vision run"):
        resolve_vision_modules_to_save(
            ["model.layers.0.q_proj"], COMPETITION_CANDIDATES["gemma4_12b"].vision_trainable_patterns
        )


def test_make_sham_derangement_zero_self_matches() -> None:
    records = [{"record_id": f"r-{i}", "text": f"letter text {i}"} for i in range(12)]
    sham = make_sham_derangement(records, seed=20261125)
    assert len(sham) == len(records)
    for orig, s in zip(records, sham, strict=True):
        assert s["text"] != orig["text"]
        assert s["sham_donor_record_id"] != orig["record_id"]
        assert s["arm"] == "sham"


def test_assign_deterministic_split_groups_consistently() -> None:
    first = assign_deterministic_split("revcity:series_001")
    second = assign_deterministic_split("revcity:series_001")
    assert first == second

    splits = {assign_deterministic_split(f"series_{i}") for i in range(200)}
    assert splits == {"train", "val", "test_public"}


def test_should_use_4bit_across_gpus_and_arms() -> None:
    g12 = COMPETITION_CANDIDATES["gemma4_12b"]
    g26 = COMPETITION_CANDIDATES["gemma4_26b_a4b"]
    ink = COMPETITION_CANDIDATES["inkling2_mini"]
    q35 = COMPETITION_CANDIDATES["qwen36_35b_a3b"]

    # L4 24 GB requires 4-bit for both zero-shot and training across all candidates
    for spec in (g12, g26, ink, q35):
        assert should_use_4bit(spec, 24.0, "baseline_zeroshot") is True
        assert should_use_4bit(spec, 24.0, "real") is True

    # A100 40 GB runs 12B zero-shot in BF16, but 26B/35B zero-shot and all training in 4-bit
    assert should_use_4bit(g12, 40.0, "baseline_zeroshot") is False
    assert should_use_4bit(g12, 40.0, "real") is True
    for spec in (g26, ink, q35):
        assert should_use_4bit(spec, 40.0, "baseline_zeroshot") is True
        assert should_use_4bit(spec, 40.0, "real") is True

    # H100 / A100 80 GB runs 12B, 26B, and 35B in BF16 cleanly
    for spec in (g12, g26, ink, q35):
        assert should_use_4bit(spec, 80.0, "baseline_zeroshot") is False
        assert should_use_4bit(spec, 80.0, "real") is False


def test_generated_notebook_valid_json_ast_and_public_only(tmp_path: Path) -> None:
    nb_path = write_competition_colab_notebook(tmp_path / "test_colab.ipynb")
    raw_text = nb_path.read_text(encoding="utf-8")
    assert "SMOKE_TEST_MODE" in raw_text
    assert "AutoModelForMultimodalLM" in raw_text
    assert "userdata" in raw_text
    assert "should_use_4bit" in raw_text
    assert "PUBLISH_TO_HF_DATASET" in raw_text
    assert "INCLUDE_BENTHAM" in raw_text
    assert "parse_page_xml_lines" in raw_text

    nb = json.loads(raw_text)
    assert nb["nbformat"] == 4
    assert len(nb["cells"]) == 8

    for idx, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        code_lines = [ln for ln in cell["source"] if not ln.lstrip().startswith("!")]
        ast.parse("".join(code_lines), filename=f"cell_{idx}.py")


def test_halo_yaml_configs_match_schema_and_ink2_rules() -> None:
    config_dir = Path(__file__).resolve().parents[1] / "configs" / "halo"
    expected_files = {
        "gemma4_12b_sft.yaml",
        "gemma4_26b_a4b_sft.yaml",
        "inkling2_mini_sft.yaml",
        "inkling2_small_sft.yaml",
        "archival_grpo_offline.yaml",
        "inkling2_offline_grpo.yaml",
        "archival_grpo_online.yaml",
    }
    yaml_files = [p for p in config_dir.glob("*.yaml") if not p.name.startswith("._")]
    assert {p.name for p in yaml_files} == expected_files

    refused_keys = {"assistant_only_loss", "completion_only_loss", "max_pixels", "warmup_ratio"}
    loaded = {}
    for p in yaml_files:
        cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
        assert isinstance(cfg, dict)
        assert not (refused_keys & set(cfg)), f"{p.name} contains Halo startup-refused keys"
        loaded[p.name] = cfg

    for name in ("gemma4_12b_sft.yaml", "gemma4_26b_a4b_sft.yaml"):
        assert loaded[name]["attn_implementation"] == "sdpa"
        assert loaded[name]["lora_target_modules"] == "all-linear"
        assert loaded[name]["train_on_completions_only"] is True

    for name in ("inkling2_mini_sft.yaml", "inkling2_small_sft.yaml", "inkling2_offline_grpo.yaml"):
        assert loaded[name]["pad_token"] == "<|endoftext|>"

    assert "r_proj" in loaded["inkling2_mini_sft.yaml"]["lora_target_modules"]
    for name in ("inkling2_small_sft.yaml", "inkling2_offline_grpo.yaml"):
        assert loaded[name]["frozen_quantized_experts"] is True
        assert "o_proj" in loaded[name]["lora_target_modules"]
