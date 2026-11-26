"""Unit tests for the competition-only Colab Pro pipeline and notebook generator."""

import ast
import json
from pathlib import Path

from jinja2 import Environment
from jinja2.ext import Extension
import pytest
import yaml


class _GenerationTagExtension(Extension):
    """Minimal passthrough for HF `{% generation %}...{% endgeneration %}` Jinja blocks."""

    tags = {"generation"}

    def parse(self, parser):  # type: ignore[override]
        next(parser.stream)
        body = parser.parse_statements(["name:endgeneration"], drop_needle=True)
        return body

from eahtr.colab_pipeline import (
    COMPETITION_CANDIDATES,
    assign_deterministic_split,
    assign_revcity_item_groups,
    make_sham_derangement,
    parse_revcity_stem_id,
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
    }
    assert "vision_tower" in COMPETITION_CANDIDATES["gemma4_12b"].vision_trainable_patterns
    assert COMPETITION_CANDIDATES["gemma4_12b"].lora_target_modules == ("all-linear",)
    assert COMPETITION_CANDIDATES["gemma4_26b_a4b"].lora_target_modules == ("all-linear",)
    assert COMPETITION_CANDIDATES["inkling2_mini"].quant_method == "modelopt"
    assert "r_proj" in COMPETITION_CANDIDATES["inkling2_mini"].lora_target_modules
    for spec in COMPETITION_CANDIDATES.values():
        validate_competition_candidate(spec)

    assert parse_revcity_stem_id("apsrevcity%3A1008") == ("apsrevcity", 1008, "apsrevcity_1008")
    assert parse_revcity_stem_id("lcprevcity_159_OBJ") == ("lcprevcity", 159, "lcprevcity_159_OBJ")
    groups = assign_revcity_item_groups(
        ["apsrevcity%3A1008", "apsrevcity%3A1009", "apsrevcity%3A1015", "lcprevcity_159_OBJ"],
        creator_map={"lcprevcity_159_OBJ": "hand_xyz"},
    )
    assert groups["apsrevcity%3A1008"]["document_id"] == groups["apsrevcity%3A1009"]["document_id"]
    assert groups["apsrevcity%3A1015"]["document_id"] != groups["apsrevcity%3A1008"]["document_id"]
    assert groups["lcprevcity_159_OBJ"]["writer_key"] == "scribed_hand:hand_xyz"
    assert groups["lcprevcity_159_OBJ"]["hand_attributed"] == "true"


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

    # L4 24 GB requires 4-bit for both zero-shot and training across all candidates
    for spec in (g12, g26, ink):
        assert should_use_4bit(spec, 24.0, "baseline_zeroshot") is True
        assert should_use_4bit(spec, 24.0, "real") is True

    # A100 40 GB runs 12B zero-shot in BF16, but 26B zero-shot and all training in 4-bit
    assert should_use_4bit(g12, 40.0, "baseline_zeroshot") is False
    assert should_use_4bit(g12, 40.0, "real") is True
    for spec in (g26, ink):
        assert should_use_4bit(spec, 40.0, "baseline_zeroshot") is True
        assert should_use_4bit(spec, 40.0, "real") is True

    # H100 / A100 80 GB runs 12B, 26B, and Inkling in BF16 cleanly
    for spec in (g12, g26, ink):
        assert should_use_4bit(spec, 80.0, "baseline_zeroshot") is False
        assert should_use_4bit(spec, 80.0, "real") is False


def test_generated_notebook_valid_json_ast_and_public_only(tmp_path: Path) -> None:
    nb_path = write_competition_colab_notebook(tmp_path / "test_colab.ipynb")
    raw_text = nb_path.read_text(encoding="utf-8")
    expected_markers = (
        "SMOKE_TEST_MODE",
        "AutoModelForMultimodalLM",
        "userdata",
        "should_use_4bit",
        "PUBLISH_TO_HF_DATASET",
        "assign_revcity_item_groups",
        "FINAL_FREEZE_EVAL_TEST_PUBLIC",
        "swap_images",
        "score_line_breaks",
        "paired_bootstrap_ci",
        "prefer_gold_chosen=True",
        "_eval_scores_{cache_key}",
        "hard_case_panel",
        "USE_GOOGLE_DRIVE",
        "llm_int8_skip_modules",
        "gradient_checkpointing_kwargs",
        "use_reentrant",
        "max_soft_tokens = 1120",
        "image_seq_length = 1120",
        "attn_implementation",
        "MAX_SEQ_LEN = 4096",
        "MAX_EVAL_NEW_TOKENS = 2048",
        "torch.equal",
        "_resolve_turn_terminator",
        "has_live_vision_grad",
        "accum_count",
        "thinking_leak_pages",
        "enable_thinking=False",
        "val_test_audit",
        "python -m mlx_vlm convert",
        "build_preference_pairs",
        "ROLLOUT_TEMPS",
        "exclude_modules=vision_modules",
        "modelopt",
        "SC_PHYS_PAGES",
        "Missing required",
    )
    for marker in expected_markers:
        assert marker in raw_text, f"Missing expected notebook marker: {marker}"
    for forbidden in ("q" + "wen", "worst3_failures_public_test"):
        assert forbidden not in raw_text.lower(), f"Forbidden stale string in notebook: {forbidden}"
    assert "upgrade transformers peft accelerate bitsandbytes rapidfuzz pyyaml pillow" not in raw_text

    nb = json.loads(raw_text)
    assert nb["nbformat"] == 4
    assert len(nb["cells"]) == 8

    for idx, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        code_lines = [ln for ln in cell["source"] if not ln.lstrip().startswith("!")]
        ast.parse("".join(code_lines), filename=f"cell_{idx}.py")


def test_halo_yaml_configs_match_schema_and_ink2_rules() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    config_dir = repo_root / "configs" / "halo"
    expected_files = {
        "gemma4_12b_sft.yaml",
        "gemma4_26b_a4b_sft.yaml",
        "gemma4_12b_vlm_smpo.yaml",
        "gemma4_26b_a4b_vlm_smpo.yaml",
        "gemma4_12b_vlm_dpo.yaml",
        "inkling2_mini_sft.yaml",
        "inkling2_small_sft.yaml",
        "inkling2_small_vlm_smpo.yaml",
        "archival_grpo_offline.yaml",
        "inkling2_offline_grpo.yaml",
        "archival_grpo_online.yaml",
    }
    yaml_files = [p for p in config_dir.glob("*.yaml") if not p.name.startswith("._")]
    assert {p.name for p in yaml_files} == expected_files

    refused_keys = {"assistant_only_loss", "completion_only_loss", "max_pixels", "warmup_ratio"}
    loaded: dict[str, dict] = {}
    for p in yaml_files:
        cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
        assert isinstance(cfg, dict)
        assert not (refused_keys & set(cfg)), f"{p.name} contains Halo startup-refused keys"
        assert cfg.get("bf16") is True, f"{p.name} must pin bf16: true"
        assert "test_size" not in cfg, f"{p.name} must omit test_size on pre-split data"
        if cfg.get("expert_parallel_size", 1) > 1 and cfg.get("use_peft"):
            assert cfg.get("lora_target_modules") != "all-linear", (
                f"{p.name} must not use all-linear under EP > 1"
            )
        if "chat_template" in cfg:
            tpl_path = repo_root / str(cfg["chat_template"])
            assert tpl_path.is_file(), f"{p.name} references missing template {tpl_path}"
        loaded[p.name] = cfg

    for name in ("gemma4_12b_sft.yaml", "gemma4_12b_vlm_smpo.yaml", "gemma4_12b_vlm_dpo.yaml"):
        assert loaded[name]["attn_implementation"] == "sdpa"
        assert loaded[name]["lora_target_modules"] == "all-linear"
        assert loaded[name]["force_chat_template"] is True
    for name in ("gemma4_26b_a4b_sft.yaml", "gemma4_26b_a4b_vlm_smpo.yaml"):
        assert loaded[name]["attn_implementation"] == "sdpa"
        assert loaded[name]["expert_parallel_size"] == 8
        assert loaded[name]["lora_r"] % 8 == 0
        assert set(loaded[name]["lora_target_modules"]) >= {
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_up_proj",
            "down_proj",
        }
    for name in ("gemma4_12b_vlm_smpo.yaml", "gemma4_26b_a4b_vlm_smpo.yaml", "inkling2_small_vlm_smpo.yaml"):
        cfg = loaded[name]
        assert cfg["loss_type"] == "smooth_lower_bound"
        assert cfg["use_margin_schedule"] is True
        assert "images_field" not in cfg
        assert "truncation_mode" not in cfg
        assert cfg.get("freeze_vision_tower") is not True
        assert cfg["lora_dropout"] == 0.0
        assert float(cfg["learning_rate"]) <= 5.0e-6

    dpo_cfg = loaded["gemma4_12b_vlm_dpo.yaml"]
    assert dpo_cfg["images_field"] == "images"
    assert dpo_cfg["loss_type"] == ["sigmoid", "sft"]
    assert "precompute_ref_log_probs" not in dpo_cfg
    assert "generation_max_prompt_length" not in dpo_cfg
    assert "max_prompt_length" not in dpo_cfg

    for name in (
        "inkling2_mini_sft.yaml",
        "inkling2_small_sft.yaml",
        "inkling2_small_vlm_smpo.yaml",
        "inkling2_offline_grpo.yaml",
    ):
        assert loaded[name]["pad_token"] == "<|endoftext|>"

    assert "r_proj" in loaded["inkling2_mini_sft.yaml"]["lora_target_modules"]
    for name in ("inkling2_small_sft.yaml", "inkling2_small_vlm_smpo.yaml", "inkling2_offline_grpo.yaml"):
        assert loaded[name]["frozen_quantized_experts"] is True
        assert loaded[name]["expert_parallel_size"] >= 2
        assert loaded[name]["lora_r"] % 8 == 0
        assert "o_proj" in loaded[name]["lora_target_modules"]


def test_halo_jinja_templates_satisfy_smpo_and_inference_prefix_invariant() -> None:
    tpl_dir = Path(__file__).resolve().parents[1] / "configs" / "halo" / "templates"
    env = Environment(extensions=[_GenerationTagExtension])
    env.globals["raise_exception"] = lambda msg: (_ for _ in ()).throw(ValueError(msg))

    prompt_msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Transcribe."}]}]
    completion_msgs = [{"role": "assistant", "content": [{"type": "text", "text": "Dear Sir,"}]}]

    g4_tpl = env.from_string((tpl_dir / "gemma4_htr_nothink.jinja").read_text(encoding="utf-8"))
    g4_prompt_closed = g4_tpl.render(messages=prompt_msgs, bos_token="<bos>", add_generation_prompt=False)
    g4_prompt_open = g4_tpl.render(messages=prompt_msgs, bos_token="<bos>", add_generation_prompt=True)
    g4_full = g4_tpl.render(messages=prompt_msgs + completion_msgs, bos_token="<bos>", add_generation_prompt=False)
    assert g4_full.startswith(g4_prompt_closed)
    assert g4_full.startswith(g4_prompt_open)
    assert "<|turn>model\n<|channel>thought\n<channel|>" in g4_prompt_open

    ink_tpl = env.from_string((tpl_dir / "inkling2_htr_nothink.jinja").read_text(encoding="utf-8"))
    ink_prompt_closed = ink_tpl.render(messages=prompt_msgs, add_generation_prompt=False)
    ink_prompt_open = ink_tpl.render(messages=prompt_msgs, add_generation_prompt=True)
    ink_full = ink_tpl.render(messages=prompt_msgs + completion_msgs, add_generation_prompt=False)
    assert ink_full.startswith(ink_prompt_closed)
    assert ink_full.startswith(ink_prompt_open)
    assert "Thinking effort level: 0<" in ink_prompt_open
    assert "Thinking effort level: 0.9" not in ink_prompt_open
