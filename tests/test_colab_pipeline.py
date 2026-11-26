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
    CompetitionCandidateSpec,
    assign_balanced_group_splits,
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
    assert set(COMPETITION_CANDIDATES) == {"gemma4_12b"}
    assert COMPETITION_CANDIDATES["gemma4_12b"].hf_revision == "43c9d014b231"
    assert "vision_tower" in COMPETITION_CANDIDATES["gemma4_12b"].vision_trainable_patterns
    assert COMPETITION_CANDIDATES["gemma4_12b"].lora_target_modules == ("all-linear",)
    for spec in COMPETITION_CANDIDATES.values():
        validate_competition_candidate(spec)

    assert parse_revcity_stem_id("apsrevcity%3A1008") == ("apsrevcity", 1008, "apsrevcity_1008")
    assert parse_revcity_stem_id("lcprevcity_159_OBJ") == ("lcprevcity", 159, "lcprevcity_159_OBJ")
    groups = assign_revcity_item_groups(
        ["apsrevcity%3A1008", "apsrevcity%3A1012", "apsrevcity%3A1025", "lcprevcity_159_OBJ", "mss_b_h56d_008-001", "mss_b_h56d_008-003"],
        creator_map={"lcprevcity_159_OBJ": "hand_xyz"},
    )
    assert groups["apsrevcity%3A1008"]["document_id"] == groups["apsrevcity%3A1012"]["document_id"]
    assert groups["apsrevcity%3A1025"]["document_id"] != groups["apsrevcity%3A1008"]["document_id"]
    assert groups["mss_b_h56d_008-001"]["document_id"] == groups["mss_b_h56d_008-003"]["document_id"]
    assert groups["lcprevcity_159_OBJ"]["writer_key"] == "scribed_hand:hand_xyz"
    assert groups["lcprevcity_159_OBJ"]["hand_attributed"] == "true"

    col_groups = assign_revcity_item_groups(
        ["apsrevcity%3A16255", "apsrevcity%3A16700", "apsrevcity%3A7228", "apsrevcity%3A7540", "apsrevcity%3A14950", "apsrevcity%3A15229"],
        max_id_gap=30,
    )
    assert col_groups["apsrevcity%3A16255"]["document_id"] == col_groups["apsrevcity%3A16700"]["document_id"]
    assert col_groups["apsrevcity%3A7228"]["document_id"] == col_groups["apsrevcity%3A7540"]["document_id"]
    assert col_groups["apsrevcity%3A14950"]["document_id"] == col_groups["apsrevcity%3A15229"]["document_id"]


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

    vision_tower_modules = [
        "model.vision_tower",
        "model.vision_tower.encoder",
        "model.multi_modal_projector",
        "model.multi_modal_projector.linear",
    ]
    assert resolve_vision_modules_to_save(
        vision_tower_modules, COMPETITION_CANDIDATES["gemma4_12b"].vision_trainable_patterns
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

    stats = {
        "giant_diary": {"pages": 194, "institution": "hsp"},
        **{f"hsp_{i}": {"pages": 12, "institution": "hsp"} for i in range(10)},
        **{f"aps_{i}": {"pages": 10, "institution": "aps"} for i in range(10)},
        **{f"lcp_{i}": {"pages": 8, "institution": "lcprevcity"} for i in range(10)},
    }
    balanced = assign_balanced_group_splits(stats, val_fraction=0.10, test_fraction=0.10, max_eval_group_pages=40)
    assert balanced["giant_diary"] == "train"
    val_groups = [g for g, sp in balanced.items() if sp == "val"]
    test_groups = [g for g, sp in balanced.items() if sp == "test_public"]
    assert len(val_groups) >= 3 and len(test_groups) >= 3
    assert {stats[g]["institution"] for g in val_groups} == {"hsp", "aps", "lcprevcity"}
    assert {stats[g]["institution"] for g in test_groups} == {"hsp", "aps", "lcprevcity"}


def test_should_use_4bit_across_gpus_and_arms() -> None:
    g12 = COMPETITION_CANDIDATES["gemma4_12b"]
    high_vram_spec = CompetitionCandidateSpec(
        key="synthetic_large",
        hf_repo="example/synthetic-large",
        license="Apache-2.0",
        params_b=24.0,
        active_params_b=8.0,
        lora_target_modules=("all-linear",),
        vision_trainable_patterns=("vision_tower",),
        max_pixels=2048 * 2048,
        mlx_quant_bits=4,
        est_mlx_peak_gb=16.0,
        min_bf16_vram_gb=72.0,
    )

    # L4 24 GB requires 4-bit for both zero-shot and training
    for spec in (g12, high_vram_spec):
        assert should_use_4bit(spec, 24.0, "baseline_zeroshot") is True
        assert should_use_4bit(spec, 24.0, "real") is True

    # A100 40 GB runs 12B zero-shot in BF16, but 12B training and higher-VRAM specs in 4-bit
    assert should_use_4bit(g12, 40.0, "baseline_zeroshot") is False
    assert should_use_4bit(g12, 40.0, "real") is True
    assert should_use_4bit(high_vram_spec, 40.0, "baseline_zeroshot") is True
    assert should_use_4bit(high_vram_spec, 40.0, "real") is True

    # H100 / A100 80 GB runs both in BF16 cleanly
    for spec in (g12, high_vram_spec):
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
        "dhash_review_candidates",
        "discarded_oom_microbatches",
        "max_id_gap=30",
        "_merge_saved_adapter",
        "_fold_key",
        "_assign_balanced_folds",
        "CHECKPOINT_EVERY_OPT_STEPS",
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
        "verify_exported_vlm_dir",
        "assign_balanced_group_splits",
        "REVCITY_ZENODO_MD5",
        "hard_case_held_out",
        "ROLLOUT_SOURCE",
        "ALLOW_IN_SAMPLE_TRAIN_ROLLOUTS",
        "transformers==5.23.0",
        "PINNED_REPO_COMMIT",
        "is_out_of_fold",
        "image_dhash",
        "hf_revision",
        "_stratified_eval_slice",
        "OOF_NUM_FOLDS",
        "has_empty_alto_text",
        "missing_or_unverified_gt",
        "_select_doc_disjoint_pilot",
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
        "gemma4_12b_vlm_smpo.yaml",
        "gemma4_26b_a4b_sft.yaml",
        "gemma4_26b_a4b_vlm_smpo.yaml",
        "gemma4_31b_sft.yaml",
        "gemma4_31b_vlm_smpo.yaml",
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

    for name in (
        "gemma4_12b_sft.yaml",
        "gemma4_12b_vlm_smpo.yaml",
        "gemma4_26b_a4b_sft.yaml",
        "gemma4_26b_a4b_vlm_smpo.yaml",
        "gemma4_31b_sft.yaml",
        "gemma4_31b_vlm_smpo.yaml",
    ):
        assert loaded[name]["attn_implementation"] == "sdpa"
        assert loaded[name]["force_chat_template"] is True
        assert loaded[name]["gradient_checkpointing"] is True
    for name in ("gemma4_12b_sft.yaml", "gemma4_12b_vlm_smpo.yaml", "gemma4_31b_sft.yaml", "gemma4_31b_vlm_smpo.yaml"):
        assert loaded[name]["lora_target_modules"] == "all-linear"
    for name in ("gemma4_26b_a4b_sft.yaml", "gemma4_26b_a4b_vlm_smpo.yaml"):
        assert loaded[name]["expert_parallel_size"] == 8
        assert loaded[name]["moe_balancing"] == "none"
        assert loaded[name]["use_peft"] is False
    for name in (
        "gemma4_12b_vlm_smpo.yaml",
        "gemma4_26b_a4b_vlm_smpo.yaml",
        "gemma4_31b_vlm_smpo.yaml",
        "inkling2_small_vlm_smpo.yaml",
    ):
        cfg = loaded[name]
        assert cfg["loss_type"] == "smooth_lower_bound"
        assert cfg["use_margin_schedule"] is True
        assert cfg["chosen_sft_ratio"] == 1.0
        assert "images_field" not in cfg
        assert "truncation_mode" not in cfg
        assert cfg.get("freeze_vision_tower") is not True
        if cfg.get("use_peft"):
            assert cfg["lora_dropout"] == 0.0
        assert float(cfg["learning_rate"]) <= 5.0e-6

    for name in (
        "inkling2_small_sft.yaml",
        "inkling2_small_vlm_smpo.yaml",
        "inkling2_offline_grpo.yaml",
    ):
        assert loaded[name]["pad_token"] == "<|endoftext|>"

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


def test_colab_completion_only_logits_and_oom_cleanup() -> None:
    import ast
    import importlib.util
    from eahtr.colab_cells_train import build_cell_5_train_helpers
    from eahtr.colab_pipeline import build_shifted_completion_mask

    cell5_src = build_cell_5_train_helpers()
    ast.parse(cell5_src)
    assert "logits_to_keep=keep_mask" in cell5_src
    assert "get_input_embeddings" in cell5_src and "torch.bfloat16" in cell5_src
    assert "torch.optim.AdamW" in cell5_src
    assert "PagedAdamW8bit" not in cell5_src
    assert "gc.collect()" in cell5_src

    if importlib.util.find_spec("torch") is not None:
        import torch
        import torch.nn.functional as F

        torch.manual_seed(0)
        vocab = 19
        logits = torch.randn(1, 10, vocab, dtype=torch.float32)
        labels = torch.tensor([[-100, -100, -100, -100, 4, 7, 2, -100, 9, 1]], dtype=torch.long)

        # Standard causal-LM loss over full sequence
        full_loss = F.cross_entropy(logits[:, :-1, :].reshape(-1, vocab), labels[:, 1:].reshape(-1), ignore_index=-100)

        # Shifted logits_to_keep loss over active positions only
        keep_mask, targets = build_shifted_completion_mask(labels)
        kept_logits = logits[keep_mask]
        compact_loss = F.cross_entropy(kept_logits.reshape(-1, vocab), targets.reshape(-1))
        assert abs(float(full_loss - compact_loss)) <= 1e-6

        # Unshifted mask on positions 4..9 produces a genuinely wrong loss (catches off-by-one shift bug)
        unshifted_mask = labels != -100
        wrong_loss = F.cross_entropy(logits[unshifted_mask].reshape(-1, vocab), targets.reshape(-1))
        assert abs(float(full_loss - wrong_loss)) > 1e-2
