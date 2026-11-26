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
    verify_stage_continuity_logprobs,
    verify_train_sample_image_tokens,
    write_halo_smoke_notebook,
)
from eahtr.halo_validate import validate_yaml_configs_tier1, validate_yaml_configs_tier2


def test_tier1_validator_passes_repo_configs() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    res = validate_yaml_configs_tier1(repo_root / "configs" / "halo", repo_root=repo_root)
    assert res["tier"] == 1
    assert res["count"] == 7


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

    bad_smpo_dir = tmp_path / "bad_smpo"
    bad_smpo_dir.mkdir()
    (bad_smpo_dir / "run_12b_smpo.yaml").write_text(
        "bf16: true\nloss_type: smooth_lower_bound\nuse_margin_schedule: true\nuse_peft: true\nlora_dropout: 0.0\nlearning_rate: 5.0e-6\nchosen_sft_ratio: 0.25\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="chosen_sft_ratio: 1.0"):
        validate_yaml_configs_tier1(bad_smpo_dir, repo_root=tmp_path)


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
    (src / "tokenizer.json").write_text('{"version": "1.0"}', encoding="utf-8")
    (src / "processor_config.json").write_text(
        json.dumps(
            {
                "image_seq_length": 280,
                "image_processor": {"max_soft_tokens": 280, "image_seq_length": 280},
            }
        ),
        encoding="utf-8",
    )
    tpl_src = tmp_path / "custom.jinja"
    tpl_src.write_text("{{ '<|turn>model\\n<|channel>thought\\n<channel|>' }}", encoding="utf-8")

    staged = prepare_gemma4_highres_dir(src, tmp_path / "staged", max_soft_tokens=1120, chat_template_path=tpl_src)
    assert not (staged / "config.json").is_symlink() and (staged / "config.json").is_file()
    assert not (staged / "tokenizer.json").is_symlink() and (staged / "tokenizer.json").is_file()
    assert (staged / "chat_template.jinja").read_text(encoding="utf-8") == tpl_src.read_text(encoding="utf-8")
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
    assert sft_cfg["save_strategy"] == "steps"
    assert sft_cfg["save_steps"] == 5
    assert smpo_cfg["model_name_or_path"] == str(tmp_path / "sm24" / "sft_merged")
    assert smpo_cfg["chosen_sft_ratio"] == 1.0
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
    smpo_80_cfg = yaml.safe_load(smpo_80.read_text(encoding="utf-8"))
    assert smpo_80_cfg["max_length"] == 4096
    assert smpo_80_cfg["load_in_4bit"] is False
    assert smpo_80_cfg["optim"] == "adamw_torch"


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
    log_ok.write_text("{'loss': '12.47', 'grad_norm': '0.5', 'epoch': '0.2'}\n{'loss': 0.91, 'epoch': '0.4'}\n", encoding="utf-8")
    assert verify_halo_run_log_finite_loss(log_ok) == [12.47, 0.91]
    with pytest.raises(ValueError, match="< 5"):
        verify_halo_run_log_finite_loss(log_ok, min_steps=5)
    log_dup_step = tmp_path / "dup_step.log"
    log_dup_step.write_text(("\n".join(["{'loss': '0.5', 'step': '1'}"] * 5)) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Found 1 training loss metrics"):
        verify_halo_run_log_finite_loss(log_dup_step, min_steps=2)
    log_eval_nan = tmp_path / "eval_nan.log"
    log_eval_nan.write_text("{'loss': '0.5', 'eval_loss': 'nan', 'step': '1'}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Non-finite loss eval_loss"):
        verify_halo_run_log_finite_loss(log_eval_nan)
    cont = verify_stage_continuity_logprobs([-2.0, -1.5], [-1.2, -0.9], [-1.2001, -0.9])
    assert cont["max_step0_diff"] <= 1e-3
    with pytest.raises(ValueError, match="diverged from merged SFT"):
        verify_stage_continuity_logprobs([-2.0, -1.5], [-1.2, -0.9], [-2.0, -1.5])
    with pytest.raises(ValueError, match="indistinguishable from base"):
        verify_stage_continuity_logprobs([-1.2, -0.9], [-1.2, -0.9], [-1.2, -0.9])

    log_nan = tmp_path / "nan.log"
    log_nan.write_text("{'loss': nan}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Non-finite"):
        verify_halo_run_log_finite_loss(log_nan)

    log_single = tmp_path / "single.log"
    log_single.write_text("{'loss': 1.42, 'eval_loss': 0.91}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="< 2"):
        verify_halo_run_log_finite_loss(log_single, min_steps=2)

    log_empty = tmp_path / "empty.log"
    log_empty.write_text("training completed with no metrics\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Found 0 training loss metrics"):
        verify_halo_run_log_finite_loss(log_empty)


def test_fetch_revcity_smoke_sample_range_and_alto_join(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import io, zipfile
    from eahtr.revcity_range import fetch_revcity_smoke_sample

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("apsrevcity%3A1001.jpg", b"FAKEJPGDATA")
        zf.writestr(
            "apsrevcity%3A1001.xml",
            "<alto><Layout><Page><PrintSpace><TextBlock>"
            "<TextLine><String CONTENT='Head'/><String CONTENT='Quarters'/><String CONTENT='Middlebrook'/></TextLine>"
            "<TextLine><String CONTENT='July'/><String CONTENT='the'/><String CONTENT='Sixth'/><String CONTENT='1779'/><String CONTENT='Sir'/></TextLine>"
            "</TextBlock></PrintSpace></Page></Layout></alto>",
        )
    zip_bytes = buf.getvalue()

    class _FakeResp:
        def __init__(self, status: int, data: bytes, cr: str) -> None:
            self.status, self._data, self.headers = status, data, {"Content-Range": cr}
        def __enter__(self) -> "_FakeResp":
            return self
        def __exit__(self, *a: object) -> None:
            return None
        def read(self, n: int = -1) -> bytes:
            return self._data if n < 0 else self._data[:n]

    import email.message, urllib.error

    req_count = {"n": 0, "did_429": False}
    def _fake_urlopen(req, timeout=60):
        req_count["n"] += 1
        if req_count["n"] == 3 and not req_count["did_429"]:
            req_count["did_429"] = True
            hdrs = email.message.Message()
            hdrs["Retry-After"] = "0"
            raise urllib.error.HTTPError(req.full_url, 429, "Too Many Requests", hdrs, None)
        r = req.headers.get("Range", "bytes=0-0").replace("bytes=", "")
        a, b = (int(x) for x in r.split("-"))
        return _FakeResp(206, zip_bytes[a : b + 1], f"bytes {a}-{b}/{len(zip_bytes)}")

    monkeypatch.setattr("eahtr.revcity_range.urllib.request.urlopen", _fake_urlopen)
    recs = fetch_revcity_smoke_sample("https://example.invalid/revcity.zip", tmp_path / "rc_imgs", max_pages=4, tail_bytes=64)
    assert len(recs) == 1
    assert "Head Quarters Middlebrook" in recs[0]["text"]
    assert req_count["did_429"] is True
    assert req_count["n"] <= 6  # coalesced reads avoid 3 separate HTTP calls per member

    call_n = {"n": 0}
    def _bad_second_urlopen(req, timeout=60):
        call_n["n"] += 1
        return _FakeResp(206 if call_n["n"] == 1 else 200, zip_bytes[:1], f"bytes 0-0/{len(zip_bytes)}")

    monkeypatch.setattr("eahtr.revcity_range.urllib.request.urlopen", _bad_second_urlopen)
    with pytest.raises(RuntimeError, match="Expected HTTP 206"):
        fetch_revcity_smoke_sample("https://example.invalid/revcity.zip", tmp_path / "rc_fail", max_pages=4, tail_bytes=64)

    def _wrong_offset_urlopen(req, timeout=60):
        r = req.headers.get("Range", "bytes=0-0").replace("bytes=", "")
        a, b = (int(x) for x in r.split("-"))
        if a > 0:
            return _FakeResp(206, zip_bytes[0 : b - a + 1], f"bytes 0-{b - a}/{len(zip_bytes)}")
        return _FakeResp(206, zip_bytes[a : b + 1], f"bytes {a}-{b}/{len(zip_bytes)}")

    monkeypatch.setattr("eahtr.revcity_range.urllib.request.urlopen", _wrong_offset_urlopen)
    with pytest.raises(RuntimeError, match="Content-Range mismatch"):
        fetch_revcity_smoke_sample("https://example.invalid/revcity.zip", tmp_path / "rc_offset", max_pages=4, tail_bytes=64)


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

    with pytest.raises(ValueError, match="Split disjointness"):
        export_halo_vlm_preference_dataset(pair_train, pair_train, tmp_path / "bad_pref", lambda s: len(s.split()))

    ok_dir = tmp_path / "ok_mlx"
    ok_dir.mkdir()
    (ok_dir / "processor_config.json").write_text(
        json.dumps({"image_seq_length": 1120, "image_processor": {"max_soft_tokens": 1120, "image_seq_length": 1120}}),
        encoding="utf-8",
    )
    g4_tpl = (Path(__file__).resolve().parents[1] / "configs" / "halo" / "templates" / "gemma4_htr_nothink.jinja").read_text(encoding="utf-8")
    (ok_dir / "chat_template.jinja").write_text(g4_tpl, encoding="utf-8")
    assert verify_exported_vlm_dir(ok_dir)["image_seq_length"] == 1120

    bad_dir = tmp_path / "bad_mlx"
    bad_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="processor_config.json"):
        verify_exported_vlm_dir(bad_dir)

    (ok_dir / "chat_template.jinja").write_text("unrelated", encoding="utf-8")
    with pytest.raises(ValueError, match="missing non-thinking"):
        verify_exported_vlm_dir(ok_dir)

    (ok_dir / "chat_template.jinja").write_text("Thinking effort level: 0.9", encoding="utf-8")
    with pytest.raises(ValueError, match="0.9 reasoning-effort"):
        verify_exported_vlm_dir(ok_dir)

    hub_like_tpl = (
        "{%- for m in messages -%}<|turn>{{ m['role'] }}\n"
        "{%- if m['role'] == 'assistant' and add_generation_prompt == false -%}<|channel>thought\n<channel|>{%- endif -%}"
        "{{ m['content'][0]['text'] }}<turn|>\n{%- endfor -%}"
        "{%- if add_generation_prompt -%}<|turn>model\n<|channel>thought\n<channel|>{%- endif -%}"
    )
    (ok_dir / "chat_template.jinja").write_text(hub_like_tpl, encoding="utf-8")
    with pytest.raises(ValueError, match="assistant-turn prefix invariance"):
        verify_exported_vlm_dir(ok_dir)

    img_bad_tpl = g4_tpl.replace("<|image|>", "<WRONG_IMG>")
    (ok_dir / "chat_template.jinja").write_text(img_bad_tpl, encoding="utf-8")
    ref_dir = tmp_path / "ref_tok"
    ref_dir.mkdir()
    (ref_dir / "chat_template.jinja").write_text(g4_tpl, encoding="utf-8")
    fake_tok_mod = types.ModuleType("transformers")
    class _FakeTok:
        eos_token_id = 106
        def __init__(self, path: str) -> None:
            self.tpl = (Path(path) / "chat_template.jinja").read_text(encoding="utf-8")
        def apply_chat_template(self, msgs: object, tokenize: bool = True, add_generation_prompt: bool = False) -> list[int]:
            rendered = "IMG_TOK" if any(p.get("type") == "image" for m in msgs for p in m.get("content", [])) else "TXT"  # type: ignore[union-attr]
            if "<WRONG_IMG>" in self.tpl and rendered == "IMG_TOK":
                return [999]
            return [1, 2]
    class _FakeAutoTok:
        @classmethod
        def from_pretrained(cls, path: str) -> _FakeTok:
            return _FakeTok(path)
    fake_tok_mod.AutoTokenizer = _FakeAutoTok  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers", fake_tok_mod)
    monkeypatch.setattr("eahtr.halo_validate.sys.version_info", sys.version_info)
    with pytest.raises(ValueError, match="diverged from"):
        verify_exported_vlm_dir(ok_dir, reference_dir=ref_dir)


def test_merge_and_eval_cli_baseline_match_and_normalized_weighting(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib.util

    repo_root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("merge_and_eval_halo", repo_root / "scripts" / "merge_and_eval_halo.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    mdl_dir = tmp_path / "mdl"
    mdl_dir.mkdir()
    (mdl_dir / "processor_config.json").write_text(
        json.dumps({"image_seq_length": 1120, "image_processor": {"max_soft_tokens": 1120, "image_seq_length": 1120}}),
        encoding="utf-8",
    )
    g4_tpl = (repo_root / "configs" / "halo" / "templates" / "gemma4_htr_nothink.jinja").read_text(encoding="utf-8")
    (mdl_dir / "chat_template.jinja").write_text(g4_tpl, encoding="utf-8")

    class _FakeImg:
        def convert(self, mode: str) -> "_FakeImg":
            return self
    fake_pil = types.ModuleType("PIL")
    fake_pil_img = types.ModuleType("PIL.Image")
    fake_pil_img.open = lambda p: _FakeImg()  # type: ignore[attr-defined]
    fake_pil.Image = fake_pil_img  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "PIL", fake_pil)
    monkeypatch.setitem(sys.modules, "PIL.Image", fake_pil_img)

    img_a = tmp_path / "a.png"
    img_a.write_bytes(b"png")
    eval_jsonl = tmp_path / "val.jsonl"
    eval_jsonl.write_text(
        "\n".join([
            json.dumps({"record_id": "r1", "document_id": "d1", "split": "val", "image_path": str(img_a), "text": "Dear   Sir"}),
            json.dumps({"record_id": "r2", "document_id": "d2", "split": "val", "image_path": str(img_a), "text": "Humble"}),
        ]) + "\n",
        encoding="utf-8",
    )
    bad_base = tmp_path / "bad_base.json"
    bad_base.write_text(json.dumps([{"record_id": "r1", "strict_cer": 0.5}]), encoding="utf-8")

    class _FakeTensor:
        def __init__(self, shape: tuple[int, ...]) -> None:
            self.shape = shape
        def to(self, dev: object) -> "_FakeTensor":
            return self
        def __getitem__(self, sl: object) -> "_FakeTensor":
            return _FakeTensor((1, 2))

    class _FakeInputs(dict):
        def to(self, dev: object) -> "_FakeInputs":
            return self

    class _FakeProc:
        @classmethod
        def from_pretrained(cls, *a: object) -> "_FakeProc":
            return cls()
        def apply_chat_template(self, msgs: object, **kw: object) -> str:
            return "P"
        def __call__(self, text: object, images: object, return_tensors: str = "pt") -> _FakeInputs:
            return _FakeInputs({"input_ids": _FakeTensor((1, 1))})
        def batch_decode(self, ids: object, skip_special_tokens: bool = True) -> list[str]:
            return ["Dear Sir"]

    class _FakeModel:
        device = "cpu"
        @classmethod
        def from_pretrained(cls, *a: object, **kw: object) -> "_FakeModel":
            return cls()
        def eval(self) -> None:
            return None
        def generate(self, **kw: object) -> _FakeTensor:
            return _FakeTensor((1, 3))

    fake_tf = types.ModuleType("transformers")
    fake_tf.AutoProcessor = _FakeProc  # type: ignore[attr-defined]
    fake_tf.AutoModelForImageTextToText = _FakeModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers", fake_tf)

    fake_torch = types.ModuleType("torch")
    class _NoGrad:
        def __enter__(self) -> None: return None
        def __exit__(self, *a: object) -> None: return None
    fake_torch.bfloat16 = "bf16"  # type: ignore[attr-defined]
    fake_torch.no_grad = lambda: _NoGrad()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    out_json = tmp_path / "out.json"
    monkeypatch.setattr(sys, "argv", ["merge_and_eval_halo.py", "--model-dir", str(mdl_dir), "--eval-jsonl", str(eval_jsonl), "--out-json", str(out_json), "--baseline-predictions", str(bad_base)])
    assert mod.main() == 2

    good_base = tmp_path / "good_base.json"
    good_base.write_text(json.dumps([{"record_id": "r1", "strict_cer": 0.5}, {"record_id": "r2", "strict_cer": 0.5}]), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["merge_and_eval_halo.py", "--model-dir", str(mdl_dir), "--eval-jsonl", str(eval_jsonl), "--out-json", str(out_json), "--baseline-predictions", str(good_base)])
    assert mod.main() == 0
    res = json.loads(out_json.read_text(encoding="utf-8"))
    assert res["scores"]["line_boundary_f1_pct"] == 100.0
    assert res["predictions"][0]["ref_chars"] == len("Dear Sir")
