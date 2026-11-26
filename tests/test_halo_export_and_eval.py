"""Unit tests for Halo dataset export, multimodal parity, rollouts, and frozen evaluation."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import types

import pytest

from eahtr.halo_export import (
    export_halo_vlm_preference_dataset,
    export_halo_vlm_sft_dataset,
    verify_exported_vlm_dir,
)
from eahtr.preferences import build_preference_pairs


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

    pair_train = [{"images": ["/tmp/a.jpg"], "prompt": [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Transcribe."}]}], "chosen": [{"role": "assistant", "content": "Dear Sir"}], "rejected": [{"role": "assistant", "content": "Bad"}], "margin": 0.8, "split": "train"}]
    pair_test = [{"images": ["/tmp/b.jpg"], "prompt": [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Transcribe."}]}], "chosen": [{"role": "assistant", "content": "Humble"}], "rejected": [{"role": "assistant", "content": "Bad"}], "margin": 0.5, "split": "val"}]
    pref_meta = export_halo_vlm_preference_dataset(pair_train, pair_test, tmp_path / "pref_out", lambda s: len(s.split()))
    assert pref_meta["include_margin"] is True
    assert "margin" in saved_dicts[-1][1]["train"].data

    with pytest.raises(ValueError, match="Split disjointness"):
        export_halo_vlm_preference_dataset(pair_train, pair_train, tmp_path / "bad_pref", lambda s: len(s.split()))

    with pytest.raises(ValueError, match="non-train or missing split row"):
        export_halo_vlm_preference_dataset(
            [{**pair_train[0], "split": "test_public"}], pair_test, tmp_path / "bad_split_pref", lambda s: len(s.split())
        )
    with pytest.raises(ValueError, match="non-train or missing split row"):
        export_halo_vlm_preference_dataset(
            [{**pair_train[0], "split": "challenge_loc"}], pair_test, tmp_path / "bad_loc_pref", lambda s: len(s.split())
        )
    with pytest.raises(ValueError, match="non-train or missing split row"):
        export_halo_vlm_preference_dataset(
            [{k: v for k, v in pair_train[0].items() if k != "split"}], pair_test, tmp_path / "missing_split_pref", lambda s: len(s.split())
        )

    pairs = build_preference_pairs(
        [
            {
                "record_id": "p1",
                "document_id": "d1",
                "writer_key": "w1",
                "image_sha256": "s1",
                "image_path": "/tmp/a.jpg",
                "text": "Dear Sir I remain your servant.",
                "completions": ["Bad\n-\n-\n-\n-"],
                "finish_reasons": ["stop"],
                "split": "train",
                "source_fold": "fold_0_of_5",
                "is_out_of_fold": False,
            }
        ]
    )
    assert pairs[0]["source_fold"] == "fold_0_of_5"
    assert pairs[0]["is_out_of_fold"] is False

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
    with pytest.raises(ValueError, match="diverged from"):
        verify_exported_vlm_dir(ok_dir, reference_dir=ref_dir)

    # Pixel-value drift check when AutoProcessor + processor_config.json are present in both dirs
    (ok_dir / "chat_template.jinja").write_text(g4_tpl, encoding="utf-8")
    (ref_dir / "processor_config.json").write_text((ok_dir / "processor_config.json").read_text(encoding="utf-8"), encoding="utf-8")
    fake_pil = types.ModuleType("PIL")
    fake_pil_img = types.ModuleType("PIL.Image")
    fake_pil_img.new = lambda mode, size, color: object()  # type: ignore[attr-defined]
    fake_pil.Image = fake_pil_img  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "PIL", fake_pil)
    monkeypatch.setitem(sys.modules, "PIL.Image", fake_pil_img)

    class _FakeAutoProc:
        def __init__(self, path: str) -> None:
            self.scale = 255.0 if Path(path).name == "ok_mlx" else 1.0

        @classmethod
        def from_pretrained(cls, path: str) -> "_FakeAutoProc":
            return cls(path)

        def apply_chat_template(self, msgs: object, tokenize: bool = False, add_generation_prompt: bool = True) -> str:
            return "P"

        def __call__(self, text: object, images: object, return_tensors: str = "pt") -> dict:
            return {"input_ids": [[1, 2]], "pixel_values": [[self.scale, self.scale]]}

    fake_tok_mod.AutoProcessor = _FakeAutoProc  # type: ignore[attr-defined]
    with pytest.raises(ValueError, match="pixel_values diverged"):
        verify_exported_vlm_dir(ok_dir, reference_dir=ref_dir)

    seen_probe_modes: list[str] = []
    fake_pil_img.frombytes = lambda mode, size, data: types.SimpleNamespace(mode=mode, size=size)  # type: ignore[attr-defined]

    class _ModeSensitiveProc:
        def __init__(self, path: str) -> None:
            self.is_exp = Path(path).name == "ok_mlx"

        @classmethod
        def from_pretrained(cls, path: str) -> "_ModeSensitiveProc":
            return cls(path)

        def apply_chat_template(self, msgs: object, tokenize: bool = False, add_generation_prompt: bool = True) -> str:
            return "P"

        def __call__(self, text: object, images: list, return_tensors: str = "pt") -> dict:
            mode = getattr(images[0], "mode", "RGB")
            seen_probe_modes.append(mode)
            pix = [[0.5, 0.5]] if not (self.is_exp and mode != "RGB") else [[0.0, 0.0]]
            return {"input_ids": [[1, 2]], "pixel_values": pix}

    fake_tok_mod.AutoProcessor = _ModeSensitiveProc  # type: ignore[attr-defined]
    with pytest.raises(ValueError, match="pixel_values diverged"):
        verify_exported_vlm_dir(ok_dir, reference_dir=ref_dir)
    assert "RGB" in seen_probe_modes and "L" in seen_probe_modes


def test_merge_and_eval_cli_guards_and_rollout_manifest_refusal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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
    out_json = tmp_path / "out.json"
    # Pre-load guards must exit 2 before touching transformers/torch:
    monkeypatch.setattr(sys, "argv", ["merge_and_eval_halo.py", "--model-dir", str(mdl_dir), "--eval-jsonl", str(eval_jsonl), "--out-json", str(out_json), "--baseline-predictions", str(tmp_path / "missing.json")])
    assert mod.main() == 2

    dup_base = tmp_path / "dup_base.json"
    dup_base.write_text(json.dumps([{"record_id": "r1", "strict_cer": 0.5}, {"record_id": "r1", "strict_cer": 0.1}, {"record_id": "r2", "strict_cer": 0.5}]), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["merge_and_eval_halo.py", "--model-dir", str(mdl_dir), "--eval-jsonl", str(eval_jsonl), "--out-json", str(out_json), "--baseline-predictions", str(dup_base)])
    assert mod.main() == 2

    bad_base = tmp_path / "bad_base.json"
    bad_base.write_text(json.dumps([{"record_id": "r1", "strict_cer": 0.5}]), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["merge_and_eval_halo.py", "--model-dir", str(mdl_dir), "--eval-jsonl", str(eval_jsonl), "--out-json", str(out_json), "--baseline-predictions", str(bad_base)])
    assert mod.main() == 2

    empty_id_base = tmp_path / "empty_id_base.json"
    empty_id_base.write_text(json.dumps([{"record_id": "", "strict_cer": 0.5}, {"record_id": "r2", "strict_cer": 0.5}]), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["merge_and_eval_halo.py", "--model-dir", str(mdl_dir), "--eval-jsonl", str(eval_jsonl), "--out-json", str(out_json), "--baseline-predictions", str(empty_id_base)])
    assert mod.main() == 2

    nan_base = tmp_path / "nan_base.json"
    nan_base.write_text(json.dumps([{"record_id": "r1", "strict_cer": float("nan")}, {"record_id": "r2", "strict_cer": 0.5}]), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["merge_and_eval_halo.py", "--model-dir", str(mdl_dir), "--eval-jsonl", str(eval_jsonl), "--out-json", str(out_json), "--baseline-predictions", str(nan_base)])
    assert mod.main() == 2

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
        tokenizer = staticmethod(lambda *a, **kw: {"input_ids": [1, 2]})

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
        def __enter__(self) -> None:
            return None

        def __exit__(self, *a: object) -> None:
            return None

    fake_torch.bfloat16 = "bf16"  # type: ignore[attr-defined]
    fake_torch.no_grad = lambda: _NoGrad()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    good_base = tmp_path / "good_base.json"
    good_base.write_text(json.dumps([{"record_id": "r1", "strict_cer": 0.5}, {"record_id": "r2", "strict_cer": 0.5}]), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["merge_and_eval_halo.py", "--model-dir", str(mdl_dir), "--eval-jsonl", str(eval_jsonl), "--out-json", str(out_json), "--baseline-predictions", str(good_base)])
    assert mod.main() == 0
    res = json.loads(out_json.read_text(encoding="utf-8"))
    assert res["scores"]["line_boundary_f1_pct"] == 100.0
    assert res["predictions"][0]["ref_chars"] == len("Dear Sir")

    r_spec = importlib.util.spec_from_file_location("generate_halo_rollouts", repo_root / "scripts" / "generate_halo_rollouts.py")
    assert r_spec and r_spec.loader
    r_mod = importlib.util.module_from_spec(r_spec)
    r_spec.loader.exec_module(r_mod)
    hc_jsonl = tmp_path / "hard_case_held_out.jsonl"
    hc_jsonl.write_text(json.dumps({"record_id": "h1", "image_path": str(img_a), "text": "Dear Sir"}) + "\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["generate_halo_rollouts.py", "--model-dir", str(mdl_dir), "--train-jsonl", str(hc_jsonl), "--out-pref-jsonl", str(tmp_path / "p.jsonl")])
    assert r_mod.main() == 2

    valid_name_jsonl = tmp_path / "train_validated.jsonl"
    valid_name_jsonl.write_text(json.dumps({"record_id": "v1", "document_id": "d1", "writer_key": "w1", "split": "train", "image_path": str(img_a), "text": "Dear Sir"}) + "\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["generate_halo_rollouts.py", "--model-dir", str(mdl_dir), "--train-jsonl", str(valid_name_jsonl), "--out-pref-jsonl", str(tmp_path / "p_ok.jsonl")])
    assert r_mod.main() == 0


def test_full_weight_gemma4_export_parity_fail_closed(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    from eahtr.halo_export import verify_exported_vlm_dir

    staged = tmp_path / "staged_26b"
    exported = tmp_path / "bf16_export"
    staged.mkdir()
    exported.mkdir()

    proc_payload = {"image_seq_length": 1120, "image_processor": {"max_soft_tokens": 1120, "image_seq_length": 1120}}
    g4_tpl = (repo_root / "configs" / "halo" / "templates" / "gemma4_htr_nothink.jinja").read_text(encoding="utf-8")
    (staged / "processor_config.json").write_text(json.dumps(proc_payload), encoding="utf-8")
    (staged / "chat_template.jinja").write_text(g4_tpl, encoding="utf-8")

    # Missing processor_config.json or chat_template.jinja fails closed
    with pytest.raises(FileNotFoundError, match="processor_config"):
        verify_exported_vlm_dir(exported)

    # Reverted default 280 soft tokens fails closed
    (exported / "processor_config.json").write_text(
        json.dumps({"image_seq_length": 280, "image_processor": {"max_soft_tokens": 280, "image_seq_length": 280}}),
        encoding="utf-8",
    )
    (exported / "chat_template.jinja").write_text(g4_tpl, encoding="utf-8")
    with pytest.raises(ValueError, match="1120"):
        verify_exported_vlm_dir(exported)

    # Restoring patched files passes
    (exported / "processor_config.json").write_text(json.dumps(proc_payload), encoding="utf-8")
    res = verify_exported_vlm_dir(exported)
    assert res["image_seq_length"] == 1120
    assert res["chat_template_verified"] is True

    # Removing chat_template.jinja fails closed
    (exported / "chat_template.jinja").unlink()
    with pytest.raises(ValueError, match="missing chat_template"):
        verify_exported_vlm_dir(exported)
