"""Unit tests for Inkling/VLM vision-token budgeting, grid resizing, and processor measurement."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import types

import pytest

from eahtr.vision_budget import (
    compute_target_grid_dimensions,
    count_grid_image_tokens,
    encode_image_for_vlm,
    measure_max_vision_tokens,
    parse_cell_size,
    read_image_cell_size,
    resize_image_for_token_budget,
)


def test_read_image_cell_size_and_grid_token_math(tmp_path: Path) -> None:
    proc_dir = tmp_path / "inkling"
    proc_dir.mkdir()
    (proc_dir / "processor_config.json").write_text(
        json.dumps({"image_processor": {"size": {"height": 40, "width": 40}, "patch_size": 20, "merge_size": 2}}),
        encoding="utf-8",
    )
    assert read_image_cell_size(proc_dir) == (40, 40)
    assert read_image_cell_size({"image_processor": {"patch_size": 20, "merge_size": 2}}) == (40, 40)
    assert count_grid_image_tokens(800, 1200, cell_w=40, cell_h=40) == 600
    assert count_grid_image_tokens(801, 1200, cell_w=40, cell_h=40) == 630

    with pytest.raises(FileNotFoundError):
        read_image_cell_size(tmp_path / "missing")
    with pytest.raises(ValueError):
        count_grid_image_tokens(0, 100)


def test_compute_target_grid_dimensions_exact_multiples_and_no_upscale() -> None:
    # Exact multiple downgrade under budget=2000 for 3000x4000 page
    w, h, toks = compute_target_grid_dimensions(3000, 4000, max_image_tokens=2000, cell_w=40, cell_h=40)
    assert w % 40 == 0 and h % 40 == 0
    assert w <= 3000 and h <= 4000
    assert toks <= 2000
    assert count_grid_image_tokens(w, h, cell_w=40, cell_h=40) == toks

    # Small image never upscales beyond original width/height
    sw, sh, stoks = compute_target_grid_dimensions(400, 320, max_image_tokens=2000, cell_w=40, cell_h=40)
    assert (sw, sh, stoks) == (400, 320, 80)

    # Extreme aspect ratio still enforces toks <= max_image_tokens
    ew, eh, etoks = compute_target_grid_dimensions(4000, 40, max_image_tokens=10, cell_w=40, cell_h=40)
    assert ew == 400 and eh == 40 and etoks == 10


def test_encode_image_and_resize_with_mock_pil(monkeypatch: pytest.MonkeyPatch) -> None:
    class _StubImg:
        def __init__(self, size: tuple[int, int], mode: str = "L") -> None:
            self.size = size
            self.mode = mode
            self.closed = False
            self.saved: list[tuple[str, dict]] = []

        def convert(self, mode: str) -> "_StubImg":
            return _StubImg(self.size, mode=mode)

        def resize(self, size: tuple[int, int], resample: object = None) -> "_StubImg":
            return _StubImg(size, mode=self.mode)

        def save(self, buf: object, format: str = "PNG", **kwargs: object) -> None:
            self.saved.append((format, kwargs))
            buf.write(b"IMG")  # type: ignore[attr-defined]

        def close(self) -> None:
            self.closed = True

    fake_pil = types.ModuleType("PIL")
    fake_pil_img = types.ModuleType("PIL.Image")
    fake_pil_img.Resampling = types.SimpleNamespace(LANCZOS="LANCZOS_TAG")  # type: ignore[attr-defined]
    fake_pil.Image = fake_pil_img  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "PIL", fake_pil)
    monkeypatch.setitem(sys.modules, "PIL.Image", fake_pil_img)

    raw = _StubImg((3000, 4000), mode="L")
    resized, toks = resize_image_for_token_budget(raw, max_image_tokens=2000, cell_w=40, cell_h=40)
    assert resized.size[0] % 40 == 0 and resized.size[1] % 40 == 0
    assert toks <= 2000

    blob, ow, oh, etoks = encode_image_for_vlm(raw, max_image_tokens=2000, cell_w=40, cell_h=40, image_format="PNG")
    assert blob == b"IMG" and ow == resized.size[0] and oh == resized.size[1] and etoks == toks


def test_measure_max_vision_tokens_uses_proc_image_token_id_and_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    img_p = tmp_path / "p1.png"
    img_p.write_bytes(b"fake")
    mdl = tmp_path / "inkling"
    mdl.mkdir()
    (mdl / "processor_config.json").write_text(
        json.dumps({"image_processor": {"size": {"height": 40, "width": 40}}}), encoding="utf-8"
    )

    class _FakeImg:
        def __init__(self, size: tuple[int, int]) -> None:
            self.size = size
            self.mode = "RGB"

        def convert(self, mode: str) -> "_FakeImg":
            return self

        def resize(self, size: tuple[int, int], resample: object = None) -> "_FakeImg":
            return _FakeImg(size)

        def close(self) -> None:
            pass

    fake_pil = types.ModuleType("PIL")
    fake_pil_img = types.ModuleType("PIL.Image")
    fake_pil_img.open = lambda p: _FakeImg((240, 200))  # type: ignore[attr-defined]
    fake_pil_img.Resampling = types.SimpleNamespace(LANCZOS="LANCZOS")  # type: ignore[attr-defined]
    fake_pil.Image = fake_pil_img  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "PIL", fake_pil)
    monkeypatch.setitem(sys.modules, "PIL.Image", fake_pil_img)

    class _GoodProc:
        image_token_id = 200054
        tokenizer = types.SimpleNamespace(unk_token_id=0, convert_tokens_to_ids=lambda s: 200054 if s == "<|unused_200054|>" else 0)

        @classmethod
        def from_pretrained(cls, path: str) -> "_GoodProc":
            return cls()

        def apply_chat_template(self, msgs: object, tokenize: bool = False, add_generation_prompt: bool = True) -> str:
            return "prompt"

        def __call__(self, text: object, images: list, return_tensors: str = "pt") -> dict:
            w, h = images[0].size
            n = (w // 40) * (h // 40)
            return {"input_ids": [[10] + [200054] * n + [11]]}

    class _ZeroTokenProc(_GoodProc):
        def __call__(self, text: object, images: list, return_tensors: str = "pt") -> dict:
            return {"input_ids": [[10, 11]]}

    fake_tf = types.ModuleType("transformers")
    fake_tf.AutoProcessor = _GoodProc  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers", fake_tf)

    res = measure_max_vision_tokens([img_p], mdl, max_image_tokens=16)
    assert res["image_token_id"] == 200054
    assert 1 <= res["max_image_tokens"] <= 16
    assert res["max_image_tokens"] == 12

    fake_tf.AutoProcessor = _ZeroTokenProc  # type: ignore[attr-defined]
    with pytest.raises(ValueError, match="0 image tokens"):
        measure_max_vision_tokens([img_p], mdl, max_image_tokens=16)


def test_real_pillow_exact_cell_multiples_and_png_encoding() -> None:
    pytest.importorskip("PIL")
    from PIL import Image as RealImage

    im = RealImage.new("RGB", (3000, 4000), (240, 235, 220))
    resized_real, toks_real = resize_image_for_token_budget(im, max_image_tokens=2000, cell_w=40, cell_h=40)
    assert resized_real.size[0] % 40 == 0 and resized_real.size[1] % 40 == 0
    assert (resized_real.size[0] // 40) * (resized_real.size[1] // 40) == toks_real <= 2000
    blob_real, w_r, h_r, etoks_r = encode_image_for_vlm(
        im, max_image_tokens=2000, cell_w=40, cell_h=40, image_format="PNG"
    )
    assert blob_real.startswith(b"\x89PNG") and (w_r, h_r, etoks_r) == (resized_real.size[0], resized_real.size[1], toks_real)


def test_streaming_export_with_budget_and_cli_image_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("export_halo_datasets", repo_root / "scripts" / "export_halo_datasets.py")
    assert spec and spec.loader
    exp_cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exp_cli)

    img_root = tmp_path / "imgs"
    img_root.mkdir()
    (img_root / "tr.jpg").write_bytes(b"IMG_A")
    (img_root / "va.jpg").write_bytes(b"IMG_B")

    class _FakeImg:
        def __init__(self, size: tuple[int, int] = (200, 200)) -> None:
            self.size = size
            self.mode = "RGB"

        def convert(self, mode: str) -> "_FakeImg":
            return self

        def resize(self, size: tuple[int, int], resample: object = None) -> "_FakeImg":
            return _FakeImg(size)

        def save(self, buf: object, format: str = "PNG", **kwargs: object) -> None:
            buf.write(b"PNG_OUT")  # type: ignore[attr-defined]

        def close(self) -> None:
            pass

    fake_pil = types.ModuleType("PIL")
    fake_pil_img = types.ModuleType("PIL.Image")
    fake_pil_img.open = lambda p: _FakeImg((200, 200))  # type: ignore[attr-defined]
    fake_pil_img.Resampling = types.SimpleNamespace(LANCZOS="LANCZOS")  # type: ignore[attr-defined]
    fake_pil.Image = fake_pil_img  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "PIL", fake_pil)
    monkeypatch.setitem(sys.modules, "PIL.Image", fake_pil_img)

    seen_cache_dirs: list[str] = []
    saved: list[dict] = []

    class FakeDataset:
        def __init__(self, data: dict) -> None:
            self.data = data

        @classmethod
        def from_generator(cls, gen: object, features: object = None, cache_dir: str | None = None, gen_kwargs: dict | None = None) -> "FakeDataset":
            if cache_dir:
                seen_cache_dirs.append(cache_dir)
            cols: dict[str, list] = {}
            for row in gen(**(gen_kwargs or {})):  # type: ignore[operator]
                for k, v in row.items():
                    cols.setdefault(k, []).append(v)
            return cls(cols)

    class FakeDatasetDict(dict):
        def save_to_disk(self, path: str) -> None:
            saved.append(dict(self))

    fake_datasets = types.ModuleType("datasets")
    fake_datasets.Dataset = FakeDataset  # type: ignore[attr-defined]
    fake_datasets.DatasetDict = FakeDatasetDict  # type: ignore[attr-defined]
    fake_datasets.Features = dict  # type: ignore[attr-defined]
    fake_datasets.Image = lambda: "Image"  # type: ignore[attr-defined]
    fake_datasets.Sequence = lambda inner: ("Sequence", inner)  # type: ignore[attr-defined]
    fake_datasets.Value = lambda dtype: ("Value", dtype)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    class _FakeTok:
        @classmethod
        def from_pretrained(cls, path: str) -> "_FakeTok":
            return cls()

        def __call__(self, text: str, add_special_tokens: bool = False) -> dict:
            return {"input_ids": text.split()}

    fake_tf = types.ModuleType("transformers")
    fake_tf.AutoTokenizer = _FakeTok  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers", fake_tf)

    tr_jsonl = tmp_path / "train.jsonl"
    va_jsonl = tmp_path / "val.jsonl"
    tr_jsonl.write_text(json.dumps({"record_id": "t1", "document_id": "d1", "writer_key": "w1", "image_sha256": "s1", "image_path": "rel/tr.jpg", "text": "Dear Sir"}) + "\n", encoding="utf-8")
    va_jsonl.write_text(json.dumps({"record_id": "v1", "document_id": "d2", "writer_key": "w2", "image_sha256": "s2", "image_path": "rel/va.jpg", "text": "Madam"}) + "\n", encoding="utf-8")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_halo_datasets.py",
            "--mode", "sft",
            "--train-jsonl", str(tr_jsonl),
            "--val-jsonl", str(va_jsonl),
            "--out-dir", str(tmp_path / "arrow_out"),
            "--tokenizer", "stub/inkling",
            "--image-root", str(img_root),
            "--max-image-tokens", "16",
            "--cell-size", "40",
        ],
    )
    assert exp_cli.main() == 0
    assert saved[-1]["train"].data["images"][0][0]["bytes"] == b"PNG_OUT"
    assert saved[-1]["train"].data["images"][0][0]["path"] is None
    assert seen_cache_dirs and not Path(seen_cache_dirs[-1]).exists()

    # Preference mode streaming export with --max-image-tokens and parse_cell_size
    assert parse_cell_size("40x40") == (40, 40)
    with pytest.raises(ValueError, match="Invalid --cell-size"):
        parse_cell_size("0")
    pref_tr = tmp_path / "pref_train.jsonl"
    pref_va = tmp_path / "pref_val.jsonl"
    pref_row = {
        "document_id": "d1",
        "writer_key": "w1",
        "split": "train",
        "image_path": "rel/tr.jpg",
        "prompt": [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Transcribe."}]}],
        "chosen": [{"role": "assistant", "content": "Dear Sir"}],
        "rejected": [{"role": "assistant", "content": "Bad"}],
        "margin": 0.7,
    }
    pref_tr.write_text(json.dumps({**pref_row, "record_id": "pt1"}) + "\n", encoding="utf-8")
    pref_va.write_text(json.dumps({**pref_row, "record_id": "pv1", "document_id": "d2", "writer_key": "w2", "image_path": "rel/va.jpg", "split": "val"}) + "\n", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_halo_datasets.py",
            "--mode", "preference",
            "--train-jsonl", str(pref_tr),
            "--val-jsonl", str(pref_va),
            "--out-dir", str(tmp_path / "pref_arrow_out"),
            "--tokenizer", "stub/inkling",
            "--image-root", str(img_root),
            "--max-image-tokens", "16",
            "--cell-size", "40",
        ],
    )
    assert exp_cli.main() == 0
    assert saved[-1]["train"].data["images"][0][0]["bytes"] == b"PNG_OUT"
    assert saved[-1]["train"].data["images"][0][0]["path"] is None

    # --measure-vision-tokens must never lower --vision-tokens below CLI ceiling
    seen_vis_arg: list[int] = []
    orig_export_sft = exp_cli.export_halo_vlm_sft_dataset

    def _spy_export_sft(*args: object, **kwargs: object) -> dict:
        seen_vis_arg.append(int(kwargs["vision_tokens"]))  # type: ignore[arg-type]
        return orig_export_sft(*args, **kwargs)

    monkeypatch.setattr(exp_cli, "measure_max_vision_tokens", lambda *a, **k: {"max_image_tokens": 300})
    monkeypatch.setattr(exp_cli, "export_halo_vlm_sft_dataset", _spy_export_sft)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_halo_datasets.py",
            "--mode", "sft",
            "--train-jsonl", str(tr_jsonl),
            "--val-jsonl", str(va_jsonl),
            "--out-dir", str(tmp_path / "gemma_floor_out"),
            "--tokenizer", "stub/g4",
            "--image-root", str(img_root),
            "--vision-tokens", "1120",
            "--measure-vision-tokens",
        ],
    )
    assert exp_cli.main() == 0
    assert seen_vis_arg == [1120]


def test_real_pillow_and_datasets_resized_arrow_export(tmp_path: Path) -> None:
    pytest.importorskip("PIL")
    pytest.importorskip("datasets")
    from PIL import Image as RealImage
    from datasets import load_from_disk
    from eahtr.halo_export import export_halo_vlm_preference_dataset, export_halo_vlm_sft_dataset

    real_dir = tmp_path / "real_2400x3200"
    real_dir.mkdir()
    src_tr = real_dir / "tr.jpg"
    src_va = real_dir / "va.jpg"
    RealImage.new("RGB", (2400, 3200), (220, 210, 190)).save(src_tr, format="JPEG")
    RealImage.new("RGB", (2400, 3200), (210, 200, 180)).save(src_va, format="JPEG")

    sft_ds_dir = real_dir / "sft_ds"
    sft_meta = export_halo_vlm_sft_dataset(
        [{"record_id": "rt1", "document_id": "rd1", "writer_key": "rw1", "image_sha256": "rs1", "image_path": str(src_tr), "text": "Dear Sir"}],
        [{"record_id": "rv1", "document_id": "rd2", "writer_key": "rw2", "image_sha256": "rs2", "image_path": str(src_va), "text": "Madam"}],
        sft_ds_dir,
        lambda s: len(s.split()),
        max_image_tokens=1120,
    )
    assert "pillow_runtime" in sft_meta and sft_meta["pillow_runtime"]["resample_filter"] == "LANCZOS"

    pref_ds_dir = real_dir / "pref_ds"
    pref_meta = export_halo_vlm_preference_dataset(
        [{
            "record_id": "rpt1", "document_id": "rd1", "writer_key": "rw1", "image_sha256": "rs1",
            "split": "train", "images": [str(src_tr)],
            "prompt": [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Transcribe."}]}],
            "chosen": [{"role": "assistant", "content": "Dear Sir"}],
            "rejected": [{"role": "assistant", "content": "Bad"}],
            "margin": 0.8,
        }],
        [{
            "record_id": "rpv1", "document_id": "rd2", "writer_key": "rw2", "image_sha256": "rs2",
            "split": "val", "images": [str(src_va)],
            "prompt": [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Transcribe."}]}],
            "chosen": [{"role": "assistant", "content": "Madam"}],
            "rejected": [{"role": "assistant", "content": "Bad"}],
            "margin": 0.6,
        }],
        pref_ds_dir,
        lambda s: len(s.split()),
        max_image_tokens=1120,
    )
    assert "pillow_runtime" in pref_meta

    # Remove source JPEGs to prove Arrow datasets embedded resized bytes independently
    src_tr.unlink()
    src_va.unlink()
    assert load_from_disk(str(sft_ds_dir))["train"][0]["images"][0].size == (1120, 1520)
    assert load_from_disk(str(pref_ds_dir))["train"][0]["images"][0].size == (1120, 1520)
