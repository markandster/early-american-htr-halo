"""Tests for the shared Revolutionary City manifest builder and portable path resolution."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import struct
import sys
import types
import zipfile
import zlib

import pytest

from eahtr.revcity_build import build_revcity_manifests, file_md5, resolve_manifest_image_path


def _ensure_pil_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    if importlib.util.find_spec("PIL") is not None:
        return

    class _FakeImg:
        def __init__(self, width: int = 16, height: int = 16) -> None:
            self.size = (width, height)
            self.mode = "RGB"

        def __enter__(self) -> "_FakeImg":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def convert(self, mode: str) -> "_FakeImg":
            return self

        def resize(self, size: tuple[int, int], *args: object, **kwargs: object) -> "_FakeImg":
            return _FakeImg(*size)

        def getdata(self) -> list[int]:
            return [i % 256 for i in range(72)]

    pil_mod = types.ModuleType("PIL")
    img_mod = types.ModuleType("PIL.Image")
    img_mod.open = lambda p: _FakeImg()  # type: ignore[attr-defined]
    pil_mod.Image = img_mod  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "PIL", pil_mod)
    monkeypatch.setitem(sys.modules, "PIL.Image", img_mod)


def _make_valid_png_bytes(seed: int, width: int = 16, height: int = 16) -> bytes:
    """Create minimal valid RGB PNG bytes using stdlib zlib + struct."""
    def _chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    raw = b"".join(
        b"\x00" + bytes(c for x in range(width) for c in ((seed + x * 11 + y * 5) % 256, (seed * 3 + x) % 256, (seed * 7 + y) % 256))
        for y in range(height)
    )
    return b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", zlib.compress(raw)) + _chunk(b"IEND", b"")


def _make_alto(text: str) -> str:
    strings = "".join(f'<String CONTENT="{w}"/>' for w in text.split())
    return (
        '<alto><Layout><Page WIDTH="64" HEIGHT="64"><PrintSpace>'
        f"<TextBlock><TextLine>{strings}</TextLine></TextBlock>"
        "</PrintSpace></Page></Layout></alto>"
    )


def _write_synthetic_revcity_zip(dest: Path) -> Path:
    samples: list[tuple[str, str]] = []
    for i in range(1, 46):
        samples.append(
            (
                f"apsrevcity%3A{1000 + i}",
                f"Large ledger page number {i} written in Philadelphia on July {i} 1776 by clerk alpha beta gamma.",
            )
        )
    for inst, base in [
        ("apsrevcity", 2000),
        ("apsrevcity", 2100),
        ("hsprevcity", 3000),
        ("hsprevcity", 3100),
        ("lcprevcity", 4000),
        ("lcprevcity", 4100),
    ]:
        for p in range(3):
            samples.append(
                (
                    f"{inst}%3A{base + p}",
                    f"Small group {inst} base {base} page {p} unique document letter text with historical spelling and ink.",
                )
            )
    with zipfile.ZipFile(dest, "w") as zf:
        for idx, (stem, txt) in enumerate(samples):
            zf.writestr(f"{stem}.png", _make_valid_png_bytes(idx + 1))
            zf.writestr(f"{stem}.xml", _make_alto(txt))
    return dest


def test_revcity_build_pins_deterministic_split_hash_and_md5_guard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _ensure_pil_stub(monkeypatch)
    zip_path = _write_synthetic_revcity_zip(tmp_path / "revcity-htr.zip")
    zip_md5 = file_md5(zip_path)
    zip_bytes = zip_path.stat().st_size

    out = build_revcity_manifests(
        zip_path,
        tmp_path / "pages",
        tmp_path / "manifests",
        expected_bytes=zip_bytes,
        expected_md5=zip_md5,
        min_eval_pages=1,
        path_mode="relative",
    )
    pairs = sorted(
        (row["record_id"], split_name)
        for split_name in ("train", "val", "test_public")
        for row in out["manifests"][split_name]
    )
    digest = hashlib.sha256(json.dumps(pairs).encode("utf-8")).hexdigest()
    assert len(pairs) == 63
    assert out["splits"]["train"]["pages"] == 57
    assert out["splits"]["val"]["pages"] == 3
    assert out["splits"]["test_public"]["pages"] == 3
    assert digest == "554475ac5b9f804d11e25a68f5923572550f8235a6e418a49f35c2a45c899dd9"

    with pytest.raises(ValueError, match="MD5 mismatch"):
        build_revcity_manifests(
            zip_path,
            tmp_path / "pages_bad",
            tmp_path / "manifests_bad",
            expected_bytes=zip_bytes,
            expected_md5="0" * 32,
            min_eval_pages=1,
        )


def test_revcity_build_transitive_closure_relocation_and_empty_zip_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _ensure_pil_stub(monkeypatch)
    # Inject a 12-word exact-text collision across two separate document runs (1010 vs 2001)
    # plus a 32-gram overlap between another pair (1011 vs 3001) to prove transitive union-find.
    shared_32 = " ".join(f"token{i}" for i in range(34))
    zip_path = _write_synthetic_revcity_zip(tmp_path / "closure.zip")
    with zipfile.ZipFile(zip_path, "a") as zf:
        zf.writestr("apsrevcity%3A2099.png", _make_valid_png_bytes(90))
        zf.writestr(
            "apsrevcity%3A2099.xml",
            _make_alto("Large ledger page number 1 written in Philadelphia on July 1 1776 by clerk alpha beta gamma."),
        )
        zf.writestr("hsprevcity%3A3099.png", _make_valid_png_bytes(91))
        zf.writestr("hsprevcity%3A3099.xml", _make_alto(f"{shared_32} trailing words here"))
        zf.writestr("apsrevcity%3A1046.png", _make_valid_png_bytes(92))
        zf.writestr("apsrevcity%3A1046.xml", _make_alto(f"prefix words {shared_32}"))

    out_dir = tmp_path / "bundle"
    summary = build_revcity_manifests(
        zip_path,
        out_dir / "pages",
        out_dir / "manifests",
        expected_bytes=None,
        expected_md5=None,
        min_eval_pages=1,
        path_mode="relative",
    )
    rec_split = {
        r["record_id"]: sp
        for sp in ("train", "val", "test_public")
        for r in summary["manifests"][sp]
    }
    assert rec_split["revcity-apsrevcity_1001"] == rec_split["revcity-apsrevcity_2099"]
    assert rec_split["revcity-apsrevcity_1046"] == rec_split["revcity-hsprevcity_3099"]

    # Relocate bundle to prove relative manifest image_path portability + --image-root override
    relocated = tmp_path / "relocated_bundle"
    shutil.move(str(out_dir), str(relocated))
    train_manifest = relocated / "manifests" / "train.jsonl"
    first_row = json.loads(train_manifest.read_text(encoding="utf-8").splitlines()[0])
    assert not Path(first_row["image_path"]).is_absolute()
    resolved = resolve_manifest_image_path(first_row["image_path"], train_manifest)
    assert resolved.is_file()

    flat_root = tmp_path / "flat_imgs"
    flat_root.mkdir()
    shutil.copy2(resolved, flat_root / resolved.name)
    assert resolve_manifest_image_path("/stale/content/pages/" + resolved.name, train_manifest, flat_root).is_file()
    with pytest.raises(FileNotFoundError, match="Relative manifest image not found"):
        resolve_manifest_image_path("does_not_exist.png", train_manifest)

    # Empty and corrupt ZIPs must fail closed without writing empty splits
    empty_zip = tmp_path / "empty.zip"
    with zipfile.ZipFile(empty_zip, "w"):
        pass
    with pytest.raises(ValueError, match="No paired image"):
        build_revcity_manifests(empty_zip, tmp_path / "ep", tmp_path / "em", expected_bytes=None, expected_md5=None)
    assert not (tmp_path / "em" / "train.jsonl").exists()


def test_build_revcity_manifests_cli_contract(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _ensure_pil_stub(monkeypatch)
    repo_root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("build_revcity_manifests", repo_root / "scripts" / "build_revcity_manifests.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    zip_path = _write_synthetic_revcity_zip(tmp_path / "cli_revcity.zip")
    out_dir = tmp_path / "cli_out"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_revcity_manifests.py",
            "--zip-path",
            str(zip_path),
            "--out-dir",
            str(out_dir),
            "--skip-md5",
            "--min-eval-pages",
            "1",
        ],
    )
    assert mod.main() == 0
    assert (out_dir / "manifests" / "train.jsonl").is_file()

    # Exporting without --tokenizer fails closed cleanly with exit code 2
    monkeypatch.setattr(
        sys,
        "argv",
        ["build_revcity_manifests.py", "--zip-path", str(zip_path), "--out-dir", str(out_dir), "--export-halo-dir", str(tmp_path / "arr")],
    )
    assert mod.main() == 2

    # Missing archive fails closed with exit code 2
    monkeypatch.setattr(
        sys,
        "argv",
        ["build_revcity_manifests.py", "--zip-path", str(tmp_path / "missing.zip"), "--out-dir", str(out_dir), "--skip-md5"],
    )
    assert mod.main() == 2

