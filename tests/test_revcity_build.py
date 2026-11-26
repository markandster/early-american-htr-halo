"""Tests for the shared Revolutionary City manifest builder and portable path resolution."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
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
