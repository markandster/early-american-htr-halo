"""HTTP Range ZIP reader to sample RevCity pages without loading 9.99 GB in RAM."""

from __future__ import annotations

import io
from pathlib import Path
import re
from typing import Any
import urllib.request
import zipfile

from .candidates import parse_revcity_stem_id
from .corpus import parse_alto_xml_lines
from .hygiene import clean_diplomatic_target


class _HTTPRangeFile(io.RawIOBase):
    """Seekable HTTP Range reader with a cached ZIP tail."""

    def __init__(self, url: str, total_size: int, tail_offset: int, tail_bytes: bytes) -> None:
        self.url, self.total_size, self.pos = url, total_size, 0
        self.tail_offset, self.tail_bytes = tail_offset, tail_bytes

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self.pos = offset
        elif whence == io.SEEK_CUR:
            self.pos += offset
        elif whence == io.SEEK_END:
            self.pos = self.total_size + offset
        return self.pos

    def read(self, size: int = -1) -> bytes:
        end = self.total_size if size < 0 else min(self.total_size, self.pos + size)
        if end <= self.pos:
            return b""
        if self.pos >= self.tail_offset:
            chunk = self.tail_bytes[self.pos - self.tail_offset : end - self.tail_offset]
        else:
            req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end - 1}"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                chunk = resp.read()
        self.pos += len(chunk)
        return chunk


def fetch_revcity_smoke_sample(
    url: str, out_dir: Path, *, max_pages: int = 16, tail_bytes: int = 2_500_000
) -> list[dict[str, Any]]:
    """Extract up to max_pages paired RevCity pages via HTTP Range EOCD without RAM bloat."""
    out_dir.mkdir(parents=True, exist_ok=True)
    probe = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
    with urllib.request.urlopen(probe, timeout=60) as resp:
        cr = resp.headers.get("Content-Range", "")
        match = re.search(r"/(\d+)$", cr)
        if resp.status != 206 or not match:
            raise RuntimeError(f"HTTP Range unsupported for {url} (status={resp.status}, Content-Range={cr!r})")
        total_size = int(match.group(1))
    tail_off = max(0, total_size - tail_bytes)
    with urllib.request.urlopen(
        urllib.request.Request(url, headers={"Range": f"bytes={tail_off}-{total_size - 1}"}), timeout=60
    ) as resp:
        tail = resp.read()
    with zipfile.ZipFile(_HTTPRangeFile(url, total_size, tail_off, tail)) as zf:
        by_stem: dict[str, dict[str, str]] = {}
        for name in zf.namelist():
            low = name.lower()
            stem = Path(name).stem
            if low.endswith((".jpg", ".jpeg", ".png")):
                by_stem.setdefault(stem, {})["img"] = name
            elif low.endswith(".xml"):
                by_stem.setdefault(stem, {})["xml"] = name
        paired = [(s, d["img"], d["xml"]) for s, d in by_stem.items() if "img" in d and "xml" in d]
        paired.sort(key=lambda row: parse_revcity_stem_id(row[0])[0])
        records: list[dict[str, Any]] = []
        seen_insts: set[str] = set()
        ordered = [r for r in paired if (i := parse_revcity_stem_id(r[0])[0]) not in seen_insts and not seen_insts.add(i)] + [r for r in paired]
        seen_stems: set[str] = set()
        for stem, img_m, xml_m in ordered:
            if stem in seen_stems:
                continue
            seen_stems.add(stem)
            text = clean_diplomatic_target(parse_alto_xml_lines(zf.read(xml_m).decode("utf-8", errors="replace")))
            if len(text.split()) < 8:
                continue
            img_out = out_dir / f"{stem}{Path(img_m).suffix.lower()}"
            img_out.write_bytes(zf.read(img_m))
            inst, _, _ = parse_revcity_stem_id(stem)
            records.append({"record_id": stem, "document_id": stem, "writer_key": inst, "image_path": str(img_out), "text": text})
            if len(records) >= max_pages:
                break
    return records
