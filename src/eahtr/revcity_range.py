"""HTTP Range ZIP reader to sample RevCity pages without loading 9.99 GB in RAM."""

from __future__ import annotations

import io
from pathlib import Path
import re
import time
from typing import Any
import urllib.error
import urllib.request
import zipfile

from .candidates import parse_revcity_stem_id
from .corpus import parse_alto_xml_lines
from .hygiene import clean_diplomatic_target

_CONTENT_RANGE_RE = re.compile(r"^bytes\s+(\d+)-(\d+)/(\d+)$", re.I)


def _fetch_range_with_retry(
    url: str,
    start: int,
    end: int,
    *,
    total_size: int | None = None,
    max_attempts: int = 5,
) -> tuple[bytes, int]:
    """Fetch exact HTTP 206 byte range with strict Content-Range check and 429/5xx backoff."""
    expected_len = end - start + 1
    last_err: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                status = getattr(resp, "status", None)
                if status != 206:
                    raise RuntimeError(f"Expected HTTP 206 for bytes={start}-{end}, got {status}")
                cr = str(resp.headers.get("Content-Range", "")).strip()
                match = _CONTENT_RANGE_RE.match(cr)
                if not match:
                    raise RuntimeError(f"Missing or malformed Content-Range {cr!r} for bytes={start}-{end}")
                cr_start, cr_end, cr_total = int(match.group(1)), int(match.group(2)), int(match.group(3))
                if cr_start != start or cr_end != end or (total_size is not None and cr_total != total_size):
                    raise RuntimeError(
                        f"Content-Range mismatch: got {cr!r}, expected bytes {start}-{end}/{total_size or cr_total}"
                    )
                chunk = resp.read(expected_len + 1)
                if len(chunk) != expected_len:
                    raise RuntimeError(f"Range read length mismatch: got {len(chunk)} bytes, expected {expected_len}")
                return chunk, cr_total
        except urllib.error.HTTPError as exc:
            last_err = exc
            if exc.code not in (429, 502, 503, 504) or attempt == max_attempts:
                raise
            retry_after = exc.headers.get("Retry-After") if exc.headers else None
            delay_s = float(retry_after) if retry_after and retry_after.replace(".", "", 1).isdigit() else min(32.0, 2.0 * (2 ** (attempt - 1)))
            time.sleep(delay_s)
        except (urllib.error.URLError, TimeoutError) as exc:
            last_err = exc
            if attempt == max_attempts:
                raise
            time.sleep(min(32.0, 2.0 * (2 ** (attempt - 1))))
    raise RuntimeError(f"Failed range request bytes={start}-{end} for {url}: {last_err}")


class _HTTPRangeFile(io.RawIOBase):
    """Seekable HTTP Range reader with a cached ZIP tail and member prefetch buffer."""

    def __init__(self, url: str, total_size: int, tail_offset: int, tail_bytes: bytes) -> None:
        self.url, self.total_size, self.pos = url, total_size, 0
        self.tail_offset, self.tail_bytes = tail_offset, tail_bytes
        self._buf_start = -1
        self._buf_bytes = b""

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

    def prefetch(self, start: int, end: int) -> None:
        """Fetch an entry's header+payload span in one HTTP request."""
        clamped_end = min(self.total_size - 1, end)
        if start < 0 or clamped_end < start or start >= self.tail_offset:
            return
        if self._buf_start <= start and clamped_end < self._buf_start + len(self._buf_bytes):
            return
        chunk, _ = _fetch_range_with_retry(self.url, start, clamped_end, total_size=self.total_size)
        self._buf_start = start
        self._buf_bytes = chunk

    def read(self, size: int = -1) -> bytes:
        end = self.total_size if size < 0 else min(self.total_size, self.pos + size)
        if end <= self.pos:
            return b""
        if self.pos >= self.tail_offset:
            chunk = self.tail_bytes[self.pos - self.tail_offset : end - self.tail_offset]
        elif self._buf_start <= self.pos and end <= self._buf_start + len(self._buf_bytes):
            rel = self.pos - self._buf_start
            chunk = self._buf_bytes[rel : rel + (end - self.pos)]
        else:
            chunk, _ = _fetch_range_with_retry(self.url, self.pos, end - 1, total_size=self.total_size)
        self.pos += len(chunk)
        return chunk


def fetch_revcity_smoke_sample(
    url: str, out_dir: Path, *, max_pages: int = 16, tail_bytes: int = 2_500_000
) -> list[dict[str, Any]]:
    """Extract up to max_pages paired RevCity pages via HTTP Range EOCD without RAM bloat."""
    out_dir.mkdir(parents=True, exist_ok=True)
    _, total_size = _fetch_range_with_retry(url, 0, 0)
    tail_off = max(0, total_size - tail_bytes)
    tail, _ = _fetch_range_with_retry(url, tail_off, total_size - 1, total_size=total_size)
    range_fh = _HTTPRangeFile(url, total_size, tail_off, tail)
    with zipfile.ZipFile(range_fh) as zf:
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
            xml_info = zf.getinfo(xml_m)
            range_fh.prefetch(xml_info.header_offset, xml_info.header_offset + 512 + xml_info.compress_size)
            raw_lines = parse_alto_xml_lines(zf.read(xml_m).decode("utf-8", errors="replace"), sort_regions=False)
            text = clean_diplomatic_target("\n".join(raw_lines))
            if len(text.split()) < 8:
                continue
            img_info = zf.getinfo(img_m)
            range_fh.prefetch(img_info.header_offset, img_info.header_offset + 512 + img_info.compress_size)
            img_out = out_dir / f"{stem}{Path(img_m).suffix.lower()}"
            img_out.write_bytes(zf.read(img_m))
            inst, _, _ = parse_revcity_stem_id(stem)
            records.append({"record_id": stem, "document_id": stem, "writer_key": inst, "image_path": str(img_out), "text": text})
            if len(records) >= max_pages:
                break
    return records
