"""Offline tests for the standalone LOC By the People challenge manifest builder."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
import zipfile

import pytest

from eahtr.loc_challenge import build_loc_challenge_manifest
from eahtr.revcity_build import resolve_manifest_image_path


def _write_synthetic_loc_zip(zip_path: Path) -> None:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=["Item", "ItemId", "Transcription", "DownloadUrl"])
    writer.writeheader()
    writer.writerow(
        {
            "Item": "Farm Report 1775",
            "ItemId": "farm-1",
            "Transcription": "Corn twenty bushels harvested on Tuesday morning",
            "DownloadUrl": "https://example.invalid/farm.jpg",
        }
    )
    writer.writerow(
        {
            "Item": "General Orders",
            "ItemId": "item-10",
            "Transcription": "Too short",
            "DownloadUrl": "https://example.invalid/short.jpg",
        }
    )
    writer.writerow(
        {
            "Item": "General Orders",
            "ItemId": "item-10",
            "Transcription": "Head Quarters Cambridge ſeventeenth day of Auguſt [crossed] one thouſand ſeven hundred",
            "DownloadUrl": "https://example.invalid/p1.jpg",
        }
    )
    writer.writerow(
        {
            "Item": "Receipt Book",
            "ItemId": "item-22",
            "Transcription": "Received of the Quarter Maſter forty pounds lawful money for forage",
            "DownloadUrl": "https://example.invalid/p2.jpg",
        }
    )
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("washington.csv", buf.getvalue().encode("utf-8"))


def _jpeg_bytes(rgb: tuple[int, int, int]) -> bytes:
    Image = pytest.importorskip("PIL.Image")
    bio = io.BytesIO()
    Image.new("RGB", (64, 64), color=rgb).save(bio, format="JPEG")
    return bio.getvalue()


def test_build_loc_challenge_manifest_filters_caches_and_audits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    zip_path = tmp_path / "loc.zip"
    _write_synthetic_loc_zip(zip_path)
    img_map = {
        "https://example.invalid/p1.jpg": _jpeg_bytes((12, 34, 56)),
        "https://example.invalid/p2.jpg": _jpeg_bytes((90, 80, 70)),
    }
    fetches: list[str] = []

    def _fake_fetch(url: str, **kwargs: object) -> bytes:
        fetches.append(url)
        return img_map[url]

    monkeypatch.setattr("eahtr.loc_challenge.fetch_bytes_polite", _fake_fetch)

    out_dir = tmp_path / "loc_out"
    rev_train = [{"record_id": "rc-1", "document_id": "aps:1", "writer_key": "aps:1", "image_sha256": "deadbeef", "text": "unrelated revcity document text here today"}]
    res = build_loc_challenge_manifest(
        out_dir,
        max_pages=10,
        path_mode="relative",
        zip_path=zip_path,
        revcity_splits={"train": rev_train},
    )

    assert len(fetches) == 2
    assert len(res["challenge_rows"]) == 2
    assert len(res["hard_case_rows"]) == 1
    assert res["counts"]["loc_skipped_farm_report"] == 1
    assert res["counts"]["loc_dropped_under_8_words"] == 1
    assert res["audit"]["cross_corpus_audits"]["train"]["passed"] is True
    assert all("ſ" not in row["text"] and "[" not in row["text"] for row in res["challenge_rows"])

    manifest_path = out_dir / "manifests" / "challenge_loc.jsonl"
    saved_rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(saved_rows) == 2
    resolved = resolve_manifest_image_path(saved_rows[0]["image_path"], manifest_path)
    assert resolved.is_file()

    # Re-running with populated pages/ must reuse cached JPEGs without re-fetching URLs.
    fetches.clear()
    build_loc_challenge_manifest(out_dir, max_pages=10, path_mode="relative", zip_path=zip_path)
    assert fetches == []
