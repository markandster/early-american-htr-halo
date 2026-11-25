"""Rights-cleared corpus gating and ALTO-XML diplomatic extraction."""

from __future__ import annotations

import xml.etree.ElementTree as ET

from .hygiene import clean_diplomatic_target, has_adjacent_duplicate_lines, has_table_leader_dots

ALLOWED_LICENSES: frozenset[str] = frozenset({"CC-BY-4.0", "CC0-1.0", "Public Domain"})


def verify_source_license(source_name: str, license_id: str) -> None:
    """Enforce that only permissively licensed public datasets enter training or evaluation."""
    if license_id not in ALLOWED_LICENSES:
        raise ValueError(
            f"Source {source_name!r} has disallowed license {license_id!r}; "
            f"expected one of {sorted(ALLOWED_LICENSES)}"
        )


def _local_tag(tag: str) -> str:
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def parse_alto_xml_lines(xml_text: str) -> list[str]:
    """Extract ordered line strings from a namespace-agnostic ALTO-XML document."""
    root = ET.fromstring(xml_text)
    lines: list[str] = []
    for elem in root.iter():
        if _local_tag(elem.tag) != "TextLine":
            continue
        words: list[str] = []
        for child in elem:
            if _local_tag(child.tag) == "String":
                content = child.attrib.get("CONTENT", "").strip()
                if content:
                    words.append(content)
        joined = " ".join(words).strip()
        if joined:
            lines.append(joined)
    return lines


def filter_corpus_record(record: dict) -> tuple[bool, str]:
    """Validate license provenance, target cleanliness, and anti-loop record hygiene."""
    source_name = str(record.get("source_name", "unknown"))
    license_id = str(record.get("license", ""))
    try:
        verify_source_license(source_name, license_id)
    except ValueError as exc:
        return False, f"disallowed_license:{exc}"

    raw_text = str(record.get("text", ""))
    if has_table_leader_dots(raw_text):
        return False, "dropped_table_leader_dots"
    if has_adjacent_duplicate_lines(raw_text):
        return False, "dropped_duplicate_line_slip"

    cleaned = clean_diplomatic_target(raw_text)
    if not cleaned:
        return False, "dropped_empty_after_hygiene"

    return True, "accepted"
