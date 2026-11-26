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


_SECONDARY_BLOCK_TYPES = ("address", "marginal", "margin", "endorse", "caption", "header", "footer")


def _float_attr(elem: ET.Element, name: str, default: float = 0.0) -> tuple[bool, float]:
    raw = elem.attrib.get(name)
    if raw is None:
        return False, default
    try:
        return True, float(raw)
    except ValueError:
        return False, default


def _textline_text(line_elem: ET.Element) -> str:
    words: list[str] = []
    for child in line_elem:
        if _local_tag(child.tag) == "String":
            content = child.attrib.get("CONTENT", "").strip()
            if content:
                words.append(content)
    return " ".join(words).strip()


def parse_alto_xml_lines(xml_text: str, *, sort_regions: bool = True) -> list[str]:
    """Extract ordered line strings from ALTO-XML, optionally sorting regions by type and geometry."""
    root = ET.fromstring(xml_text)
    blocks = [el for el in root.iter() if _local_tag(el.tag) == "TextBlock"]
    if not blocks:
        lines: list[str] = []
        for elem in root.iter():
            if _local_tag(elem.tag) == "TextLine":
                joined = _textline_text(elem)
                if joined:
                    lines.append(joined)
        return lines

    page_w = 0.0
    for pg in root.iter():
        if _local_tag(pg.tag) == "Page":
            _, page_w = _float_attr(pg, "WIDTH", 0.0)
            break

    block_rows: list[tuple[tuple[int, int, float, int], list[tuple[tuple[float, float, int], str]]]] = []
    for b_idx, blk in enumerate(blocks):
        b_type = str(blk.attrib.get("TYPE") or blk.attrib.get("TAGREFS") or "").lower()
        priority = 1 if any(tok in b_type for tok in _SECONDARY_BLOCK_TYPES) else 0
        has_v, vpos = _float_attr(blk, "VPOS", 0.0)
        has_h, hpos = _float_attr(blk, "HPOS", 0.0)
        col_half = 1 if (page_w > 0 and has_h and hpos >= page_w * 0.52) else 0
        row_bucket = int(vpos // 40) if has_v else b_idx
        sort_key = (priority, col_half, row_bucket, hpos if has_h else float(b_idx))

        blk_lines: list[tuple[tuple[float, float, int], str]] = []
        for l_idx, child in enumerate(blk.iter()):
            if _local_tag(child.tag) == "TextLine":
                txt = _textline_text(child)
                if txt:
                    lv_ok, lv = _float_attr(child, "VPOS", float(l_idx))
                    lh_ok, lh = _float_attr(child, "HPOS", float(l_idx))
                    l_key = (lv if lv_ok else float(l_idx), lh if lh_ok else 0.0, l_idx)
                    blk_lines.append((l_key, txt))
        if blk_lines:
            block_rows.append((sort_key, blk_lines))

    if sort_regions:
        block_rows.sort(key=lambda item: item[0])
    out: list[str] = []
    for _, blk_lines in block_rows:
        ordered = sorted(blk_lines, key=lambda item: item[0]) if sort_regions else blk_lines
        out.extend(txt for _, txt in ordered)
    return out


def detect_two_page_opening(
    xml_text: str,
    *,
    width: int | float | None = None,
    height: int | float | None = None,
) -> bool:
    """Detect whether an ALTO page scan spans a landscape two-page opening."""
    root = ET.fromstring(xml_text)
    pw, ph = width, height
    for pg in root.iter():
        if _local_tag(pg.tag) == "Page":
            ok_w, w_attr = _float_attr(pg, "WIDTH", 0.0)
            ok_h, h_attr = _float_attr(pg, "HEIGHT", 0.0)
            if pw is None and ok_w:
                pw = w_attr
            if ph is None and ok_h:
                ph = h_attr
            break
    if pw and ph and float(ph) > 0 and (float(pw) / float(ph)) >= 1.25:
        return True
    if pw and float(pw) > 0:
        centers: list[float] = []
        for blk in root.iter():
            if _local_tag(blk.tag) == "TextBlock":
                ok_h, hpos = _float_attr(blk, "HPOS", 0.0)
                ok_w, b_w = _float_attr(blk, "WIDTH", 0.0)
                if ok_h:
                    centers.append((hpos + (b_w * 0.5 if ok_w else 0.0)) / float(pw))
        if centers and any(c < 0.42 for c in centers) and any(c > 0.58 for c in centers):
            return True
    return False


def _extract_unicode_from_equiv(equiv_elem: ET.Element) -> str:
    """Return stripped text from a direct <Unicode> child of a <TextEquiv> element."""
    for sub in equiv_elem:
        if _local_tag(sub.tag) == "Unicode" and sub.text:
            cand = sub.text.strip()
            if cand:
                return cand
    return ""


def parse_page_xml_lines(xml_text: str) -> list[str]:
    """Extract ordered line strings from a namespace-agnostic PAGE-XML document."""
    root = ET.fromstring(xml_text)
    lines: list[str] = []
    for elem in root.iter():
        if _local_tag(elem.tag) != "TextLine":
            continue
        # Check direct <TextEquiv>/<Unicode> children first so preceding <Word> children cannot shadow the line text.
        line_text = ""
        for child in elem:
            if _local_tag(child.tag) == "TextEquiv":
                line_text = _extract_unicode_from_equiv(child)
                if line_text:
                    break
        if not line_text:
            word_parts: list[str] = []
            for child in elem:
                if _local_tag(child.tag) != "Word":
                    continue
                for sub in child:
                    if _local_tag(sub.tag) == "TextEquiv":
                        word_txt = _extract_unicode_from_equiv(sub)
                        if word_txt:
                            word_parts.append(word_txt)
                            break
            line_text = " ".join(word_parts).strip()
        if line_text:
            lines.append(line_text)
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
