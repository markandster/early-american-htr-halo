"""Unit tests for diplomatic hygiene, license allowlist, ALTO-XML, and split auditing."""

import xml.etree.ElementTree as ET

import pytest

from eahtr.corpus import (
    detect_two_page_opening,
    filter_corpus_record,
    parse_alto_xml_lines,
    parse_page_xml_lines,
    verify_source_license,
)
from eahtr.hygiene import (
    audit_split_disjointness,
    clean_diplomatic_target,
    dhash_hamming_distance,
    has_adjacent_duplicate_lines,
    has_table_leader_dots,
)


def test_clean_diplomatic_target_strips_brackets_carets_and_dash_lines() -> None:
    raw = "ſir, I am^2 [the] bearer\n-\n   of   theſe  [lines] m[_]ung.\n"
    cleaned = clean_diplomatic_target(raw)
    assert cleaned == "sir, I am2 bearer\nof these mung."


def test_duplicate_lines_and_leader_dots_detection() -> None:
    slip = "Camp near Middlebrook\nThis is a repeated line in ground truth\nThis is a repeated line in ground truth\n"
    assert has_adjacent_duplicate_lines(slip) is True
    assert has_adjacent_duplicate_lines("Line one\nLine two\n") is False

    table = "20 Bushels Wheat . . . . . . . . £4 10s"
    assert has_table_leader_dots(table) is True
    assert has_table_leader_dots("20 Bushels Wheat £4 10s") is False


def test_license_allowlist_enforcement() -> None:
    verify_source_license("Revolutionary City", "CC-BY-4.0")
    verify_source_license("LOC By the People", "Public Domain")

    with pytest.raises(ValueError):
        verify_source_license("Restricted Edition", "CC-BY-NC-4.0")
    with pytest.raises(ValueError):
        verify_source_license("Private Archive", "All Rights Reserved")


def test_parse_alto_xml_lines_extracts_ordered_text() -> None:
    alto = """<?xml version="1.0" encoding="UTF-8"?>
    <alto xmlns="http://www.loc.gov/standards/alto/ns-v4#">
      <Layout>
        <Page ID="P1" WIDTH="2000" HEIGHT="1500">
          <PrintSpace HPOS="0" VPOS="0" WIDTH="2000" HEIGHT="1500">
            <TextBlock ID="TB_M" TYPE="marginalia" HPOS="50" VPOS="40" WIDTH="100" HEIGHT="30">
              <TextLine ID="TLM" VPOS="40" HPOS="50"><String CONTENT="Note"/></TextLine>
            </TextBlock>
            <TextBlock ID="TB_R" HPOS="1200" VPOS="100" WIDTH="500" HEIGHT="200">
              <TextLine ID="TLR" VPOS="100" HPOS="1200"><String CONTENT="Right"/><String CONTENT="Col"/></TextLine>
            </TextBlock>
            <TextBlock ID="TB_L" HPOS="150" VPOS="100" WIDTH="500" HEIGHT="200">
              <TextLine ID="TL1" VPOS="100" HPOS="150"><String CONTENT="Head"/><String CONTENT="Quarters"/></TextLine>
              <TextLine ID="TL2" VPOS="140" HPOS="150"><String CONTENT="July"/><String CONTENT="1779"/></TextLine>
            </TextBlock>
          </PrintSpace>
        </Page>
      </Layout>
    </alto>
    """
    assert parse_alto_xml_lines(alto) == ["Head Quarters", "July 1779", "Right Col", "Note"]
    assert detect_two_page_opening(alto) is True


def test_parse_page_xml_lines_extracts_ordered_text() -> None:
    page_xml = """<?xml version="1.0" encoding="UTF-8"?>
    <PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15">
      <Page imageFilename="ledger_001.jpg" imageWidth="1000" imageHeight="1500">
        <TextRegion id="r1">
          <TextLine id="l1">
            <Coords points="0,0 100,0 100,20 0,20"/>
            <Word id="w1"><TextEquiv><Unicode>Col.</Unicode></TextEquiv></Word>
            <Word id="w2"><TextEquiv><Unicode>Washington</Unicode></TextEquiv></Word>
            <TextEquiv><Unicode>Col. Washington observes</Unicode></TextEquiv>
          </TextLine>
          <TextLine id="l2">
            <Word id="w3"><TextEquiv><Unicode>our</Unicode></TextEquiv></Word>
            <Word id="w4"><TextEquiv><Unicode>Public Stores</Unicode></TextEquiv></Word>
          </TextLine>
          <TextLine id="l3">
            <TextEquiv><Unicode>   </Unicode></TextEquiv>
          </TextLine>
        </TextRegion>
      </Page>
    </PcGts>
    """
    assert parse_page_xml_lines(page_xml) == [
        "Col. Washington observes",
        "our Public Stores",
    ]
    assert parse_page_xml_lines("<PcGts><Page><TextRegion><TextLine><TextEquiv><Unicode></Unicode></TextEquiv></TextLine></TextRegion></Page></PcGts>") == []
    with pytest.raises(ET.ParseError):
        parse_page_xml_lines("<PcGts><unclosed>")


def test_filter_corpus_record_gates() -> None:
    ok, reason = filter_corpus_record(
        {
            "source_name": "Revolutionary City",
            "license": "CC-BY-4.0",
            "text": "Head Quarters Middlebrook\nYour obedient servant",
        }
    )
    assert ok is True
    assert reason == "accepted"

    bad_lic, reason_lic = filter_corpus_record(
        {"source_name": "NonCommercial", "license": "CC-BY-NC", "text": "Hello"}
    )
    assert bad_lic is False
    assert reason_lic.startswith("disallowed_license:")

    bad_dots, reason_dots = filter_corpus_record(
        {
            "source_name": "LOC By the People",
            "license": "Public Domain",
            "text": "Corn ............. 14 bushels",
        }
    )
    assert bad_dots is False
    assert reason_dots == "dropped_table_leader_dots"


def test_audit_split_disjointness_catches_all_leak_vectors() -> None:
    shared_text = " ".join(f"word{i}" for i in range(40))
    train = [
        {
            "document_id": "doc-1",
            "writer_key": "writer-a",
            "image_sha256": "aaa111",
            "text": shared_text,
        }
    ]
    clean_eval = [
        {
            "document_id": "doc-2",
            "writer_key": "writer-b",
            "image_sha256": "bbb222",
            "text": " ".join(f"other{i}" for i in range(40)),
        }
    ]
    assert audit_split_disjointness(train, clean_eval)["passed"] is True

    leaky_eval = [
        {
            "document_id": "doc-1",
            "writer_key": "writer-a",
            "image_sha256": "aaa111",
            "text": shared_text,
        }
    ]
    report = audit_split_disjointness(train, leaky_eval)
    assert report["passed"] is False
    assert report["doc_collisions"] == ["doc-1"]
    assert report["writer_collisions"] == ["writer-a"]
    assert report["image_collisions"] == ["aaa111"]
    assert report["ngram_collisions"] > 0
    assert report["exact_text_collisions"] == 1

    short_dup_train = [
        {"document_id": "d-short-1", "writer_key": "w-short-1", "image_sha256": "s1", "text": "Boston July 6 1776 Your humble servant Sir"},
    ]
    short_dup_eval = [
        {"document_id": "d-short-2", "writer_key": "w-short-2", "image_sha256": "s2", "text": "boston   july 6 1776 your humble servant sir"},
    ]
    short_report = audit_split_disjointness(short_dup_train, short_dup_eval, ngram_size=32)
    assert short_report["passed"] is False
    assert short_report["exact_text_collisions"] == 1

    assert dhash_hamming_distance("ffffffffffffffff", "fffffffffffffffe") == 1
    dhash_report = audit_split_disjointness(
        [{"record_id": "t1", "document_id": "dt1", "writer_key": "wt1", "image_sha256": "st1", "image_dhash": "0000000000000000", "text": "alpha beta gamma"}],
        [{"record_id": "e1", "document_id": "de1", "writer_key": "we1", "image_sha256": "se1", "image_dhash": "0000000000000001", "text": "delta epsilon zeta"}],
        max_dhash_distance=4,
    )
    assert dhash_report["passed"] is False
    assert dhash_report["perceptual_image_collisions"] == [("t1", "e1", 1)]
