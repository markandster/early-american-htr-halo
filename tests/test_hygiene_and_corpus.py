"""Unit tests for diplomatic hygiene, license allowlist, ALTO-XML, and split auditing."""

import xml.etree.ElementTree as ET

import pytest

from eahtr.corpus import (
    filter_corpus_record,
    parse_alto_xml_lines,
    parse_page_xml_lines,
    verify_source_license,
)
from eahtr.hygiene import (
    audit_split_disjointness,
    clean_diplomatic_target,
    has_adjacent_duplicate_lines,
    has_table_leader_dots,
)


def test_clean_diplomatic_target_strips_brackets_carets_and_dash_lines() -> None:
    raw = "ſir, I am^2 [the] bearer\n-\n   of   theſe  [lines].\n"
    cleaned = clean_diplomatic_target(raw)
    assert cleaned == "sir, I am2 bearer\nof these ."


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
        <Page ID="P1">
          <PrintSpace>
            <TextBlock ID="TB1">
              <TextLine ID="TL1">
                <String CONTENT="Head"/>
                <String CONTENT="Quarters"/>
              </TextLine>
              <TextLine ID="TL2">
                <String CONTENT="July"/>
                <String CONTENT="1779"/>
              </TextLine>
            </TextBlock>
          </PrintSpace>
        </Page>
      </Layout>
    </alto>
    """
    assert parse_alto_xml_lines(alto) == ["Head Quarters", "July 1779"]


def test_parse_page_xml_lines_extracts_ordered_text() -> None:
    page_xml = """<?xml version="1.0" encoding="UTF-8"?>
    <PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15">
      <Page imageFilename="bentham_001.jpg" imageWidth="1000" imageHeight="1500">
        <TextRegion id="r1">
          <TextLine id="l1">
            <Coords points="0,0 100,0 100,20 0,20"/>
            <Word id="w1"><TextEquiv><Unicode>Mr.</Unicode></TextEquiv></Word>
            <Word id="w2"><TextEquiv><Unicode>Bentham</Unicode></TextEquiv></Word>
            <TextEquiv><Unicode>Mr. Bentham observes</Unicode></TextEquiv>
          </TextLine>
          <TextLine id="l2">
            <Word id="w3"><TextEquiv><Unicode>that</Unicode></TextEquiv></Word>
            <Word id="w4"><TextEquiv><Unicode>Public Utility</Unicode></TextEquiv></Word>
          </TextLine>
          <TextLine id="l3">
            <TextEquiv><Unicode>   </Unicode></TextEquiv>
          </TextLine>
        </TextRegion>
      </Page>
    </PcGts>
    """
    assert parse_page_xml_lines(page_xml) == [
        "Mr. Bentham observes",
        "that Public Utility",
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
