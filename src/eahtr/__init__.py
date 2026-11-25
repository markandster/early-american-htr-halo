"""Early American HTR post-training utilities for Halo."""

from .corpus import ALLOWED_LICENSES, filter_corpus_record, parse_alto_xml_lines, verify_source_license
from .hygiene import (
    audit_split_disjointness,
    clean_diplomatic_target,
    extract_word_ngrams,
    has_adjacent_duplicate_lines,
    has_table_leader_dots,
)
from .metrics import ScoreBreakdown, normalize_lenient, normalize_strict, score_pair
from .rewards import (
    ArchivalRewardConfig,
    ArchivalRewardResult,
    count_editorial_leaks,
    count_modernizations,
    detect_repetition_loop,
    score_group_completions,
    score_transcription,
)

__all__ = [
    "ALLOWED_LICENSES",
    "ArchivalRewardConfig",
    "ArchivalRewardResult",
    "ScoreBreakdown",
    "audit_split_disjointness",
    "clean_diplomatic_target",
    "count_editorial_leaks",
    "count_modernizations",
    "detect_repetition_loop",
    "extract_word_ngrams",
    "filter_corpus_record",
    "has_adjacent_duplicate_lines",
    "has_table_leader_dots",
    "normalize_lenient",
    "normalize_strict",
    "parse_alto_xml_lines",
    "score_group_completions",
    "score_pair",
    "score_transcription",
    "verify_source_license",
]
