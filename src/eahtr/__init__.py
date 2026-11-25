"""Early American HTR post-training utilities for Halo."""

from .colab_pipeline import (
    BENTHAM_ICFHR2016_ZENODO_URL,
    COMPETITION_CANDIDATES,
    DIPLOMATIC_PROMPT,
    CompetitionCandidateSpec,
    assign_deterministic_split,
    make_sham_derangement,
    resolve_vision_modules_to_save,
    should_use_4bit,
    validate_competition_candidate,
    write_competition_colab_notebook,
)
from .corpus import (
    ALLOWED_LICENSES,
    filter_corpus_record,
    parse_alto_xml_lines,
    parse_page_xml_lines,
    verify_source_license,
)
from .hf_dataset import (
    build_hf_dataset_card,
    push_splits_to_hf_hub,
    validate_records_for_public_publish,
)
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
    "BENTHAM_ICFHR2016_ZENODO_URL",
    "COMPETITION_CANDIDATES",
    "CompetitionCandidateSpec",
    "DIPLOMATIC_PROMPT",
    "ScoreBreakdown",
    "assign_deterministic_split",
    "audit_split_disjointness",
    "build_hf_dataset_card",
    "clean_diplomatic_target",
    "count_editorial_leaks",
    "count_modernizations",
    "detect_repetition_loop",
    "extract_word_ngrams",
    "filter_corpus_record",
    "has_adjacent_duplicate_lines",
    "has_table_leader_dots",
    "make_sham_derangement",
    "normalize_lenient",
    "normalize_strict",
    "parse_alto_xml_lines",
    "parse_page_xml_lines",
    "push_splits_to_hf_hub",
    "resolve_vision_modules_to_save",
    "score_group_completions",
    "score_pair",
    "score_transcription",
    "should_use_4bit",
    "validate_competition_candidate",
    "validate_records_for_public_publish",
    "verify_source_license",
    "write_competition_colab_notebook",
]
