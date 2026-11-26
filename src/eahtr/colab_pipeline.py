"""Public-data-only Colab Pro training pipeline & notebook generator facade."""

from __future__ import annotations

import json
from pathlib import Path
import sys

if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "eahtr"

from .candidates import (
    COMPETITION_CANDIDATES,
    DIPLOMATIC_PROMPT,
    LOC_WASHINGTON_ITEM_JSON_URL,
    REVCITY_ZENODO_BYTES,
    REVCITY_ZENODO_MD5,
    REVCITY_ZENODO_URL,
    CompetitionCandidateSpec,
    assign_balanced_group_splits,
    assign_deterministic_split,
    assign_revcity_item_groups,
    make_sham_derangement,
    parse_revcity_stem_id,
    resolve_vision_modules_to_save,
    should_use_4bit,
    validate_competition_candidate,
)
from .colab_cells_data import (
    build_cell_1_markdown,
    build_cell_2_setup,
    build_cell_3_config,
    build_cell_4_corpus,
)
from .colab_cells_eval import (
    build_cell_6_run,
    build_cell_7_rollouts,
    build_cell_8_mlx,
)
from .colab_cells_train import build_cell_5_train_helpers

__all__ = [
    "COMPETITION_CANDIDATES",
    "CompetitionCandidateSpec",
    "DIPLOMATIC_PROMPT",
    "LOC_WASHINGTON_ITEM_JSON_URL",
    "REVCITY_ZENODO_BYTES",
    "REVCITY_ZENODO_MD5",
    "REVCITY_ZENODO_URL",
    "assign_balanced_group_splits",
    "assign_deterministic_split",
    "assign_revcity_item_groups",
    "make_sham_derangement",
    "parse_revcity_stem_id",
    "resolve_vision_modules_to_save",
    "should_use_4bit",
    "validate_competition_candidate",
    "write_competition_colab_notebook",
]


def _code_cell(code: str) -> dict:
    lines = [line + "\n" for line in code.strip("\n").splitlines()]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines,
    }


def _md_cell(md: str) -> dict:
    lines = [line + "\n" for line in md.strip("\n").splitlines()]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": lines,
    }


def write_competition_colab_notebook(out_path: Path) -> Path:
    """Generate a self-contained Colab Pro notebook strictly for public competition datasets."""
    for spec in COMPETITION_CANDIDATES.values():
        validate_competition_candidate(spec)

    cells = [
        _md_cell(build_cell_1_markdown()),
        _code_cell(build_cell_2_setup()),
        _code_cell(build_cell_3_config()),
        _code_cell(build_cell_4_corpus()),
        _code_cell(build_cell_5_train_helpers()),
        _code_cell(build_cell_6_run()),
        _code_cell(build_cell_7_rollouts()),
        _code_cell(build_cell_8_mlx()),
    ]

    nb = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"gpuType": "A100", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".part")
    tmp.write_text(json.dumps(nb, indent=2) + "\n", encoding="utf-8")
    tmp.replace(out_path)
    return out_path


if __name__ == "__main__":
    default_out = Path(__file__).resolve().parents[2] / "notebooks" / "colab_competition_pretrain_sft.ipynb"
    written = write_competition_colab_notebook(default_out)
    print(f"Wrote Colab notebook -> {written}")
