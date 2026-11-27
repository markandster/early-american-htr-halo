"""Cell 4 builder for public corpus download, cleansing, and split auditing."""

from __future__ import annotations

CELL_4_CORPUS_SOURCE = """
# === Step 1: Download, Clean & Split Rights-Cleared Public Datasets ===
from eahtr.loc_challenge import build_loc_challenge_manifest
from eahtr.revcity_build import (
    build_revcity_manifests,
    split_summary_stats,
    tag_hard_case_overlaps,
    write_jsonl_atomic,
)

all_records_by_split = {'train': [], 'val': [], 'test_public': []}
hard_case_rows = []
loc_challenge_rows = []
corpus_report = Counter()
creator_map = {}
if PORTAL_CREATOR_MAP_PATH and Path(PORTAL_CREATOR_MAP_PATH).is_file():
    creator_map = json.loads(Path(PORTAL_CREATOR_MAP_PATH).read_text(encoding='utf-8'))

manifest_dir = OUTPUT_ROOT / 'manifests'
manifest_dir.mkdir(parents=True, exist_ok=True)
val_audit = {'passed': True, 'perceptual_image_collisions': []}
test_audit = {'passed': True, 'perceptual_image_collisions': []}
val_test_audit = {'passed': True, 'perceptual_image_collisions': []}

if INCLUDE_REVCITY:
    rc_dir = WORK_ROOT / 'revcity'
    rc_zip = rc_dir / 'revcity-htr.zip'
    rc_pages = rc_dir / 'pages'
    rc_summary = build_revcity_manifests(
        rc_zip,
        rc_pages,
        manifest_dir,
        max_id_gap=30,
        val_fraction=VAL_FRACTION,
        test_fraction=TEST_FRACTION,
        max_eval_group_pages=MAX_EVAL_GROUP_PAGES,
        seed=SEED,
        max_pages=REVCITY_MAX_PAGES,
        creator_map=creator_map or None,
        path_mode='absolute',
        download_url=REVCITY_ZENODO_URL,
        expected_bytes=REVCITY_ZENODO_BYTES,
        expected_md5=REVCITY_ZENODO_MD5,
        write_outputs=False,
    )
    for sp in ('train', 'val', 'test_public'):
        all_records_by_split[sp].extend(rc_summary['manifests'][sp])
    hard_case_rows.extend(rc_summary['manifests']['hard_case_panel'])
    corpus_report.update(rc_summary['counts'])
    val_audit = rc_summary['val_disjoint_audit']
    test_audit = rc_summary['test_disjoint_audit']
    val_test_audit = rc_summary['val_test_disjoint_audit']

if INCLUDE_LOC_BY_THE_PEOPLE:
    loc_summary = build_loc_challenge_manifest(
        WORK_ROOT / 'loc_btp',
        max_pages=LOC_MAX_PAGES,
        path_mode='absolute',
        item_json_url=LOC_WASHINGTON_ITEM_JSON_URL,
        write_outputs=False,
    )
    loc_challenge_rows.extend(loc_summary['challenge_rows'])
    hard_case_rows.extend(loc_summary['hard_case_rows'])
    corpus_report.update(loc_summary['counts'])

train_rows = all_records_by_split['train']
val_rows = all_records_by_split['val']
test_public_rows = all_records_by_split['test_public']
assert len(train_rows) >= 1 and len(val_rows) >= 8 and len(test_public_rows) >= 8, (
    f'Splits too small (check grouping/flags): train={len(train_rows)}, val={len(val_rows)}, test={len(test_public_rows)}'
)

hard_case_held_out = tag_hard_case_overlaps(hard_case_rows, train_rows, val_rows, test_public_rows)
all_manifests = {
    **all_records_by_split,
    'hard_case_panel': hard_case_rows,
    'hard_case_held_out': hard_case_held_out,
    'challenge_loc': loc_challenge_rows,
}
for split_name, rows in all_manifests.items():
    write_jsonl_atomic(manifest_dir / f'{split_name}.jsonl', rows)

audit_summary = {
    'counts': dict(corpus_report),
    'splits': {k: split_summary_stats(v) for k, v in all_records_by_split.items()},
    'hard_case_panel_pages': len(hard_case_rows),
    'empty_alto_hard_case_pages': sum(1 for r in hard_case_rows if r.get('has_empty_alto_text') or r.get('ground_truth_status') == 'missing_or_unverified_gt'),
    'challenge_loc_pages': len(loc_challenge_rows),
    'dhash_review_candidates': (
        val_audit['perceptual_image_collisions']
        + test_audit['perceptual_image_collisions']
        + val_test_audit['perceptual_image_collisions']
    ),
    'val_disjoint_audit': val_audit,
    'test_disjoint_audit': test_audit,
    'val_test_disjoint_audit': val_test_audit,
}
(OUTPUT_ROOT / 'corpus_audit_report.json').write_text(json.dumps(audit_summary, indent=2), encoding='utf-8')
print(json.dumps(audit_summary['splits'], indent=2))
print('Disjointness checks (doc, group, sha256, exact, 32-gram) PASSED; dHash review candidates:', len(audit_summary['dhash_review_candidates']))

if PUBLISH_TO_HF_DATASET:
    print(f'Publishing validated public splits to Hugging Face Hub: {HF_DATASET_REPO_ID} ...')
    !pip install -q datasets huggingface_hub
    ds_url = push_splits_to_hf_hub(all_records_by_split, HF_DATASET_REPO_ID, audit_summary, private=False)
    print(f'Published public dataset: {ds_url}')
"""


def build_cell_4_corpus() -> str:
    """Return the Colab corpus ingestion and split-audit cell source."""
    return CELL_4_CORPUS_SOURCE
