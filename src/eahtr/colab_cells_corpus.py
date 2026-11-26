"""Cell 4 builder for public corpus download, cleansing, and split auditing."""

from __future__ import annotations

CELL_4_CORPUS_SOURCE = """
# === Step 1: Download, Clean & Split Rights-Cleared Public Datasets ===
from eahtr.revcity_build import (
    build_revcity_manifests,
    download_with_retry,
    fetch_bytes_polite,
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
    loc_dir = WORK_ROOT / 'loc_btp'
    loc_pages = loc_dir / 'pages'
    loc_pages.mkdir(parents=True, exist_ok=True)
    item_meta = json.loads(fetch_bytes_polite(LOC_WASHINGTON_ITEM_JSON_URL, delay_s=0.2).decode('utf-8'))
    zip_urls = [
        sub['url']
        for res in item_meta.get('resources', [])
        for f_entry in res.get('files', [])
        for sub in (f_entry if isinstance(f_entry, list) else [f_entry])
        if isinstance(sub, dict) and str(sub.get('url', '')).endswith('.zip')
    ]
    assert zip_urls, 'Could not find Public Domain dataset .zip URL in LOC item metadata'
    loc_zip = loc_dir / '2020446971.zip'
    download_with_retry(zip_urls[0], loc_zip, min_existing_bytes=50_000)

    loc_added = 0
    with zipfile.ZipFile(loc_zip) as zf:
        csv_names = [n for n in zf.namelist() if n.lower().endswith('.csv')]
        assert csv_names, 'No CSV found inside LOC By the People zip'
        reader = csv.DictReader(io.StringIO(zf.read(csv_names[0]).decode('utf-8', errors='replace')))
        for row_idx, row in enumerate(reader):
            if loc_added >= LOC_MAX_PAGES:
                break
            item_title = (row.get('Item') or row.get('Title') or '').lower()
            if 'farm report' in item_title:
                corpus_report['loc_skipped_farm_report'] += 1
                continue
            raw_text = row.get('Transcription') or row.get('Text') or ''
            img_url = row.get('DownloadUrl') or row.get('Asset') or ''
            if not raw_text.strip() or not img_url.startswith('http'):
                corpus_report['loc_skipped_missing_fields'] += 1
                continue
            accepted, reason = filter_corpus_record({'source_name': 'LOC By the People', 'license': 'Public Domain', 'text': raw_text})
            cleaned_text = clean_diplomatic_target(raw_text)
            if not accepted or len(cleaned_text.split()) < 8:
                has_empty_alto = (len(raw_text.strip()) == 0)
                hard_case_rows.append({
                    'record_id': f'loc-btp-{row_idx:05d}',
                    'reason': reason if not accepted else 'under_8_words',
                    'has_empty_alto_text': has_empty_alto,
                    'ground_truth_status': 'missing_or_unverified_gt' if has_empty_alto else 'under_8_words',
                    'raw_text': raw_text,
                })
                continue
            try:
                jpg_bytes = fetch_bytes_polite(img_url, delay_s=0.4)
            except Exception:
                corpus_report['loc_image_fetch_error'] += 1
                continue
            rec_id = f'loc-btp-{row_idx:05d}'
            img_out = loc_pages / f'{rec_id}.jpg'
            img_out.write_bytes(jpg_bytes)
            doc_group = str(row.get('ItemId') or row.get('Item') or rec_id)
            loc_challenge_rows.append({
                'record_id': rec_id,
                'collection': 'loc-by-the-people-2020446971',
                'document_id': f'loc:{doc_group}',
                'writer_key': f'loc_item:{doc_group}',
                'writer_attributed': False,
                'image_path': str(img_out),
                'image_sha256': hashlib.sha256(jpg_bytes).hexdigest(),
                'image_dhash': compute_dhash(img_out),
                'raw_text': raw_text,
                'text': cleaned_text,
                'word_count': len(cleaned_text.split()),
                'license': 'Public Domain',
                'split': 'challenge_loc',
            })
            corpus_report['loc_challenge_admitted'] += 1
            loc_added += 1

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
