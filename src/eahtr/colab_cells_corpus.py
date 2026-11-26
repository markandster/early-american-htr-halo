"""Cell 4 builder for public corpus download, cleansing, and split auditing."""

from __future__ import annotations

CELL_4_CORPUS_SOURCE = """
# === Step 1: Download, Clean & Split Rights-Cleared Public Datasets ===
def download_with_retry(url: str, dest: Path, max_attempts: int = 4, min_existing_bytes: int = 1_000_000_000) -> None:
    if dest.exists() and dest.stat().st_size >= min_existing_bytes:
        print(f'Reusing cached file: {dest} ({dest.stat().st_size / 1e6:.1f} MB)')
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + '.part')
    last_err = None
    for attempt in range(1, max_attempts + 1):
        try:
            existing = part.stat().st_size if part.exists() else 0
            headers = {'User-Agent': 'early-american-htr-halo/0.1 (public-corpus-research)'}
            if existing > 0:
                headers['Range'] = f'bytes={existing}-'
            req = Request(url, headers=headers)
            with urlopen(req, timeout=120) as resp:
                status = getattr(resp, 'status', 200)
                mode = 'ab' if (existing > 0 and status == 206) else 'wb'
                total = int(resp.headers.get('Content-Length') or 0) + (existing if mode == 'ab' else 0)
                with part.open(mode) as fh, tqdm(
                    total=total or None,
                    initial=existing if mode == 'ab' else 0,
                    unit='B',
                    unit_scale=True,
                    desc=dest.name,
                ) as pbar:
                    while True:
                        chunk = resp.read(4 << 20)
                        if not chunk:
                            break
                        fh.write(chunk)
                        pbar.update(len(chunk))
            part.replace(dest)
            return
        except (HTTPError, URLError, OSError, TimeoutError) as exc:
            last_err = exc
            wait_s = min(60, 5 * attempt)
            print(f'[Attempt {attempt}/{max_attempts}] Download error ({type(exc).__name__}: {exc}); retrying in {wait_s}s...')
            time.sleep(wait_s)
    raise RuntimeError(f'Failed to download {url}: {last_err}')


def fetch_bytes_polite(url: str, max_attempts: int = 3, delay_s: float = 0.5) -> bytes:
    last_err = None
    for attempt in range(1, max_attempts + 1):
        try:
            time.sleep(delay_s)
            req = Request(url, headers={'User-Agent': 'early-american-htr-halo/0.1 (public-corpus-research)'})
            with urlopen(req, timeout=60) as resp:
                return resp.read()
        except (HTTPError, URLError, OSError, TimeoutError) as exc:
            last_err = exc
            time.sleep(min(30, 3 * attempt))
    raise RuntimeError(f'Failed GET {url}: {last_err}')


all_records_by_split = {'train': [], 'val': [], 'test_public': []}
corpus_report = Counter()

if INCLUDE_REVCITY:
    rc_dir = WORK_ROOT / 'revcity'
    rc_zip = rc_dir / 'revcity-htr.zip'
    rc_pages = rc_dir / 'pages'
    rc_pages.mkdir(parents=True, exist_ok=True)
    download_with_retry(REVCITY_ZENODO_URL, rc_zip, min_existing_bytes=1_000_000_000)

    with zipfile.ZipFile(rc_zip) as zf:
        names = set(zf.namelist())
        stems = sorted(n[:-4] for n in names if n.endswith('.xml') and (n[:-4] + '.jpg') in names)
        if REVCITY_MAX_PAGES is not None:
            stems = stems[:REVCITY_MAX_PAGES]
        print(f'Scanning {len(stems)} paired (JPG, ALTO-XML) pages in Zenodo 15776323 (CC-BY-4.0)')

        for stem in tqdm(stems, desc='RevCity ALTO + Hygiene'):
            safe_id = Path(stem).name.replace('%3A', '_')
            try:
                lines = parse_alto_xml_lines(zf.read(stem + '.xml').decode('utf-8', errors='replace'))
            except Exception:
                corpus_report['revcity_xml_parse_error'] += 1
                continue
            raw_text = '\\n'.join(lines)
            rec_probe = {'source_name': 'Revolutionary City', 'license': 'CC-BY-4.0', 'text': raw_text}
            accepted, reason = filter_corpus_record(rec_probe)
            if not accepted:
                corpus_report[f'revcity_{reason}'] += 1
                continue

            cleaned_text = clean_diplomatic_target(raw_text)
            if len(cleaned_text.split()) < 3:
                corpus_report['revcity_dropped_too_short'] += 1
                continue

            jpg_bytes = zf.read(stem + '.jpg')
            img_sha = hashlib.sha256(jpg_bytes).hexdigest()
            img_out = rc_pages / f'{safe_id}.jpg'
            if not img_out.exists():
                img_out.write_bytes(jpg_bytes)

            # Group multi-page letters together by stripping trailing -NNN
            group_key = re.sub(r'[-_]\\d+$', '', safe_id)
            split = assign_deterministic_split(
                f'revcity:{group_key}', val_fraction=VAL_FRACTION, test_fraction=TEST_FRACTION
            )
            all_records_by_split[split].append({
                'record_id': f'revcity-{safe_id}',
                'collection': 'revcity-zenodo-15776323',
                'document_id': f'revcity:{group_key}',
                'writer_key': f'revcity_series:{group_key}',
                'image_path': str(img_out),
                'image_sha256': img_sha,
                'text': cleaned_text,
                'word_count': len(cleaned_text.split()),
                'license': 'CC-BY-4.0',
                'split': split,
            })
            corpus_report[f'revcity_admitted_{split}'] += 1

if INCLUDE_LOC_BY_THE_PEOPLE:
    loc_dir = WORK_ROOT / 'loc_btp'
    loc_pages = loc_dir / 'pages'
    loc_pages.mkdir(parents=True, exist_ok=True)
    item_meta = json.loads(fetch_bytes_polite(LOC_WASHINGTON_ITEM_JSON_URL, delay_s=0.2).decode('utf-8'))
    zip_urls = []
    for res in item_meta.get('resources', []):
        for f_entry in res.get('files', []):
            for sub in (f_entry if isinstance(f_entry, list) else [f_entry]):
                u = sub.get('url', '') if isinstance(sub, dict) else ''
                if u.endswith('.zip'):
                    zip_urls.append(u)
    assert zip_urls, 'Could not find Public Domain dataset .zip URL in LOC item metadata'
    loc_zip = loc_dir / '2020446971.zip'
    download_with_retry(zip_urls[0], loc_zip, min_existing_bytes=50_000)

    loc_added = 0
    with zipfile.ZipFile(loc_zip) as zf:
        csv_names = [n for n in zf.namelist() if n.lower().endswith('.csv')]
        assert csv_names, 'No CSV found inside LOC By the People zip'
        csv_text = zf.read(csv_names[0]).decode('utf-8', errors='replace')
        reader = csv.DictReader(io.StringIO(csv_text))
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
            accepted, reason = filter_corpus_record({
                'source_name': 'LOC By the People',
                'license': 'Public Domain',
                'text': raw_text,
            })
            if not accepted:
                corpus_report[f'loc_{reason}'] += 1
                continue
            cleaned_text = clean_diplomatic_target(raw_text)
            if len(cleaned_text.split()) < 5:
                corpus_report['loc_dropped_too_short'] += 1
                continue
            try:
                jpg_bytes = fetch_bytes_polite(img_url, delay_s=0.4)
            except Exception:
                corpus_report['loc_image_fetch_error'] += 1
                continue
            img_sha = hashlib.sha256(jpg_bytes).hexdigest()
            rec_id = f'loc-btp-{row_idx:05d}'
            img_out = loc_pages / f'{rec_id}.jpg'
            img_out.write_bytes(jpg_bytes)
            doc_group = str(row.get('ItemId') or row.get('Item') or rec_id)
            split = assign_deterministic_split(
                f'loc:{doc_group}', val_fraction=VAL_FRACTION, test_fraction=TEST_FRACTION
            )
            all_records_by_split[split].append({
                'record_id': rec_id,
                'collection': 'loc-by-the-people-2020446971',
                'document_id': f'loc:{doc_group}',
                'writer_key': f'loc_item:{doc_group}',
                'image_path': str(img_out),
                'image_sha256': img_sha,
                'text': cleaned_text,
                'word_count': len(cleaned_text.split()),
                'license': 'Public Domain',
                'split': split,
            })
            corpus_report[f'loc_admitted_{split}'] += 1
            loc_added += 1

if INCLUDE_BENTHAM:
    bentham_dir = WORK_ROOT / 'bentham'
    bentham_tgz = bentham_dir / 'icfhr2016.tgz'
    bentham_pages = bentham_dir / 'pages'
    bentham_pages.mkdir(parents=True, exist_ok=True)
    download_with_retry(BENTHAM_ICFHR2016_ZENODO_URL, bentham_tgz, min_existing_bytes=100_000_000)

    bentham_added = 0
    with tarfile.open(bentham_tgz, 'r:gz') as tf:
        members = [m for m in tf.getmembers() if m.isfile()]
        by_stem = {}
        for m in members:
            ext = Path(m.name).suffix.lower()
            stem = Path(m.name).stem.lower()
            if ext in {'.jpg', '.jpeg', '.png'}:
                by_stem.setdefault(stem, {})['img'] = m
            elif ext == '.xml':
                by_stem.setdefault(stem, {})['xml'] = m
            elif ext == '.txt':
                by_stem.setdefault(stem, {})['txt'] = m

        for stem in sorted(by_stem):
            if bentham_added >= BENTHAM_MAX_PAGES:
                break
            entry = by_stem[stem]
            if 'img' not in entry or ('xml' not in entry and 'txt' not in entry):
                continue
            raw_text = ''
            if 'xml' in entry:
                xml_str = tf.extractfile(entry['xml']).read().decode('utf-8', errors='replace')
                try:
                    lines = parse_page_xml_lines(xml_str)
                    if not lines:
                        lines = parse_alto_xml_lines(xml_str)
                    raw_text = '\\n'.join(lines)
                except Exception:
                    corpus_report['bentham_xml_parse_error'] += 1
                    continue
            else:
                raw_text = tf.extractfile(entry['txt']).read().decode('utf-8', errors='replace')

            accepted, reason = filter_corpus_record({
                'source_name': 'Bentham ICFHR2016',
                'license': 'CC-BY-4.0',
                'text': raw_text,
            })
            if not accepted:
                corpus_report[f'bentham_{reason}'] += 1
                continue
            cleaned_text = clean_diplomatic_target(raw_text)
            if len(cleaned_text.split()) < 3:
                corpus_report['bentham_dropped_too_short'] += 1
                continue

            img_bytes = tf.extractfile(entry['img']).read()
            img_sha = hashlib.sha256(img_bytes).hexdigest()
            img_ext = Path(entry['img'].name).suffix.lower()
            safe_stem = re.sub(r'[^a-zA-Z0-9._-]+', '_', stem)
            img_out = bentham_pages / f'{safe_stem}{img_ext}'
            if not img_out.exists():
                img_out.write_bytes(img_bytes)

            group_key = re.sub(r'[-_]\\d+$', '', safe_stem)
            # Keep Bentham ICFHR-2016 strictly in train so shared secretarial hands never straddle test_public.
            split = 'train'
            all_records_by_split[split].append({
                'record_id': f'bentham-{safe_stem}',
                'collection': 'bentham-icfhr2016-zenodo-1164045',
                'document_id': f'bentham:{group_key}',
                'writer_key': 'bentham_collection:icfhr2016',
                'image_path': str(img_out),
                'image_sha256': img_sha,
                'text': cleaned_text,
                'word_count': len(cleaned_text.split()),
                'license': 'CC-BY-4.0',
                'split': split,
            })
            corpus_report[f'bentham_admitted_{split}'] += 1
            bentham_added += 1

train_rows = all_records_by_split['train']
val_rows = all_records_by_split['val']
test_public_rows = all_records_by_split['test_public']
assert train_rows and val_rows and test_public_rows, (
    f'All splits must be non-empty: train={len(train_rows)}, val={len(val_rows)}, test={len(test_public_rows)}'
)

val_audit = audit_split_disjointness(train_rows, val_rows, ngram_size=32)
test_audit = audit_split_disjointness(train_rows, test_public_rows, ngram_size=32)
val_test_audit = audit_split_disjointness(val_rows, test_public_rows, ngram_size=32)
assert val_audit['passed'], f'Train/Val leakage detected: {val_audit}'
assert test_audit['passed'], f'Train/Test leakage detected: {test_audit}'
assert val_test_audit['passed'], f'Val/Test leakage detected: {val_test_audit}'

manifest_dir = OUTPUT_ROOT / 'manifests'
manifest_dir.mkdir(parents=True, exist_ok=True)
for split_name, rows in all_records_by_split.items():
    (manifest_dir / f'{split_name}.jsonl').write_text(
        '\\n'.join(json.dumps(r, ensure_ascii=False) for r in rows) + '\\n',
        encoding='utf-8',
    )

audit_summary = {
    'counts': dict(corpus_report),
    'train_sides': len(train_rows),
    'train_words': sum(r['word_count'] for r in train_rows),
    'val_sides': len(val_rows),
    'val_words': sum(r['word_count'] for r in val_rows),
    'test_public_sides': len(test_public_rows),
    'test_public_words': sum(r['word_count'] for r in test_public_rows),
    'val_disjoint_audit': val_audit,
    'test_disjoint_audit': test_audit,
    'val_test_disjoint_audit': val_test_audit,
}
(OUTPUT_ROOT / 'corpus_audit_report.json').write_text(json.dumps(audit_summary, indent=2), encoding='utf-8')
print(json.dumps({k: v for k, v in audit_summary.items() if 'audit' not in k}, indent=2))
print('Disjointness checks PASSED: train vs val, train vs test_public, and val vs test_public.')

if PUBLISH_TO_HF_DATASET:
    print(f'Publishing validated public splits to Hugging Face Hub: {HF_DATASET_REPO_ID} ...')
    !pip install -q datasets huggingface_hub
    ds_url = push_splits_to_hf_hub(all_records_by_split, HF_DATASET_REPO_ID, audit_summary, private=False)
    print(f'Published public dataset: {ds_url}')
"""


def build_cell_4_corpus() -> str:
    """Return the Colab corpus ingestion and split-audit cell source."""
    return CELL_4_CORPUS_SOURCE
