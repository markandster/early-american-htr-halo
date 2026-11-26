"""Cell 4 builder for public corpus download, cleansing, and split auditing."""

from __future__ import annotations

CELL_4_CORPUS_SOURCE = """
# === Step 1: Download, Clean & Split Rights-Cleared Public Datasets ===
def _file_md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open('rb') as fh:
        for chunk in iter(lambda: fh.read(4 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def download_with_retry(
    url: str,
    dest: Path,
    max_attempts: int = 4,
    min_existing_bytes: int = 1_000_000_000,
    expected_bytes: int | None = None,
    expected_md5: str | None = None,
) -> None:
    if dest.exists() and dest.stat().st_size >= min_existing_bytes:
        size_ok = expected_bytes is None or dest.stat().st_size == expected_bytes
        md5_ok = expected_md5 is None or _file_md5(dest) == expected_md5
        if size_ok and md5_ok:
            print(f'Reusing verified archive: {dest} ({dest.stat().st_size / 1e6:.1f} MB)')
            return
        print(f'Cached archive {dest} failed size/MD5 check; re-downloading...')
        dest.unlink()
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
            if expected_bytes is not None and dest.stat().st_size != expected_bytes:
                raise RuntimeError(f'Download byte count mismatch for {dest}: {dest.stat().st_size} != {expected_bytes}')
            if expected_md5 is not None and _file_md5(dest) != expected_md5:
                raise RuntimeError(f'Download MD5 mismatch for {dest}')
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


def _stratified_smoke_slice(rows_by_split: dict, max_total: int, seed: int) -> dict:
    total = sum(len(v) for v in rows_by_split.values())
    if total <= max_total:
        return rows_by_split
    rng = random.Random(seed)
    out = {'train': [], 'val': [], 'test_public': []}
    min_eval = min(24, max(8, max_total // 10))
    budgets = {
        'val': min(len(rows_by_split['val']), max(min_eval, int(max_total * VAL_FRACTION))),
        'test_public': min(len(rows_by_split['test_public']), max(min_eval, int(max_total * TEST_FRACTION))),
    }
    budgets['train'] = min(len(rows_by_split['train']), max(1, max_total - budgets['val'] - budgets['test_public']))
    for split, k in budgets.items():
        by_group = {}
        for item in rows_by_split[split]:
            gid = item[6].get('document_id') or item[1]
            by_group.setdefault(gid, []).append(item)
        gkeys = list(by_group)
        rng.shuffle(gkeys)
        for gk in gkeys:
            rng.shuffle(by_group[gk])
        picked = []
        while len(picked) < k and any(by_group.values()):
            for gk in gkeys:
                if by_group[gk]:
                    picked.append(by_group[gk].pop())
                    if len(picked) >= k:
                        break
        out[split] = picked
    return out


all_records_by_split = {'train': [], 'val': [], 'test_public': []}
hard_case_rows = []
loc_challenge_rows = []
corpus_report = Counter()
creator_map = {}
if PORTAL_CREATOR_MAP_PATH and Path(PORTAL_CREATOR_MAP_PATH).is_file():
    creator_map = json.loads(Path(PORTAL_CREATOR_MAP_PATH).read_text(encoding='utf-8'))

if INCLUDE_REVCITY:
    rc_dir = WORK_ROOT / 'revcity'
    rc_zip = rc_dir / 'revcity-htr.zip'
    rc_pages = rc_dir / 'pages'
    rc_pages.mkdir(parents=True, exist_ok=True)
    download_with_retry(
        REVCITY_ZENODO_URL,
        rc_zip,
        min_existing_bytes=1_000_000_000,
        expected_bytes=REVCITY_ZENODO_BYTES,
        expected_md5=REVCITY_ZENODO_MD5,
    )

    with zipfile.ZipFile(rc_zip) as zf:
        by_stem = {}
        for member in zf.namelist():
            low = member.lower()
            stem = Path(member).stem
            if low.endswith(('.jpg', '.jpeg', '.png')):
                by_stem.setdefault(stem, {})['img'] = member
            elif low.endswith('.xml'):
                by_stem.setdefault(stem, {})['xml'] = member
        paired_stems = sorted(s for s, d in by_stem.items() if 'img' in d and 'xml' in d)
        print(f'Scanning all {len(paired_stems)} paired (JPG/PNG + ALTO-XML) pages in Zenodo 15776323 (CC-BY-4.0)')

        # Assign volume/item groups (max_id_gap=5) & merge exact/n-gram linked keys before splitting
        item_groups = assign_revcity_item_groups(paired_stems, max_id_gap=5, creator_map=creator_map or None)
        admitted_items = []
        for stem in paired_stems:
            entry = by_stem[stem]
            xml_str = zf.read(entry['xml']).decode('utf-8', errors='replace')
            try:
                lines = parse_alto_xml_lines(xml_str, sort_regions=False)
            except Exception:
                corpus_report['revcity_xml_parse_error'] += 1
                continue
            raw_text = '\\n'.join(lines)
            accepted, reason = filter_corpus_record({'source_name': 'Revolutionary City', 'license': 'CC-BY-4.0', 'text': raw_text})
            cleaned_text = clean_diplomatic_target(raw_text)
            info = item_groups.get(stem, {})
            safe_id = info.get('safe_id') or stem.replace('%3A', '_')
            if not accepted or len(cleaned_text.split()) < 8:
                drop_tag = reason if not accepted else 'dropped_under_8_words'
                corpus_report[f'revcity_{drop_tag}'] += 1
                img_ext = Path(entry['img']).suffix.lower()
                hc_img = rc_pages / f'hard_{safe_id}{img_ext}'
                if not hc_img.exists():
                    hc_img.write_bytes(zf.read(entry['img']))
                is_blank = (len(cleaned_text.strip()) == 0 and len(raw_text.strip()) == 0)
                hard_case_rows.append({
                    'record_id': f'revcity-{safe_id}',
                    'document_id': info.get('document_id') or f'revcity:{safe_id}',
                    'writer_key': info.get('writer_key') or f'unattributed_item:{safe_id}',
                    'image_path': str(hc_img),
                    'reason': drop_tag,
                    'is_blank_page': is_blank,
                    'ground_truth_status': 'blank_page' if is_blank else ('empty_after_hygiene' if not cleaned_text.strip() else 'nonblank'),
                    'raw_text': raw_text,
                    'text': cleaned_text or raw_text,
                })
                continue
            base_key = info['writer_key'] if info.get('hand_attributed') == 'true' else (info.get('document_id') or f'revcity:{safe_id}')
            admitted_items.append((stem, safe_id, entry['img'], xml_str, raw_text, cleaned_text, info, base_key))

        # Union-find transitive closure across 32-gram & >=8-word exact-text matches so related docs never straddle splits
        uf_parent = {it[7]: it[7] for it in admitted_items}
        def _uf_find(k):
            while uf_parent[k] != k:
                uf_parent[k] = uf_parent[uf_parent[k]]
                k = uf_parent[k]
            return k
        def _uf_union(a, b):
            ra, rb = _uf_find(a), _uf_find(b)
            if ra != rb:
                uf_parent[max(ra, rb)] = min(ra, rb)
        seen_sig = {}
        for _, _, _, _, _, c_txt, _, b_key in admitted_items:
            sigs = set(extract_word_ngrams(c_txt, n=32))
            words_norm = ' '.join(c_txt.lower().split())
            if len(words_norm.split()) >= 8:
                sigs.add('exact:' + hashlib.sha256(words_norm.encode('utf-8')).hexdigest())
            for sig in sigs:
                if sig in seen_sig:
                    _uf_union(seen_sig[sig], b_key)
                else:
                    seen_sig[sig] = b_key

        group_stats = {}
        for _, _, _, _, _, _, info, b_key in admitted_items:
            root_g = _uf_find(b_key)
            ent = group_stats.setdefault(root_g, {'pages': 0, 'institution': info.get('institution') or 'revcity'})
            ent['pages'] += 1
        split_map = assign_balanced_group_splits(
            group_stats,
            val_fraction=VAL_FRACTION,
            test_fraction=TEST_FRACTION,
            max_eval_group_pages=MAX_EVAL_GROUP_PAGES,
            seed=SEED,
        )
        split_buckets = {'train': [], 'val': [], 'test_public': []}
        for stem, safe_id, img_m, xml_str, raw_text, cleaned_text, info, base_key in admitted_items:
            split = split_map[_uf_find(base_key)]
            split_buckets[split].append((stem, safe_id, img_m, xml_str, raw_text, cleaned_text, info))

        if REVCITY_MAX_PAGES is not None:
            split_buckets = _stratified_smoke_slice(split_buckets, REVCITY_MAX_PAGES, SEED)

        for split, items in split_buckets.items():
            for stem, safe_id, img_member, xml_str, raw_text, cleaned_text, info in tqdm(
                items, desc=f'Extract RevCity {split}'
            ):
                img_ext = Path(img_member).suffix.lower()
                img_bytes = zf.read(img_member)
                img_sha = hashlib.sha256(img_bytes).hexdigest()
                img_out = rc_pages / f'{safe_id}{img_ext}'
                if not img_out.exists():
                    img_out.write_bytes(img_bytes)
                dhash_hex = compute_dhash(img_out)
                all_records_by_split[split].append({
                    'record_id': f'revcity-{safe_id}',
                    'collection': 'revcity-zenodo-15776323',
                    'document_id': info.get('document_id') or f'revcity:{safe_id}',
                    'writer_key': info.get('writer_key') or f'unattributed_item:{safe_id}',
                    'writer_attributed': info.get('hand_attributed') == 'true',
                    'is_two_page_opening': detect_two_page_opening(xml_str),
                    'image_format': img_ext.lstrip('.'),
                    'image_path': str(img_out),
                    'image_sha256': img_sha,
                    'image_dhash': dhash_hex,
                    'raw_text': raw_text,
                    'text': cleaned_text,
                    'word_count': len(cleaned_text.split()),
                    'license': 'CC-BY-4.0',
                    'split': split,
                })
                corpus_report[f'revcity_admitted_{split}'] += 1
                corpus_report[f'revcity_ext_{img_ext.lstrip(".")}'] += 1

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
                hard_case_rows.append({'record_id': f'loc-btp-{row_idx:05d}', 'reason': reason if not accepted else 'under_8_words', 'raw_text': raw_text})
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

val_audit = audit_split_disjointness(train_rows, val_rows, ngram_size=32, max_dhash_distance=4)
test_audit = audit_split_disjointness(train_rows, test_public_rows, ngram_size=32, max_dhash_distance=4)
val_test_audit = audit_split_disjointness(val_rows, test_public_rows, ngram_size=32, max_dhash_distance=4)
assert val_audit['passed'], f'Train/Val leakage detected: {val_audit}'
assert test_audit['passed'], f'Train/Test leakage detected: {test_audit}'
assert val_test_audit['passed'], f'Val/Test leakage detected: {val_test_audit}'

train_doc_ids = {r['document_id'] for r in train_rows}
hard_case_held_out = []
for hc in hard_case_rows:
    in_train = hc.get('document_id') in train_doc_ids
    hc['panel_role'] = 'diagnostic_overlaps_train' if in_train else 'held_out_from_train'
    if not in_train:
        hard_case_held_out.append(hc)

manifest_dir = OUTPUT_ROOT / 'manifests'
manifest_dir.mkdir(parents=True, exist_ok=True)
all_manifests = {
    **all_records_by_split,
    'hard_case_panel': hard_case_rows,
    'hard_case_held_out': hard_case_held_out,
    'challenge_loc': loc_challenge_rows,
}
for split_name, rows in all_manifests.items():
    (manifest_dir / f'{split_name}.jsonl').write_text(
        '\\n'.join(json.dumps(r, ensure_ascii=False) for r in rows) + ('\\n' if rows else ''),
        encoding='utf-8',
    )

def _split_stats(rows):
    return {
        'pages': len(rows),
        'documents': len({r['document_id'] for r in rows}),
        'attributed_hands': len({r['writer_key'] for r in rows if r.get('writer_attributed')}),
        'unattributed_pages': sum(1 for r in rows if not r.get('writer_attributed')),
        'two_page_openings': sum(1 for r in rows if r.get('is_two_page_opening')),
        'words': sum(r['word_count'] for r in rows),
        'collections': dict(Counter(r['collection'] for r in rows)),
    }

audit_summary = {
    'counts': dict(corpus_report),
    'splits': {k: _split_stats(v) for k, v in all_records_by_split.items()},
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
