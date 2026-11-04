"""Combine inspected Group 5 logs by dataset/method/seed, preserving conflicts."""
import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZipFile

CODE = Path(__file__).resolve().parents[1]
LOCAL = CODE / 'result' / 'all_methods'
ARCHIVE = CODE / 'result' / 'MUSDP_Group5_Updated_Result.zip'
OUT = CODE / 'result' / 'all_methods_combined'
MEASURES = ('G_mean', 'AUC', 'Popt', 'Recall20')


def combine():
    settings = json.loads((LOCAL / 'settings.json').read_text())
    datasets, methods = settings['datasets'], settings['methods']
    local = [json.loads(s) for s in (LOCAL / 'rounds.jsonl').read_text().splitlines() if s.strip()]
    with ZipFile(ARCHIVE) as archive:
        assert archive.testzip() is None, 'ZIP integrity failure'
        remote = [json.loads(s) for s in archive.read('result/rounds.jsonl').decode().splitlines() if s.strip()]
        completion = json.loads(archive.read('result/dataset_completion.json'))
        archive_members = {i.filename: hashlib.sha256(archive.read(i.filename)).hexdigest()
                           for i in archive.infolist() if not i.is_dir()}
        # Independently verify supplied full-round medians against the raw log.
        for measure in MEASURES:
            raw = defaultdict(list)
            for row in remote:
                raw[row['dataset'], row['method']].append(row[measure])
            table = csv.DictReader(archive.read(f'result/{measure}_median_100rounds.csv').decode('utf-8-sig').splitlines())
            for row in table:
                for method, value in row.items():
                    if method != 'Dataset' and value and value.lower() != 'nan':
                        assert len(raw[row['Dataset'], method]) == 100
                        assert math.isclose(float(value), statistics.median(raw[row['Dataset'], method]), abs_tol=1e-12)

    def index(rows):
        result = {}
        for row in rows:
            key = (row['dataset'], row['method'], row['seed'])
            assert key[0] in datasets and key[1] in methods and 0 <= key[2] < 100
            assert key not in result, f'Duplicate source seed: {key}'
            assert all(math.isfinite(row[m]) and 0 <= row[m] <= 1 for m in MEASURES)
            result[key] = row
        return result

    local_index, remote_index = index(local), index(remote)
    overlap = local_index.keys() & remote_index.keys()
    conflicts = []
    for key in sorted(overlap):
        differences = {m: {'local': local_index[key][m], 'archive': remote_index[key][m]}
                       for m in MEASURES if not math.isclose(local_index[key][m], remote_index[key][m], rel_tol=1e-10, abs_tol=1e-12)}
        if differences:
            conflicts.append({'dataset': key[0], 'method': key[1], 'seed': key[2],
                              'differences': differences, 'selected_source': 'local',
                              'local_record': local_index[key], 'archive_record': remote_index[key]})
    merged = dict(remote_index)
    merged.update(local_index)
    ordered = sorted(merged, key=lambda k: (datasets.index(k[0]), k[2], methods.index(k[1])))
    values, seeds = defaultdict(list), defaultdict(set)
    normalized, provenance = [], []
    for key in ordered:
        row = dict(merged[key])
        provenance.append({'dataset': key[0], 'method': key[1], 'seed': key[2],
                           'selected_source': 'local' if key in local_index else 'archive',
                           'original_round': row['round'], 'present_in_both': key in overlap})
        row['round'] = row['seed'] - settings['base_seed']
        normalized.append(row)
        seeds[key[:2]].add(key[2])
        for measure in MEASURES:
            values[key[0], key[1], measure].append(row[measure])

    OUT.mkdir(exist_ok=False)
    def json_file(name, value):
        (OUT / name).write_text(json.dumps(value, indent=2), encoding='utf-8')
    def json_lines(name, rows):
        (OUT / name).write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
    json_lines('rounds.jsonl', normalized)
    json_lines('provenance.jsonl', provenance)
    json_lines('conflicts.jsonl', conflicts)
    counts = {d: {m: len(seeds[d, m]) for m in methods} for d in datasets}
    json_file('successful_rounds.json', counts)
    with (OUT / 'Repetitions.csv').open('w', newline='') as output:
        writer = csv.writer(output); writer.writerow(['Dataset', *methods])
        writer.writerows([d, *[counts[d][m] for m in methods]] for d in datasets)
    for measure in MEASURES:
        for partial in (False, True):
            name = measure + ('_partial.csv' if partial else '.csv')
            with (OUT / name).open('w', newline='') as output:
                writer = csv.writer(output); writer.writerow(['Dataset', *methods])
                for dataset in datasets:
                    writer.writerow([dataset, *[
                        statistics.median(values[dataset, method, measure])
                        if counts[dataset][method] > 0 and (partial or counts[dataset][method] == 100) else ''
                        for method in methods]])
    settings.update(aggregation='median', summary_source='Deduplicated rounds.jsonl',
                    formal_summary_policy='Exactly 100 distinct seeds per dataset and method; partial values in *_partial.csv',
                    conflict_policy='Retain local record; preserve both records in conflicts.jsonl',
                    imported_parameters_verified='Archive metadata verifies only documented fields; source code and data hashes unavailable')
    json_file('settings.json', settings)
    json_file('archive_completion_metadata.json', completion)
    manifest = {'archive': str(ARCHIVE), 'archive_sha256': hashlib.sha256(ARCHIVE.read_bytes()).hexdigest(),
                'local_rounds_sha256': hashlib.sha256((LOCAL / 'rounds.jsonl').read_bytes()).hexdigest(),
                'archive_members_sha256': archive_members, 'local_records': len(local), 'archive_records': len(remote),
                'overlapping_seeds': len(overlap), 'matching_overlaps': len(overlap)-len(conflicts),
                'conflicting_overlaps': len(conflicts), 'conflicts_by_method': dict(Counter(r['method'] for r in conflicts)),
                'combined_records': len(normalized),
                'archive_rounds_normalized_to_seed': sum(row['round'] != row['seed'] for row in remote),
                'complete_100_round_datasets_by_method': {m: sum(counts[d][m] == 100 for d in datasets) for m in methods},
                'caveats': ['No source-code, dependency-version, or dataset hashes supplied by the archive.',
                            'ManualUp/ManualDown conflicts affect AUC and G_mean; underlying cause is unverified.',
                            'Archive contains no feature-importance results; local interpretation experiment remains separate.']}
    json_file('merge_manifest.json', manifest)
    (OUT / 'README.md').write_text(
        '# Combined Group 5 prediction results\n\n'
        'Sources remain unchanged. Merge identity is dataset + method + seed; resumed round numbers are normalized to seed. '
        'Existing local records take precedence for overlapping seeds. Both versions of differing results are preserved in conflicts.jsonl. '
        'provenance.jsonl records the selected source and original round number.\n\n'
        'The four primary metric tables use medians and require 100 distinct successful seeds. '
        'Incomplete estimates are isolated in *_partial.csv and must not be presented as full 100-round results. '
        'Repetitions.csv gives the actual coverage. Nine methods cover all 28 datasets; TCLP covers six datasets fully, '
        '33 rounds on Derby 10.3.1.4, and one round on each Groovy release.\n\n'
        '212 overlapping ManualUp/ManualDown records disagree in classification scores. Local precedence is an explicit '
        'reconciliation policy, not proof of equivalence between computers. Archive provenance is incomplete; '
        'interpret cross-machine comparisons with this limitation. No feature-importance runs were imported.\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in manifest.items() if k != 'archive_members_sha256'}, indent=2))


if __name__ == '__main__':
    combine()
