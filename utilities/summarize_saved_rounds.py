"""Rebuild median performance tables from saved rounds without fitting models."""
import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

MEASURES = ('G_mean', 'AUC', 'Popt', 'Recall20')


def summarize_saved_rounds(result_dir):
    result_dir = Path(result_dir)
    settings_path = result_dir / 'settings.json'
    settings = json.loads(settings_path.read_text(encoding='utf-8'))
    datasets, methods = settings['datasets'], settings['methods']
    values = defaultdict(list)
    seen = set()
    lines = (result_dir / 'rounds.jsonl').read_text(encoding='utf-8').splitlines(keepends=True)
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            # An active writer may not yet have finished the last record.
            if i == len(lines) - 1 and not line.endswith('\n'):
                break
            raise
        key = (row['dataset'], row['method'], row['round'])
        if key in seen:
            raise ValueError(f'Duplicate saved round: {key}')
        if key[0] not in datasets or key[1] not in methods:
            raise ValueError(f'Unconfigured dataset or method: {key}')
        seen.add(key)
        for measure in MEASURES:
            value = float(row[measure])
            if not math.isfinite(value):
                raise ValueError(f'Nonfinite {measure}: {key}')
            values[(key[0], key[1], measure)].append(value)
    for measure in MEASURES:
        path = result_dir / (measure + '.csv')
        temporary = path.with_suffix('.csv.tmp')
        with temporary.open('w', newline='', encoding='utf-8') as output:
            writer = csv.writer(output)
            writer.writerow(['Dataset', *methods])
            for dataset in datasets:
                writer.writerow([dataset, *[
                    statistics.median(values[(dataset, method, measure)])
                    if values[(dataset, method, measure)] else ''
                    for method in methods]])
        temporary.replace(path)
    settings['aggregation'] = 'median'
    settings['summary_source'] = 'Completed records in rounds.jsonl'
    settings_path.write_text(json.dumps(settings, indent=2), encoding='utf-8')
    return len(seen)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('result_dir', type=Path)
    args = parser.parse_args()
    print(f'Rebuilt median tables from {summarize_saved_rounds(args.result_dir)} saved method-rounds')
