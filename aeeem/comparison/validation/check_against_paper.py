"""Run SC, ManualDown and ManualUp through the AEEEM comparison pipeline on the
original 28 releases and compare the median Pd and Pf with the paper.

    python -m aeeem.comparison.validation.check_against_paper --reps 100

paper_pd_pf.csv holds the per-release medians of Tables 5 (Pd) and 6 (Pf) of
the online supplementary (online_supplementary/supplementary.pdf), the only
per-release values the study publishes for these baselines. G-mean and AUC
follow from Pd and Pf in each round (sqrt(Pd(1-Pf)) and (Pd+1-Pf)/2), so
matching Pd and Pf checks the binary side of the pipeline end to end.
"""
import argparse
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[3]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import pandas as pd

from aeeem.comparison import baselines
from aeeem.comparison.evaluation import evaluate
from utilities.bootstrapCV import outofsample_bootstrap

HERE = Path(__file__).resolve().parent
METHODS = ('SC', 'ManualDown', 'ManualUp')


def run(reps, output):
    reference = pd.read_csv(HERE / 'paper_pd_pf.csv')
    rows = []
    for dataset in sorted(reference['dataset'].unique()):
        data = pd.read_csv(CODE_DIR / 'data' / f'{dataset}.csv')
        for rnd in range(reps):
            _, _, test_data, truth, _, _ = outofsample_bootstrap(data.copy(), rnd)
            loc = test_data['CountLineCode'].to_numpy(dtype=float)
            predictions = dict(SC=baselines.sc(test_data), ManualDown=baselines.manual_down(loc),
                               ManualUp=baselines.manual_up(loc))
            for method, prediction in predictions.items():
                values = evaluate(truth, prediction, loc)
                rows.append(dict(dataset=dataset, round=rnd, method=method,
                                 Pd=values['Pd'], Pf=values['Pf']))
        print(dataset, flush=True)
    ours = (pd.DataFrame(rows).groupby(['dataset', 'method'])[['Pd', 'Pf']].median()
            .stack().rename('our_median').reset_index().rename(columns={'level_2': 'measure'}))
    merged = reference.merge(ours, on=['dataset', 'method', 'measure'])
    merged['abs_diff'] = (merged['our_median'] - merged['paper_median']).abs()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output / 'paper_check.csv', index=False)
    summary = merged.groupby(['method', 'measure'])['abs_diff'].agg(
        mean='mean', max='max', within_0005=lambda d: (d <= 0.0005).mean(),
        within_002=lambda d: (d <= 0.02).mean())
    summary.to_csv(output / 'paper_check_summary.csv')
    print(summary.to_string())
    return merged


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--reps', type=int, default=100)
    parser.add_argument('--output', default=str(CODE_DIR / 'result' / 'aeeem_comparison_paper_check'))
    args = parser.parse_args()
    run(args.reps, args.output)


if __name__ == '__main__':
    main()
