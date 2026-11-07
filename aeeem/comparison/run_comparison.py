"""Run the comparison baselines on the five AEEEM systems.

From the repository root:

    python -m aeeem.comparison.run_comparison --reps 100 --output result/aeeem_comparison

Every method sees the same out-of-bootstrap evaluation sample in a round. The
seed is base_seed + round, the same seeds as the MUSDP Part runner, so its
MUSDP rounds pair with these. True labels are used only for evaluation.
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[2]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import numpy as np
import pandas as pd

from aeeem.comparison import baselines
from aeeem.comparison.evaluation import evaluate
from aeeem.comparison.summarize import summarize
from aeeem.loader import CHURN_COLUMN, LOC_COLUMN, SYSTEMS, load_system
from utilities.bootstrapCV import outofsample_bootstrap

METHODS = ('SC', 'ManualDown', 'ManualUp', 'Churn', 'ChurnSet')


def predict(method, system, test_data):
    loc = test_data[LOC_COLUMN].to_numpy(dtype=float)
    if method == 'SC':
        return baselines.sc(test_data)
    if method == 'ManualDown':
        return baselines.manual_down(loc)
    if method == 'ManualUp':
        return baselines.manual_up(loc)
    if method == 'Churn':
        return baselines.churn(test_data[CHURN_COLUMN].to_numpy(dtype=float), loc)
    if method == 'ChurnSet':
        churn_cols = system.view('churn').columns
        return baselines.churn_set(test_data[churn_cols], loc)
    raise ValueError(f'Unknown method {method}')


def run(output, reps=100, systems=SYSTEMS, methods=METHODS, base_seed=0):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'settings.json').write_text(json.dumps(dict(
        reps=reps, systems=list(systems), methods=list(methods), base_seed=base_seed,
        seed='base_seed + zero-based round', effort='ck_oo_numberOfLinesOfCode (+1 if any 0)',
        aggregation='median over successful rounds'), indent=2))
    rows, failures = [], []
    for name in systems:
        system = load_system(name)
        frame = system.frame()
        for rnd in range(reps):
            seed = base_seed + rnd
            _, _, test_data, truth, _, test_idx = outofsample_bootstrap(frame.copy(), seed)
            loc = system.loc[test_idx]
            for method in methods:
                try:
                    values = evaluate(truth, predict(method, system, test_data), loc)
                except (ValueError, np.linalg.LinAlgError) as error:
                    failures.append(dict(system=name, round=rnd, method=method, reason=str(error)))
                    warnings.warn(f'{name} {method} round {rnd}: {error}')
                    continue
                rows.append(dict(system=name, round=rnd, seed=seed, method=method,
                                 n_test=len(truth), n_defective=int(truth.sum()), **values))
        print(f'{name}: {reps} rounds done', flush=True)
    rounds = pd.DataFrame(rows)
    rounds.to_csv(output / 'rounds.csv', index=False)
    pd.DataFrame(failures, columns=['system', 'round', 'method', 'reason']).to_csv(
        output / 'failures.csv', index=False)
    summarize(rounds, output)
    return rounds


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--output', default=str(CODE_DIR / 'result' / 'aeeem_comparison'))
    parser.add_argument('--reps', type=int, default=100)
    parser.add_argument('--systems', nargs='+', default=list(SYSTEMS))
    parser.add_argument('--methods', nargs='+', default=list(METHODS))
    parser.add_argument('--base-seed', type=int, default=0)
    args = parser.parse_args()
    run(args.output, args.reps, args.systems, args.methods, args.base_seed)


if __name__ == '__main__':
    main()
