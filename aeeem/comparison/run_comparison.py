"""Run the comparison baselines on the five AEEEM systems.

From the repository root:

    python -m aeeem.comparison.run_comparison --reps 100 --procs 8 --output result/aeeem_comparison
    python -m aeeem.comparison.run_comparison --methods SC Churn --output result/quick

Every method sees the same out-of-bootstrap evaluation sample in a round. The
seed is base_seed + round, the same seeds as the MUSDP Part runner, so its
MUSDP rounds pair with these. True labels are used only for evaluation.
"""
import argparse
import json
import sys
import warnings
from multiprocessing import Pool
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[2]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from aeeem.comparison import baselines
from aeeem.comparison.evaluation import evaluate
from aeeem.comparison.summarize import summarize
from aeeem.loader import CHURN_COLUMN, LOC_COLUMN, SYSTEMS, load_system
from utilities.bootstrapCV import outofsample_bootstrap

METHODS = ('SC', 'ManualDown', 'ManualUp', 'Churn', 'ChurnSet',
           'CLA', 'CLAMI', 'TCL', 'TCLP', 'KMedoids')


def predict(method, system, test_data, seed):
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
    if method == 'CLA':
        return baselines.cla(test_data)
    if method == 'CLAMI':
        return baselines.clami(test_data, seed)
    if method == 'TCL':
        return baselines.tcl(test_data)
    if method == 'TCLP':
        return baselines.tclp(test_data, seed)
    if method == 'KMedoids':
        return baselines.kmedoids(test_data, loc, seed)
    raise ValueError(f'Unknown method {method}')


SYSTEM_CACHE = {}


def init_worker():
    warnings.filterwarnings('ignore')
    threadpool_limits(1)  # one BLAS thread per worker; the pool provides the parallelism


def run_round(job):
    """All methods on one system's evaluation sample for one round."""
    name, rnd, seed, methods = job
    if name not in SYSTEM_CACHE:
        SYSTEM_CACHE[name] = load_system(name)
    system = SYSTEM_CACHE[name]
    _, _, test_data, truth, _, test_idx = outofsample_bootstrap(system.frame(), seed)
    loc = system.loc[test_idx]
    rows, failures = [], []
    for method in methods:
        try:
            values = evaluate(truth, predict(method, system, test_data, seed), loc)
        except (ValueError, np.linalg.LinAlgError) as error:
            failures.append(dict(system=name, round=rnd, method=method, reason=str(error)))
            continue
        rows.append(dict(system=name, round=rnd, seed=seed, method=method,
                         n_test=len(truth), n_defective=int(truth.sum()), **values))
    return rows, failures


def run(output, reps=100, systems=SYSTEMS, methods=METHODS, base_seed=0, procs=1):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'settings.json').write_text(json.dumps(dict(
        reps=reps, systems=list(systems), methods=list(methods), base_seed=base_seed,
        seed='base_seed + zero-based round', effort='ck_oo_numberOfLinesOfCode (+1 if any 0)',
        aggregation='median over successful rounds'), indent=2))
    jobs = [(name, rnd, base_seed + rnd, list(methods)) for name in systems for rnd in range(reps)]
    if procs > 1:
        with Pool(procs, initializer=init_worker) as pool:
            results = pool.map(run_round, jobs, chunksize=1)
    else:
        init_worker()
        results = [run_round(job) for job in jobs]
    rows = [row for r, _ in results for row in r]
    failures = [f for _, fs in results for f in fs]
    for f in failures:
        warnings.warn(f"{f['system']} {f['method']} round {f['round']}: {f['reason']}")
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
    parser.add_argument('--methods', nargs='+', default=list(METHODS), choices=METHODS)
    parser.add_argument('--base-seed', type=int, default=0)
    parser.add_argument('--procs', type=int, default=1)
    args = parser.parse_args()
    run(args.output, args.reps, args.systems, args.methods, args.base_seed, args.procs)


if __name__ == '__main__':
    main()
