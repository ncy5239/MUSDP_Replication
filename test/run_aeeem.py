"""Run MUSDP on the five AEEEM systems for every view configuration.

From the repository root:

    python test/run_aeeem.py --reps 100 --output result/aeeem_musdp

Reads data/AEEEM/*.arff through aeeem/loader.py. Works with both the fork
(Linux) and spawn (macOS, Windows) start methods: the loaded systems are
passed to each worker instead of being inherited from the parent.
"""
import argparse
import json
import sys
import warnings
from multiprocessing import Pool
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np
import pandas as pd
from sklearn import preprocessing
from aeeem.loader import DATA_DIR, SYSTEMS, VIEWS, load_system
from algorithms.coreg import MultiviewCoRegSpectralClustering
from algorithms.coreg.labelingCluster import labelCluster, labelCluster_v2
from utilities import performanceMeasure, rankMeasure
from utilities.AutoSpearman import AutoSpearman
from utilities.bootstrapCV import outofsample_bootstrap


# The one-view subclass
class CoRegAnyViews(MultiviewCoRegSpectralClustering):

    def _param_checks(self, Xs):
        if len(Xs) >= 2:
            return super()._param_checks(Xs)
        out = super()._param_checks([Xs[0], Xs[0]])
        self._n_views = 1
        return out[:1]


# Views (column-name prefix -> view) and systems come from aeeem/loader.py
CONFIGS = ["full5"] + ["only_" + v for v in VIEWS] + ["concat"] + ["drop_" + v for v in VIEWS]


# Load and split each system
def load(system, data_dir=DATA_DIR):
    s = load_system(system, data_dir)  # 61 metrics, every metric in exactly one view
    y = pd.Series(s.label)  # buggy/clean as 1/0
    views, kept = {}, {}
    for name in VIEWS:  # Drops constant columns, then the repo's AutoSpearman removes correlated metrics
        v = s.view(name)
        v = v.loc[:, v.std() > 0]
        views[name] = AutoSpearman(v)
        kept[name] = views[name].shape[1]
    return views, y, s.loc, kept


# Build the configurations
def config_views(cfg, views):
    if cfg == "full5":  # All five views
        return [views[v] for v in VIEWS]
    if cfg.startswith("only_"):  # One view alone
        return [views[cfg[5:]]]
    if cfg.startswith("drop_"):  # The other four views
        return [views[v] for v in VIEWS if v != cfg[5:]]
    if cfg == "concat":  # All five views' kept metrics concatenated into one table
        return [pd.concat([views[v] for v in VIEWS], axis=1)]
    raise ValueError(f"Unknown configuration {cfg}")


# Each worker receives the loaded systems once
DATA = {}


def init_worker(data):
    warnings.filterwarnings("ignore")
    DATA.update(data)


# One experimental run
def one_run(args):
    system, cfg, rep = args
    views, y, loc, _ = DATA[system]
    X = [v.reset_index(drop=True) for v in config_views(cfg, views)] + [pd.Series(y.to_numpy().copy())]
    _, _, test_data, test_label, _, test_idx = outofsample_bootstrap(X, rep)
    if len(np.unique(test_label)) < 2:
        return dict(system=system, config=cfg, rep=rep, skipped=1)
    test_loc = loc[test_idx]
    test_data = [preprocessing.scale(x) for x in test_data]
    m = CoRegAnyViews(n_clusters=2, v_lambda=1, max_iter=10, random_state=rep)  # The one-view subclass
    clus = m.fit_predict(test_data)
    votes = [labelCluster(x, clus.copy()) for x in test_data]
    p1 = (np.mean(votes, axis=0) >= 0.5).astype(int)
    p2 = labelCluster_v2(clus.copy())
    row = dict(system=system, config=cfg, rep=rep, skipped=0, n_test=len(test_label))
    for tag, p in (("label1", p1), ("label2", p2)):
        AUC, G, *_ = performanceMeasure.get_measure(test_label, p)
        Popt, rec20, *_ = rankMeasure.rank_measure(p, test_loc, test_label)
        row.update({f"{tag}_G": G, f"{tag}_AUC": AUC, f"{tag}_Popt": Popt, f"{tag}_R20": rec20})
    return row


# Loads all five systems, builds every system, configuration, repeat job
def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=REPO / "result" / "aeeem_musdp")
    parser.add_argument("--reps", type=int, default=100)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--systems", nargs="+", default=list(SYSTEMS))
    parser.add_argument("--configs", nargs="+", default=CONFIGS, choices=CONFIGS)
    parser.add_argument("--procs", type=int, default=2)
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    args.output.mkdir(parents=True, exist_ok=True)
    data = {s: load(s, args.data_dir) for s in args.systems}
    with open(args.output / "metrics_kept.json", "w") as f:
        json.dump({s: data[s][3] for s in args.systems}, f, indent=1)
    jobs = [(s, c, r) for s in args.systems for c in args.configs for r in range(args.reps)]
    with Pool(args.procs, initializer=init_worker, initargs=(data,)) as pool:
        rows = pool.map(one_run, jobs, chunksize=8)
    pd.DataFrame(rows).to_csv(args.output / "results_raw.csv", index=False)
    print("done", len(rows))


if __name__ == "__main__":
    main()
