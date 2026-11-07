import sys, warnings, json
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn import preprocessing
from algorithms.coreg import MultiviewCoRegSpectralClustering
from algorithms.coreg.labelingCluster import labelCluster, labelCluster_v2
from utilities import performanceMeasure, rankMeasure
from utilities.AutoSpearman import AutoSpearman
from utilities.bootstrapCV import outofsample_bootstrap

REPO, AEEEM, OUT, REPS = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), int(sys.argv[4])
sys.path.insert(0, str(REPO))

# The one-view subclass
class CoRegAnyViews(MultiviewCoRegSpectralClustering):

    def _param_checks(self, Xs):
        if len(Xs) >= 2:
            return super()._param_checks(Xs)
        out = super()._param_checks([Xs[0], Xs[0]])
        self._n_views = 1
        return out[:1]

#Views and Systems
SYSTEMS = ["EQ", "JDT", "LC", "ML", "PDE"]
VIEWS = {  # Column-name prefix -> View
    "source": lambda c: c.startswith("ck_oo_"),
    "churn": lambda c: c.startswith("WCHU_"),
    "entropy_src": lambda c: c.startswith("LDHH_"),
    "entropy_chg": lambda c: c.startswith("Cvs"),
    "prev_defects": lambda c: "BugsFoundUntil" in c,
}
LOC_COL = "ck_oo_numberOfLinesOfCode"
CONFIGS = ["full5"] + ["only_" + v for v in VIEWS] + ["concat"] + ["drop_" + v for v in VIEWS]

# Load and split each system
def load(system):
    df = pd.read_csv(AEEEM / f"{system}.csv")
    df.columns = [c.rstrip(":") for c in df.columns]
    y = (df["class"].astype(str).str.lower() == "buggy").astype(int) # Turns buggy/clean into 1/0
    feats = df.drop(columns="class")
    loc = feats[LOC_COL].to_numpy(dtype=float)
    views, kept = {}, {}
    for name, f in VIEWS.items(): # Drops columns, then runs the repo’s AutoSpearman, removes correlated metrics
        cols = [c for c in feats.columns if f(c)]
        v = feats[cols]
        v = v.loc[:, v.std() > 0]
        views[name] = AutoSpearman(v)
        kept[name] = views[name].shape[1]
    assert sum(len(feats.columns[[VIEWS[n](c) for c in feats.columns]]) for n in VIEWS) == 61
    return views, y, loc, kept

# Build the configurations
def config_views(cfg, views):
    if cfg == "full5": # All five views
        return [views[v] for v in VIEWS]
    if cfg.startswith("only_"): # One view alone
        return [views[cfg[5:]]]
    if cfg.startswith("drop_"): # The other four views
        return [views[v] for v in VIEWS if v != cfg[5:]]
    if cfg == "concat":  # All five views’ kept metrics concatennated into one table.
        return [pd.concat([views[v] for v in VIEWS], axis=1)]

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
    m = CoRegAnyViews(n_clusters=2, v_lambda=1, max_iter=10, random_state=rep) # The one-view subclass
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

# It loads all five systems, builds every system, configuration, repeat job
DATA = {}
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    OUT.mkdir(parents=True, exist_ok=True)
    DATA = {s: load(s) for s in SYSTEMS}
    json.dump({s: DATA[s][3] for s in SYSTEMS}, open(OUT / "metrics_kept.json", "w"), indent=1)
    jobs = [(s, c, r) for s in SYSTEMS for c in CONFIGS for r in range(REPS)]
    with Pool(2) as pool:
        rows = pool.map(one_run, jobs, chunksize=8)
    pd.DataFrame(rows).to_csv(OUT / "results_raw.csv", index=False)
    print("done", len(rows))
