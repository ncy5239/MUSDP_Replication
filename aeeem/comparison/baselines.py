"""Unsupervised baselines migrated to AEEEM.

Every baseline returns a Prediction. `label` is the binary prediction
(1 = defective) used by G-mean and AUC. `score` is the inspection order for
ranking methods (higher = inspect first) used by Popt and Recall@20%; binary
methods leave it as None and are ranked by label, then predicted defect density,
exactly as the original evaluation does for SC and MUSDP.

None of the baselines sees the true labels.
"""
from dataclasses import dataclass

import numpy as np
from sklearn import preprocessing

from algorithms.CLA import CLA, CLAMI
from algorithms.KMedoids import KMedoids, rank_modules
from algorithms.ManualDownUp import manualDown, manualUp
from algorithms.SC import SC
from algorithms.TCL import TCL, TCLP


@dataclass(frozen=True)
class Prediction:
    label: np.ndarray
    score: np.ndarray | None = None


def sc(features):
    """Zhang et al. (2016) spectral clustering on all metrics, k = 2.

    Same as the original baseline: z-score inside SC, no AutoSpearman, and the
    cluster with the larger mean row sum of standardized metrics is defective.
    """
    return Prediction(label=SC(np.asarray(features, dtype=float)).astype(int))


def manual_down(loc):
    """Zhou et al. (2018): larger LOC first; the top floor(n/2) are defective."""
    loc = np.asarray(loc, dtype=float)
    return Prediction(label=manualDown(loc.copy()).astype(int), score=loc)


def manual_up(loc):
    """Zhou et al. (2018): smaller LOC first; the top floor(n/2) are defective."""
    loc = np.asarray(loc, dtype=float)
    loc = loc + 1 if np.any(loc == 0) else loc
    return Prediction(label=manualUp(loc.copy()).astype(int), score=1 / loc)


def ranked_top_half(primary, loc):
    """Rank by `primary` (descending), ties by larger LOC, then by row order.

    The top floor(n/2) are labelled defective, the same cut-off as ManualDown,
    so a churn baseline differs from ManualDown only in the metric it ranks by.
    """
    primary = np.asarray(primary, dtype=float)
    loc = np.asarray(loc, dtype=float)
    n = len(primary)
    order = np.lexsort((np.arange(n), -loc, -primary))
    score = np.empty(n)
    score[order] = np.arange(n, 0, -1)
    label = np.zeros(n, dtype=int)
    label[order[:n // 2]] = 1
    return Prediction(label=label, score=score)


def churn(churn_loc, loc):
    """Minimal churn baseline: rank by one churn metric, WCHU_numberOfLinesOfCode."""
    return ranked_top_half(churn_loc, loc)


def churn_set(churn_view, loc):
    """Predefined churn set: the sum of the 17 z-scored WCHU_* metrics.

    Z-scores are computed on the evaluation sample itself (no labels), so every
    churn metric has the same weight whatever its scale.
    """
    z = preprocessing.scale(np.asarray(churn_view, dtype=float))
    return ranked_top_half(z.sum(axis=1), loc)


# Baselines implemented for the original study (algorithms/), applied to AEEEM
# with the same settings as test/demo_all_methods.py.

def cla(features, percentile=50):
    """Nam and Kim (2015) CLA on all 61 metrics."""
    return Prediction(label=np.asarray(CLA(features, percentile=percentile), dtype=int))


def clami(features, seed, percentile=50):
    """Nam and Kim (2015) CLAMI: CLA pseudo labels, metric/instance selection, classifier."""
    return Prediction(label=np.asarray(CLAMI(features, percentile=percentile, random_state=seed), dtype=int))


def tcl(features):
    """Kumar et al. (2022) threshold clustering on the raw (nonnegative) metrics."""
    return Prediction(label=np.asarray(TCL(features), dtype=int))


def tclp(features, seed, n_estimators=500, cv_folds=10, tune_mtry=True):
    """Kumar et al. (2022) TCLP: TCL labels, selection, ROSE and random forest."""
    return Prediction(label=np.asarray(TCLP(features, random_state=seed, n_estimators=n_estimators,
                                            cv_folds=cv_folds, tune_mtry=tune_mtry), dtype=int))


def kmedoids(features, loc, seed, standardize=True, max_iter=100):
    """Yang et al. (2024) PAM k-medoids with ASFM labelling.

    Inspected defective first, then ascending LOC within both groups
    (rank_modules), the convention test/demo_all_methods.py uses for it.
    """
    label = np.asarray(KMedoids(features, random_state=seed, standardize=standardize,
                                max_iter=max_iter), dtype=int)
    order = rank_modules(label, np.asarray(loc, dtype=float))
    score = np.empty(len(label))
    score[order] = np.arange(len(label), 0, -1)
    return Prediction(label=label, score=score)
