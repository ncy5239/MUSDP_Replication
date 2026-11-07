"""One evaluation for every AEEEM prediction: G-mean, AUC, Popt, Recall@20%.

It reuses the original study's measure code (utilities/performanceMeasure.py and
utilities/rankMeasure.py) so AEEEM numbers stay comparable with the paper:

- G-mean and AUC use the binary label. AUC on a binary label equals
  (Pd + 1 - Pf) / 2, which is how the paper reports it for every method.
- Popt and Recall@20% use LOC as effort. Ranking methods (score is set) are
  inspected in score order. Binary methods are inspected defective-first,
  then by predicted density, the original CBS+ convention.
- Recall@20% includes the module that reaches or first passes 20% of LOC.

Effort is a numpy array with +1 added to every module when any LOC is 0.
rankMeasure.rank_measure does this itself with `0 in effort`, which tests
index labels, not values, when it is given a pandas Series; passing an array
here avoids that.
"""
import numpy as np
from sklearn.metrics import roc_auc_score

from utilities import performanceMeasure, rankMeasure

MEASURES = ('G_mean', 'AUC', 'Popt', 'Recall20')


def effort_from_loc(loc):
    loc = np.asarray(loc, dtype=float)
    return loc + 1 if np.any(loc == 0) else loc


def evaluate(truth, prediction, loc):
    truth = np.asarray(truth, dtype=int)
    label = np.asarray(prediction.label, dtype=int)
    if np.unique(truth).size < 2:
        raise ValueError('Evaluation sample lacks both classes')
    effort = effort_from_loc(loc)
    auc, g_mean, _, pd_, pf, *_ = performanceMeasure.get_measure(truth, label)
    if prediction.score is None:
        popt, recall20, *_ = rankMeasure.rank_measure(label.astype(float), effort, truth)
        auc_score = auc
    else:
        score = np.asarray(prediction.score, dtype=float)
        popt, recall20, *_ = rankMeasure.rank_measure(score, effort, truth, opt=1)
        auc_score = roc_auc_score(truth, score)
    values = dict(G_mean=g_mean, AUC=auc, Popt=popt, Recall20=recall20,
                  Pd=pd_, Pf=pf, AUC_score=auc_score)
    values = {k: float(v) for k, v in values.items()}
    if not np.isfinite(list(values.values())).all():
        raise ValueError('Evaluation produced a nonfinite measure')
    return values
