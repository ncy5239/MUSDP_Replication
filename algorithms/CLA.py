"""CLA and CLAMI (Nam and Kim, ASE 2015), operating on unlabeled metrics.

Rows are modules; columns must contain numeric metrics only, never defect labels.
The cutoff defaults to the paper's per-metric median. Labels are 1=buggy, 0=clean.
"""
import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class CLAMISelectionError(ValueError):
    """No violation-score group retains both pseudo-labeled classes."""


def _cluster_and_label(data, percentile):
    values = np.asarray(data, dtype=float)
    if values.ndim != 2 or 0 in values.shape:
        raise ValueError('data must be a nonempty 2-D matrix of numeric metrics')
    if not np.isfinite(values).all():
        raise ValueError('data must contain only finite metric values')
    if not np.isfinite(percentile) or not 0 <= percentile <= 100:
        raise ValueError('percentile must be between 0 and 100')
    cutoffs = np.percentile(values, percentile, axis=0)
    high = values > cutoffs
    counts = high.sum(axis=1)
    # Split distinct K clusters, not the sample-weighted distribution of K.
    # For an odd number of clusters the middle cluster belongs to clean.
    labels = (counts > np.median(np.unique(counts))).astype(np.int64)
    return values, high, labels


def CLA(data, percentile=50):
    """Return CLA labels by counting metrics strictly above their cutoffs.

Clusters above the median distinct K are buggy; the remainder are clean.
An unseparable dataset (one K cluster) therefore receives all-clean labels.
"""
    return _cluster_and_label(data, percentile)[2]


def CLAMI(data, percentile=50, classifier=None, random_state=0):
    """Select metrics/instances from CLA labels, fit, and predict all rows.

For each ascending *distinct* metric violation score, select exactly the
metrics with that score and discard rows violating any selected metric.
Use the first group retaining both classes. Raise CLAMISelectionError if
none works, rather than silently substituting another prediction method.

Default learner: standardized sklearn logistic regression (not an exact
replication of Weka's learner). A supplied sklearn estimator is cloned.
Only pseudo labels are used; the caller never supplies true defect labels.
"""
    values, high, labels = _cluster_and_label(data, percentile)
    violations = high != labels[:, None].astype(bool)
    # Equal denominators make integer violation counts equivalent to MVS.
    scores = violations.sum(axis=0)
    for score in np.unique(scores):
        selected = scores == score
        retained = ~violations[:, selected].any(axis=1)
        if np.unique(labels[retained]).size != 2:
            continue
        learner = (clone(classifier) if classifier is not None else
                   make_pipeline(StandardScaler(), LogisticRegression(
                       max_iter=1000, random_state=random_state)))
        learner.fit(values[retained][:, selected], labels[retained])
        return np.asarray(learner.predict(values[:, selected]), dtype=np.int64)
    raise CLAMISelectionError('No metric violation score retains both buggy and clean modules')
