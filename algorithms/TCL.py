"""TCL/TCLP, Kumar et al., IEEE Transactions on Reliability 71(2), 2022.

Input contains unlabeled, finite, nonnegative numeric metrics only.
The implementation choices needed beyond the paper are documented in Readme.md.
"""
import math

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold


class TCLPSelectionError(ValueError):
    """Selection cannot supply enough pseudo-labeled examples for ROSE."""


def _threshold_and_label(data):
    values = np.asarray(data, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 1:
        raise ValueError('data must have at least two rows and one metric column')
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('TCL requires finite, nonnegative raw metric values')
    transformed = np.log1p(values)
    # Paper equations (2)-(4): sample SD and exp(mu + SD), without minus 1.
    log_thresholds = transformed.mean(axis=0) + transformed.std(axis=0, ddof=1)
    with np.errstate(over='ignore'):
        thresholds = np.exp(log_thresholds)
    high = values > thresholds
    counts = high.sum(axis=1)
    labels = (counts > counts.max() // 2).astype(np.int64)
    return values, thresholds, high, counts, labels


def TCL(data, return_details=False):
    """Return 1=faulty, 0=nonfaulty labels for every original module.

    Optionally return (labels, details) with thresholds and exceedance counts.
    """
    _, thresholds, _, counts, labels = _threshold_and_label(data)
    if return_details:
        return labels, {'thresholds': thresholds, 'counts': counts,
                        'label_cutoff': int(counts.max() // 2)}
    return labels


def _select_training(high, labels):
    """Accumulate low-MVR metrics; retain low-IVR rows, including cutoff ties."""
    n, m = high.shape
    violations = high != labels[:, None].astype(bool)
    metric_scores = violations.sum(axis=0)
    minimum_metrics = max(1, math.ceil(math.log2(m)))
    # ROSE needs at least two examples of each class, not just two faulty rows.
    if min(np.bincount(labels, minlength=2)) < 2:
        raise TCLPSelectionError('TCLP requires at least two TCL-labeled modules in each class')
    for score in np.unique(metric_scores):
        columns = np.flatnonzero(metric_scores <= score)
        if len(columns) < minimum_metrics:
            continue
        instance_scores = violations[:, columns].sum(axis=1)
        for fraction in (0.70, 0.80, 0.90, 0.99):
            count = math.ceil(fraction * n)
            boundary = np.partition(instance_scores, count - 1)[count - 1]
            rows = np.flatnonzero(instance_scores <= boundary)
            if min(np.bincount(labels[rows], minlength=2)) >= 2:
                return columns, rows, fraction, metric_scores / n, instance_scores / len(columns)
    raise TCLPSelectionError('No low-violation selection retains two modules in each class')


def _rose(values, labels, random_state):
    """Numeric ROSE smoothed bootstrap, p=0.5, N=n, Gaussian diagonal kernel.

    Bandwidth per class: sd * (4 / ((d+2)*n_class))**(1/(d+4)).
    Class counts follow Bernoulli sampling, so balance is approximate. Redraw
    the class allocation if either class is absent. Do not clip Gaussian draws.
    This follows ROSE's numeric kernel method, not R's random-number sequence.
    """
    rng = np.random.default_rng(random_state)
    if min(np.bincount(labels, minlength=2)) < 2:
        raise TCLPSelectionError('ROSE requires at least two examples in each class')
    n, d = values.shape
    generated_labels = rng.integers(0, 2, size=n)
    while np.unique(generated_labels).size != 2:
        generated_labels = rng.integers(0, 2, size=n)
    generated = np.empty_like(values, dtype=float)
    for label in (0, 1):
        members = values[labels == label]
        positions = np.flatnonzero(generated_labels == label)
        bandwidth = members.std(axis=0, ddof=1) * (4 / ((d + 2) * len(members))) ** (1 / (d + 4))
        centers = members[rng.integers(0, len(members), size=len(positions))]
        generated[positions] = centers + rng.normal(size=centers.shape) * bandwidth
    return generated, generated_labels


def _fit_forest(values, labels, mtry, n_estimators, seed, n_jobs):
    balanced, balanced_labels = _rose(values, labels, seed)
    model = RandomForestClassifier(n_estimators=n_estimators, max_features=mtry,
                                   random_state=seed, n_jobs=n_jobs)
    model.fit(balanced, balanced_labels)
    return model, np.bincount(balanced_labels, minlength=2)


def TCLP(data, random_state=1, n_estimators=500, cv_folds=10,
         tune_mtry=True, n_jobs=1, return_details=False):
    """TCL labels -> MVR/IVR selection -> ROSE -> random forest -> all rows.

    Tune mtry=1..p by accuracy on selected pseudo labels. ROSE runs only on
    training folds. Folds reduce to min(cv_folds, minority count) for small
    samples. With <3 rows in either class, skip tuning and use floor(sqrt(p))
    because fold-local ROSE needs >=2 per class. Details record that fallback.
    """
    if not isinstance(n_estimators, (int, np.integer)) or n_estimators < 1:
        raise ValueError('n_estimators must be a positive integer')
    if not isinstance(cv_folds, (int, np.integer)) or cv_folds < 2:
        raise ValueError('cv_folds must be an integer of at least 2')
    values, thresholds, high, counts, labels = _threshold_and_label(data)
    columns, rows, fraction, mvr, ivr = _select_training(high, labels)
    train = values[rows][:, columns]
    target = labels[rows]
    p = len(columns)
    mtry = max(1, math.floor(math.sqrt(p)))
    scores = {}
    folds_used = 0
    minority_count = int(np.bincount(target, minlength=2).min())
    possible_folds = min(cv_folds, minority_count)
    if tune_mtry and minority_count - math.ceil(minority_count / possible_folds) >= 2:
        folds_used = min(cv_folds, minority_count)
        splits = list(StratifiedKFold(n_splits=folds_used, shuffle=True,
                                     random_state=random_state).split(train, target))
        for candidate in range(1, p + 1):
            accuracies = []
            for fold, (fit, validate) in enumerate(splits):
                model, _ = _fit_forest(train[fit], target[fit], candidate,
                                       n_estimators, random_state + fold + 1, n_jobs)
                accuracies.append(model.score(train[validate], target[validate]))
            scores[candidate] = float(np.mean(accuracies))
        # Ascending candidates give deterministic preference to smaller mtry on ties.
        mtry = max(scores, key=scores.get)
    model, balanced_counts = _fit_forest(train, target, mtry, n_estimators, random_state, n_jobs)
    predicted = model.predict(values[:, columns]).astype(np.int64)
    if return_details:
        return predicted, {
            'thresholds': thresholds, 'counts': counts, 'initial_labels': labels,
            'selected_metrics': columns, 'selected_instances': rows,
            'retention_fraction': fraction, 'metric_violation_ratios': mvr,
            'instance_violation_ratios': ivr, 'mtry': int(mtry),
            'cv_folds_used': folds_used, 'cv_accuracy': scores,
            'rose_class_counts': balanced_counts,
            'tuning_status': ('tuned' if folds_used else
                              'disabled' if not tune_mtry else 'insufficient_class_support')}
    return predicted
