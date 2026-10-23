"""Paired, reproducible perturbation importance for multiview clustering."""

import numpy as np
from sklearn.base import clone
from algorithms.coreg.labelingCluster import labelCluster, labelCluster_v2


def predict_labels(X, cluster_labels):
    """Keep view votes and the size-based labeling independent."""
    votes = [labelCluster(x, cluster_labels.copy()) for x in X]
    # Majority vote provides binary labels for the classification measures.
    metric_labels = (np.mean(votes, axis=0) >= 0.5).astype(int)
    return metric_labels, labelCluster_v2(cluster_labels.copy())


def perturbation_importance(model, X, y, strategy="mean_replacement",
                            random_state=42, n_repeats=1):
    """Return baseline labels/accuracy and per-feature accuracy decreases.

    Features follow view order, then column order. Each perturbed fit uses
    the same clustering seed as the baseline. Permutations use a separate
    local generator. The caller's model and input arrays are not modified.
    Mean replacement uses the evaluation sample's column mean.
    """
    if strategy not in {"mean_replacement", "permutation"}:
        raise ValueError("Unknown perturbation strategy: " + strategy)
    if n_repeats < 1:
        raise ValueError("n_repeats must be positive")
    arrays = [np.asarray(x, dtype=float).copy() for x in X]
    y = np.asarray(y)
    rng = np.random.default_rng(random_state)

    def fit_labels(views):
        estimator = clone(model).set_params(random_state=random_state)
        return predict_labels(views, estimator.fit_predict(views))

    baseline_labels = fit_labels(arrays)
    baseline = np.array([np.mean(p == y) for p in baseline_labels])
    accuracies = []
    for view, x in enumerate(arrays):
        for column in range(x.shape[1]):
            scores = []
            for _ in range(n_repeats if strategy == "permutation" else 1):
                changed = [a.copy() for a in arrays]
                changed[view][:, column] = (
                    rng.permutation(x[:, column]) if strategy == "permutation"
                    else np.mean(x[:, column])
                )
                labels = fit_labels(changed)
                scores.append([np.mean(p == y) for p in labels])
            accuracies.append(np.mean(scores, axis=0))
    perturbed = np.asarray(accuracies).reshape(-1, 2)
    return {
        "baseline_labels": baseline_labels,
        "baseline_accuracy": baseline,
        "perturbed_accuracy": perturbed,
        "importance": baseline - perturbed,
    }


def mean_replacement_importance(model, X, y, random_state=42):
    """Accuracy decrease after replacing each feature with its mean."""
    return perturbation_importance(model, X, y, "mean_replacement", random_state)


def permutation_importance(model, X, y, random_state=42, n_repeats=5):
    """Accuracy decrease averaged over seeded feature permutations."""
    return perturbation_importance(model, X, y, "permutation", random_state, n_repeats)
