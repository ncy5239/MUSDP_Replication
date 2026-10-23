"""PAM k-medoids and ASFM defect labeling; input contains metrics only."""
import warnings

import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler


def KMedoids(data, n_clusters=2, random_state=0, standardize=True,
             metric='euclidean', max_iter=100, return_details=False):
    """Return binary pseudo labels (1=predicted defective).

    Fit on the unlabeled input itself, with seeded distance-weighted initial
    medoids followed by best-improving PAM swaps. Both distances and ASFM use
    standardized metrics by default; disable standardize to use raw metrics.
    Clusters strictly above the unweighted mean cluster ASFM are defective.
    Thus two clusters use the higher-ASFM rule; tied ASFMs receive clean labels.
    Requires at least n_clusters distinct observations. Costs use O(n**2) memory.
    """
    values = np.asarray(data, dtype=float)
    if values.ndim != 2 or 0 in values.shape or not np.isfinite(values).all():
        raise ValueError('data must be a nonempty finite 2-D metric matrix')
    for name, value in [('n_clusters', n_clusters), ('max_iter', max_iter)]:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f'{name} must be a positive integer')
    if not 2 <= n_clusters <= len(values):
        raise ValueError('n_clusters must be between 2 and the number of rows')
    if np.unique(values, axis=0).shape[0] < n_clusters:
        raise ValueError('data must contain at least n_clusters distinct rows')
    working = StandardScaler().fit_transform(values) if standardize else values.copy()
    distances = pairwise_distances(working, metric=metric)
    if (not np.isfinite(distances).all() or (distances < 0).any()
            or not np.allclose(distances, distances.T)):
        raise ValueError('metric must produce finite nonnegative symmetric distances')
    np.fill_diagonal(distances, 0)
    rng = np.random.default_rng(random_state)
    medoids = [int(rng.integers(len(values)))]
    while len(medoids) < n_clusters:
        nearest = distances[:, medoids].min(axis=1)
        if nearest.sum() == 0:
            raise ValueError('metric does not distinguish enough observations')
        medoids.append(int(rng.choice(len(values), p=nearest / nearest.sum())))
    medoids = np.asarray(medoids)
    history = [float(distances[:, medoids].min(axis=1).sum())]
    converged = False
    for _ in range(max_iter):
        current = distances[:, medoids]
        owners = current.argmin(axis=1)
        nearest = current.min(axis=1)
        second = np.partition(current, 1, axis=1)[:, 1]
        candidates = np.setdiff1d(np.arange(len(values)), medoids)
        best_cost, best_swap = history[-1], None
        for slot in range(n_clusters):
            retained = np.where(owners == slot, second, nearest)
            # Evaluate candidates one at a time to avoid a second n-by-n matrix.
            for candidate in candidates:
                cost = float(np.minimum(retained, distances[:, candidate]).sum())
                if cost < best_cost - 1e-12 * max(1., abs(best_cost)):
                    best_cost, best_swap = cost, (slot, candidate)
        if best_swap is None:
            converged = True
            break
        medoids[best_swap[0]] = best_swap[1]
        history.append(best_cost)
    if not converged:
        warnings.warn('PAM reached max_iter; further swaps may improve the result',
                      RuntimeWarning, stacklevel=2)
    clusters = distances[:, medoids].argmin(axis=1)
    # A medoid owns itself even when a custom distance ties distinct points.
    clusters[medoids] = np.arange(n_clusters)
    sums = working.sum(axis=1)
    asfm = np.array([sums[clusters == c].mean() for c in range(n_clusters)])
    defective = asfm > asfm.mean()
    predicted = defective[clusters].astype(np.int64)
    if return_details:
        return predicted, dict(medoid_indices=medoids.copy(), cluster_labels=clusters,
                               asfm=asfm, defective_clusters=defective,
                               inertia=history[-1], cost_history=history,
                               n_swaps=len(history) - 1, converged=converged)
    return predicted


def rank_modules(labels, effort):
    """Return row indices: defective first, then ascending LOC in each group."""
    labels, effort = np.asarray(labels), np.asarray(effort, dtype=float)
    if (labels.ndim != 1 or effort.shape != labels.shape
            or not np.isin(labels, [0, 1]).all()
            or not np.isfinite(effort).all() or (effort < 0).any()):
        raise ValueError('labels must be binary and effort finite, nonnegative, and aligned')
    return np.lexsort((np.arange(len(labels)), effort, -labels.astype(int)))
