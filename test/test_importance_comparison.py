"""Focused checks for perturbation isolation, repeatability, and ranking reports."""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from algorithms.coreg import MultiviewCoRegSpectralClustering
from algorithms.coreg.mean_replacement_importance import (
    mean_replacement_importance, permutation_importance,
)
from utilities.compare_importance import compare_results


class RecordingClusterer(BaseEstimator):
    calls = []

    def __init__(self, random_state=None):
        self.random_state = random_state

    def fit_predict(self, X):
        self.calls.append(([x.copy() for x in X], self.random_state))
        return (X[0][:, 0] > 0).astype(int)


class ImportanceTests(unittest.TestCase):
    def setUp(self):
        self.X = [np.array([[-3., 1.], [-2., 3.], [2., 4.], [3., 8.]])]
        self.y = np.array([0, 0, 1, 1])

    def test_mean_replaces_only_one_column_without_mutation(self):
        original = self.X[0].copy()
        RecordingClusterer.calls = []
        result = mean_replacement_importance(RecordingClusterer(), self.X, self.y, 17)
        self.assertEqual(len(RecordingClusterer.calls), 3)
        np.testing.assert_array_equal(self.X[0], original)
        for column, (views, seed) in enumerate(RecordingClusterer.calls[1:]):
            expected = original.copy()
            expected[:, column] = original[:, column].mean()
            np.testing.assert_array_equal(views[0], expected)
            self.assertEqual(seed, 17)
        np.testing.assert_allclose(result['importance'],
                                   result['baseline_accuracy'] - result['perturbed_accuracy'])

    def test_real_clustering_is_reproducible(self):
        rng = np.random.default_rng(9)
        y = np.repeat([0, 1], 8)
        views = [rng.normal(size=(16, 2)) + y[:, None] * 3 for _ in range(3)]
        model = MultiviewCoRegSpectralClustering(max_iter=2, n_init=2)
        a = permutation_importance(model, views, y, 12, 2)
        b = permutation_importance(model, views, y, 12, 2)
        np.testing.assert_array_equal(a['importance'], b['importance'])
        self.assertIsNone(model.random_state)

    def test_permutation_preserves_values_and_uses_fixed_fit_seed(self):
        original = self.X[0].copy()
        RecordingClusterer.calls = []
        permutation_importance(RecordingClusterer(), self.X, self.y, 19, 2)
        self.assertEqual(len(RecordingClusterer.calls), 5)
        for call, (views, seed) in enumerate(RecordingClusterer.calls[1:]):
            column = call // 2
            np.testing.assert_array_equal(np.sort(views[0][:, column]),
                                          np.sort(original[:, column]))
            np.testing.assert_array_equal(views[0][:, 1 - column], original[:, 1 - column])
            self.assertEqual(seed, 19)
        np.testing.assert_array_equal(self.X[0], original)

    def test_comparison_reports_rank_reversal_and_missing_target(self):
        rows = []
        for labeling in ['label1', 'label2']:
            for strategy, values in [('permutation', [3, 2, 1]), ('mean_replacement', [1, 2, 3])]:
                for feature, value in zip(['Added_lines', 'other1', 'other2'], values):
                    rows.append(dict(dataset='sample', round=0, labeling=labeling,
                                     view=1, feature=feature, strategy=strategy, importance=value))
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temp:
            root = Path(temp)
            pd.DataFrame(rows).to_csv(root / 'importance.csv', index=False)
            _, stability = compare_results(root / 'importance.csv', root / 'report')
            np.testing.assert_allclose(stability.spearman, -1)
            targets = pd.read_csv(root / 'report' / 'added_deleted_lines_summary.csv')
            self.assertTrue((targets.loc[targets.feature == 'Del_lines', 'selection_frequency'] == 0).all())
            self.assertTrue((root / 'report' / 'report.md').exists())


if __name__ == '__main__':
    unittest.main()
