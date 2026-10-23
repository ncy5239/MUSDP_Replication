"""Regression checks for stable degree normalization in MUSDP."""
import sys
import unittest
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from algorithms.coreg import MultiviewCoRegSpectralClustering


class NormalizationTests(unittest.TestCase):
    def normalize(self, graph):
        with warnings.catch_warnings():
            warnings.simplefilter('error', RuntimeWarning)
            return MultiviewCoRegSpectralClustering()._init_umat(graph)[1]

    def test_matches_inverse_normalization(self):
        graph = np.array([[0., 2., 3.], [2., 0., 4.], [3., 4., 0.]])
        weights = np.sqrt(np.linalg.inv(np.diag(graph.sum(axis=1))))
        np.testing.assert_allclose(self.normalize(graph), weights @ graph @ weights)

    def test_large_degrees_do_not_overflow(self):
        graph = (np.ones((4, 4)) - np.eye(4)) * 1e100
        normalized = self.normalize(graph)
        self.assertTrue(np.isfinite(normalized).all())
        np.testing.assert_allclose(normalized, (np.ones((4, 4)) - np.eye(4)) / 3)

    def test_isolated_nodes_and_empty_graph(self):
        graph = np.array([[0., 2., 0.], [2., 0., 0.], [0., 0., 0.]])
        np.testing.assert_allclose(self.normalize(graph), graph / 2)
        np.testing.assert_array_equal(self.normalize(np.zeros((3, 3))), np.zeros((3, 3)))


if __name__ == '__main__':
    unittest.main()
