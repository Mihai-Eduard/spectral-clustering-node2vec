import unittest
import networkx as nx
import numpy as np
from utils import random_walks, compute_spectral_embeddings


class TestComponents1(unittest.TestCase):
    def setUp(self):
        self.simple_graph = nx.Graph(
            [(0, 3), (1, 3), (2, 4), (3, 5), (3, 6), (4, 6), (5, 6), (5, 4)]
        )

    def test_random_walks(self):
        result = random_walks(self.simple_graph, 2, 5)
        print(result)
        self.assertEqual((7 * 2, 5), result.shape)
        for n in result.flatten():
            self.assertIn(n, self.simple_graph.nodes)


class TestComponents2(unittest.TestCase):
    def setUp(self):
        self.simple_graph = nx.Graph(
            [(0, 3), (1, 3), (2, 4), (3, 5), (3, 6), (4, 6), (5, 6), (5, 4)]
        )

    def test_spectral_embeddings(self):
        emb = compute_spectral_embeddings(self.simple_graph, 3)
        np.testing.assert_almost_equal(
            np.array(
                [
                    [3.77964473e-01, -4.49723806e-01, 7.07106781e-01],
                    [3.77964473e-01, -4.49723806e-01, -7.07106781e-01],
                    [3.77964473e-01, 6.59857436e-01, -1.66533454e-15],
                    [3.77964473e-01, -2.18583490e-01, 1.11022302e-16],
                    [3.77964473e-01, 3.20716714e-01, 6.59900940e-16],
                    [3.77964473e-01, 6.87284763e-02, 1.14927529e-15],
                    [3.77964473e-01, 6.87284763e-02, 1.26029760e-15],
                ]
            ),
            emb,
        )
