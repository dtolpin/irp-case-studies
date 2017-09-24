"""Tests for MAP assignments
"""

import unittest
import irp.map
import scipy.stats


class TestMAP(unittest.TestCase):
    """Tests for MAP assignments
    """

    def test_labels_empty(self):
        """empty sequence
        """

        labels = irp.map.labels(0, 10, [], 0.01,
                                scipy.stats.uniform(4, 2),
                                scipy.stats.uniform(0.5, 1))
        self.assertEqual(labels, [], "empty")

    def test_labels_singleton(self):
        """singleton sequence
        """
        labels = irp.map.labels(0, 10, [(5, 1)], 0.01,
                                scipy.stats.uniform(4, 2),
                                scipy.stats.uniform(0.5, 1))
        self.assertEqual(labels, [0], "singleton, no intrusion")

        labels = irp.map.labels(0, 10, [(3, 1)], 0.01,
                                scipy.stats.uniform(4, 2),
                                scipy.stats.uniform(0.5, 1))
        self.assertEqual(labels, [1], "singleton, intrusion by t")

        labels = irp.map.labels(0, 10, [(5, 10)], 0.01,
                                scipy.stats.uniform(4, 2),
                                scipy.stats.uniform(0.5, 1))
        self.assertEqual(labels, [1], "singleton, intrusion by y")

    def test_labels_general(self):
        """general sequence
        """
        labels = irp.map.labels(0, 15, [(5, 1), (10, 1)], 0.01,
                                scipy.stats.uniform(4, 2),
                                scipy.stats.uniform(0.5, 1))
        self.assertEqual(labels, [0, 0], "general, no intrusion")

        labels = irp.map.labels(0, 15, [(5, 1), (10, 1), (13, 1)],
                                0.01,
                                scipy.stats.uniform(4, 2),
                                scipy.stats.uniform(0.5, 1))
        self.assertEqual(labels, [0, 0, 1], "general, intrusion by t")

        labels = irp.map.labels(0, 15, [(4, 1), (9, 10), (14, 1)],
                                0.01,
                                scipy.stats.uniform(4, 8),
                                scipy.stats.uniform(0.5, 1))
        self.assertEqual(labels, [0, 1, 0], "general, intrusion by y")


def test_suite():
    """Returns testsuite for posterior probabilities.
    """
    return unittest.TestSuite([TestMAP(test)
                               for test
                               in ["test_labels_empty",
                                   "test_labels_singleton",
                                   "test_labels_general"]])
