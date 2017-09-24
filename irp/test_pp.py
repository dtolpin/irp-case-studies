"""Tests for posterior probabilities
"""

import unittest
import irp.pp
import scipy.stats


class TestPP(unittest.TestCase):
    """Tests for posterior probabilities
    """

    def test_ps(self):
        """Marginal likelihood.
        """
        t_s = 0
        t_e = 10
        pi = 0.5
        F = scipy.stats.gamma(a=2, scale=1)
        G = scipy.stats.expon(scale=1)

        Ps, P = irp.pp._ps(t_s, t_e, [], pi, F, G)
        self.assertEqual(Ps, F.sf(t_e - t_s), "empty sequence")

        Ps, P = irp.pp._ps(t_s, t_e, [(5., 1.)], pi, F, G)
        self.assertEqual(Ps,
                         pi * F.sf(t_e - t_s) +

                         # No intrusion
                         (1 - pi) *
                         F.sf(5 - t_s) * F.sf(t_e - 5.) *
                         G.pdf(1.), "singleton")

        Ps, P = irp.pp._ps(t_s, t_e, [(3., 1.), (6., 2.)], pi, F, G)
        self.assertEqual(Ps,
                         pi * pi * F.sf(t_e - t_s) +

                         # Second event is intrusion
                         (1 - pi) * pi *
                         F.sf(3. - t_s) * F.sf(t_e - 3.) *
                         G.pdf(1.) +

                         # First event is intrusion
                         pi * (1. - pi) *
                         F.sf(6. - t_s) * F.sf(t_e - 6.) *
                         G.pdf(2.) +

                         # No intrusion
                         (1 - pi) * (1 - pi) *
                         F.sf(3. - t_s) *
                         F.pdf(6. - 3.) *
                         F.sf(t_e - 6.) *
                         G.pdf(1.) *
                         G.pdf(2.), "general case")

    def test_psnoi(self):
        """Probability of S without intrusion.
        """
        t_s = 0
        t_e = 10
        pi = 0.5
        F = scipy.stats.gamma(a=2, scale=1)
        G = scipy.stats.expon(scale=1)

        Psnoi = irp.pp._psnoi(t_s, t_e, [], pi, F, G)
        self.assertEqual(Psnoi, F.sf(t_e - t_s), "empty sequence")

        Psnoi = irp.pp._psnoi(t_s, t_e, [(5., 1.)], pi, F, G)
        self.assertEqual(Psnoi,
                         (1 - pi) *
                         F.sf(5. - t_s) * F.sf(t_e - 5.) *
                         G.pdf(1.), "singleton")

        Psnoi = irp.pp._psnoi(t_s, t_e, [(3., 1.), (6., 2.)],
                              pi, F, G)
        self.assertEqual(Psnoi,
                         (1 - pi) * (1 - pi) *
                         F.sf(3. - t_s) *
                         F.pdf(6. - 3.) *
                         F.sf(t_e - 6.) *
                         G.pdf(1.) *
                         G.pdf(2.), "general case")

    def test_marginal(self):
        """Marginal probabilities.
        """
        t_s = 0
        t_e = 10
        pi = 0.5
        F = scipy.stats.gamma(a=2, scale=1)
        G = scipy.stats.expon(scale=1)

        P = irp.pp.marginal(t_s, t_e, [], pi, F, G)
        self.assertEqual(P, [], "empty sequence")

        P = irp.pp.marginal(t_s, t_e, [(5., 1.)], pi, F, G)
        self.assertEqual(len(P), 1, "singleton")

        P = irp.pp.marginal(t_s, t_e, [(10. / 3., 1.), (20. / 3., 1.)],
                            pi, F, G)
        self.assertAlmostEqual(P[0], P[1], msg="equal probabilities")

        P = irp.pp.marginal(t_s, t_e, [(3., 1.), (6., 2.)],
                            pi, F, G)
        self.assertEqual(len(P), 2, "general case")


def test_suite():
    """Returns testsuite for posterior probabilities.
    """
    return unittest.TestSuite([TestPP(test)
                               for test in ["test_ps",
                                            "test_psnoi",
                                            "test_marginal"]])
