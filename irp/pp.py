"""Posterior probabilities
"""


def _ps(t_s, t_e, S, pi, F, G):
    """Marginal likelihood of S. Arguments:
      t_s --- start of interval
      t_e --- end of interval
      S   --- sequence of events (t_i, y_i)
      pi  --- prior intrusion probability
      F   --- interarrival distribution
      G   --- mark distribution
    Returns the marginal likelihood of S and P[], as a tuple.
    """
    N = len(S)

    # Section 5.4, Lemma 3

    P = [None] * N
    for k in range(N):
        tk, yk = S[k]
        P[k] = pi ** k * F.sf(tk - t_s)
        for j in range(k):
            tj = S[j][0]
            P[k] += P[j] * pi ** (k - j - 1) * F.pdf(tk - tj)
        P[k] *= (1 - pi) * G.pdf(yk)
    Ps = pi ** N * F.sf(t_e - t_s)
    for j in range(N):
        tj = S[j][0]
        Ps += P[j] * pi ** (N - j - 1) * F.sf(t_e - tj)

    return Ps, P


def _psnoi(t_s, t_e, S, pi, F, G):
    """Probability of S without intrusion. Arguments:
      t_s --- start of interval
      t_e --- end of interval
      S   --- sequence of events (t_i, y_i)
      pi  --- prior intrusion probability
      F   --- interarrival distribution
      G   --- mark distribution
    Returns the probability.
    """
    N = len(S)
    if N == 0:
        return F.sf(t_e - t_s)

    # Section 5.2, Lemma 2

    t = S[0][0]
    Psnoi = F.sf(t - t_s)
    for k in range(N - 1):
        t1, y1 = S[k]
        t2 = S[k + 1][0]
        Psnoi *= (1 - pi) * F.pdf(t2 - t1) * G.pdf(y1)
    t, y = S[-1]
    Psnoi *= (1 - pi) * F.sf(t_e - t) * G.pdf(y)

    return Psnoi


def intrusion(t_s, t_e, S, pi, F, G):
    """Computes the probability of an intrusion
    in the sequence, given the parameters. Arguments:
      t_s --- start of interval
      t_e --- end of interval
      S   --- sequence of events (t_i, y_i)
      pi  --- prior intrusion probability
      F   --- interarrival distribution
      G   --- mark distribution
    Returns the probability.
    """
    N = len(S)
    if N == 0:
        # An empty sequence cannot contain an intrusion
        return 0.

    # Compute probability of the sequence without intrusion
    Psnoi = _psnoi(t_s, t_e, S, pi, F, G)

    # Compute marginal likelihood of S
    Ps, _ = _ps(t_s, t_e, S, pi, F, G)

    # Section 5.4, Equation 17
    return 1. - Psnoi / Ps


def marginal(t_s, t_e, S, pi, F, G):
    """Computes the marginal probability of each
    event in the sequence to belong to the intrusion.
    Arguments:
      t_s --- start of interval
      t_e --- end of interval
      S   --- sequence of events (t_i, y_i)
      pi  --- prior intrusion probability
      F   --- interarrival distribution
      G   --- mark distribution
    Returns the list of probabilities.
    """
    N = len(S)
    if N == 0:
        return []

    # Section 5.4, Theorem 5

    Ps, Pf = _ps(t_s, t_e, S, pi, F, G)

    t_sb = -t_e
    t_eb = -t_s
    Sb = [(-t, y) for t, y in reversed(S)]

    _, Pb = _ps(t_sb, t_eb, Sb, pi, F, G)
    Pb.reverse()

    return [(1 - Pf[k] * Pb[k] /
             ((1 - pi) * G.pdf(y) * Ps))
            for k, (_, y) in enumerate(S)]
