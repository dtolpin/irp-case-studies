"""Maximum a posteriori subsequence.
"""


def labels(t_s, t_e, S, pi, F, G):
    """Computes maximum aposteriori intrusion
    subsequence of the sequence given parameters.
    Returns the sequence as a list of pairs (t, y).
    Arguments:
      t_s --- start of interval
      t_e --- end of interval
      S   --- sequence of events (t_i, y_i)
      pi  --- prior intrusion probability
      F   --- interarrival distribution
      G   --- mark distribution
    return []
    Returns the label vector: 1 - intrusion, 0 - normal behavior.
    """
    N = len(S)

    # Section 5.3

    # Allocate space for probabilities and back-links
    P = [None] * (N + 1)
    prev = [None] * (N + 1)   # back-linked list of normal events

    # Find the most likely predecessor of each event
    for k in range(N):
        tk, yk = S[k]
        P[k] = pi ** k * F.sf(tk - t_s)
        prev[k] = -1
        for j in range(k):
            tj = S[j][0]
            Pj = P[j] * pi ** (k - j - 1) * F.pdf(tk - tj)
            if Pj > P[k]:
                P[k] = Pj
                prev[k] = j
        P[k] *= (1 - pi) * G.pdf(yk)

    # Find the most likely last event
    P[N] = pi ** N * F.sf(t_e - t_s)
    prev[N] = -1
    for j in range(N):
        tj = S[j][0]
        Pj = P[j] * pi ** (N - j - 1) * F.sf(t_e - tj)
        if Pj > P[N]:
            P[N] = Pj
            prev[N] = j

    # Fill the list of labels by tracing the back links
    labels = [1] * N
    k = prev[N]
    while k != -1:
        labels[k] = 0
        k = prev[k]

    return labels
