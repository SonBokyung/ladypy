import numpy as np

__all__ = [
    'derive_P_from', 'derive_Q_from', 'payoff_PQ_self', 'payoff_PQ',
    'payoff_self', 'payoff'
]


def derive_P_from(A):
    """ Make an active matrix P with the given association matrix A.

    >>> import numpy as np
    >>> from evolg.calc import derive_P_from

    >>> A = np.random.random(size=(100, 10, 10))
    >>> P = derive_P_from(A)

    >>> np.isclose(P.sum(axis=2), np.ones((100, 10))).sum()
    1000
    """
    return (A / A.sum(axis=2).reshape(A.shape[0], A.shape[1], 1))


def derive_Q_from(A):
    """ Make a passive matrix Q with the given association matrix A.

    >>> import numpy as np
    >>> from evolg.calc import derive_Q_from

    >>> A = np.random.random(size=(100, 10, 10))
    >>> Q = derive_Q_from(A)

    >>> np.isclose(Q.sum(axis=1), np.ones((100, 10))).sum()
    1000
    """
    return (A / A.sum(axis=1).reshape(A.shape[0], 1, A.shape[2]))


def payoff_PQ_self(P, Q):
    """ Calculate payoffs for each agent itself.

    >>> import numpy as np
    >>> from evolg.calc import derive_P_from, derive_Q_from, payoff_PQ_self

    >>> A = np.random.random(size=(100, 10, 10))
    >>> P = derive_P_from(A)
    >>> Q = derive_Q_from(A)

    >>> all(np.isclose(payoff_PQ_self(P, Q), (P * Q).sum((1, 2))))
    True
    """
    return np.einsum('ijk,ijk->i', P, Q)


def payoff_PQ(P, Q):
    """ Calculate average payoffs of each agent within the group.

    >>> import numpy as np
    >>> from evolg.calc import derive_P_from, derive_Q_from, payoff_PQ

    >>> A = np.random.random(size=(100, 10, 10))
    >>> P = derive_P_from(A)
    >>> Q = derive_Q_from(A)

    >>> cmp = np.array([ \
            sum([0.5 * (P[i] * Q[j] + P[j] * Q[i]).sum() \
                 for j in range(100)]) / 100 \
            for i in range(100) \
        ])

    >>> all(np.isclose(payoff_PQ(P, Q), cmp))
    True
    """
    PQ = np.einsum('ijk,ljk->il', P, Q)
    return 0.5 * (PQ.mean(axis=0) + PQ.mean(axis=1))


def payoff_self(A):
    """ Calculate payoffs for each agent itself.

    >>> import numpy as np
    >>> from evolg import calc
    >>> from evolg.calc import derive_P_from, derive_Q_from, payoff_self

    >>> A = np.random.random(size=(100, 10, 10))
    >>> P = derive_P_from(A)
    >>> Q = derive_Q_from(A)

    >>> all(np.isclose(payoff_self(A), (P * Q).sum((1, 2))))
    True
    """
    return payoff_PQ_self(derive_P_from(A), derive_Q_from(A))


def payoff(A):
    """ Calculate average payoffs of each agent within the group.

    :param A: Association matrices.
    :type A: :class:`numpy.ndarray`

    :Example:
    >>> import numpy as np
    >>> from evolg import calc
    >>> from evolg.calc import derive_P_from, derive_Q_from, payoff

    >>> A = np.random.random(size=(100, 10, 10))
    >>> P = derive_P_from(A)
    >>> Q = derive_Q_from(A)

    >>> cmp = np.array([ \
            sum([0.5 * (P[i] * Q[j] + P[j] * Q[i]).sum() \
                 for j in range(100)]) / 100 \
            for i in range(100) \
        ])

    >>> all(np.isclose(payoff(A), cmp))
    True
    """
    return payoff_PQ(derive_P_from(A), derive_Q_from(A))
