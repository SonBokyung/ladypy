from __future__ import division, absolute_import
import numpy as np
from numba import jit, autojit

__all__ = [
    'derive_P_from', 'derive_Q_from', 'payoff', 'payoff_avg', 'cross_entropy'
]


@autojit
def derive_P_from(A):
    """ Make an active matrix P with the given association matrix A.

    >>> import numpy as np
    >>> from ladypy.calc import derive_P_from

    >>> A = np.random.random(size=(100, 10, 10))
    >>> P = derive_P_from(A)

    >>> np.isclose(P.sum(axis=2), np.ones((100, 10))).sum()
    1000
    """
    return (A / A.sum(axis=2).reshape(A.shape[0], A.shape[1], 1))


@autojit
def derive_Q_from(A):
    """ Make a passive matrix Q with the given association matrix A.

    >>> import numpy as np
    >>> from ladypy.calc import derive_Q_from

    >>> A = np.random.random(size=(100, 10, 10))
    >>> Q = derive_Q_from(A)

    >>> np.isclose(Q.sum(axis=1), np.ones((100, 10))).sum()
    1000
    """
    return (A / A.sum(axis=1).reshape(A.shape[0], 1, A.shape[2]))


@autojit
def payoff(P, Q):
    return np.einsum('ijk,ljk->il', P, Q)


@autojit
def payoff_avg(P, Q):
    """ Calculate average payoffs of each agent within the group.

    >>> import numpy as np
    >>> from ladypy.calc import derive_P_from, derive_Q_from, payoff_PQ

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
    PQ = payoff(P, Q)
    return 0.5 * (PQ.mean(axis=0) + PQ.mean(axis=1))


@autojit
def cross_entropy(P, Q):
    return np.einsum('aji,bji->abj', P, -1 * np.log(Q)).mean(axis=2)
