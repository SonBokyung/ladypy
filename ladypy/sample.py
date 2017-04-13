from __future__ import absolute_import
import numpy as np
from numba import jit, autojit, vectorize

__all__ = ['sample_response']


@autojit
def _sample_1d(pr, l, k):
    return np.bincount(
        np.random.choice(l, k, p=pr), minlength=l).astype(np.float)


@autojit
def _sample_1d_rho(pr, l, k, rho):
    return np.bincount(
        np.where(
            np.random.random(k) > rho,
            np.random.choice(l, k, p=pr), np.random.choice(l, k)),
        minlength=l).astype(np.float)


def sample_response(p, l, k, rho):
    """ Sample k responses from the given active matrix p.
    """

    if rho == 0:
        A = np.apply_along_axis(lambda x: _sample_1d(x, l, k), 1, p)
    else:
        A = np.apply_along_axis(lambda x: _sample_1d_rho(x, l, k, rho), 1, p)
    return A.astype(np.float)
