import numpy as np

__all__ = ['sample']


def _sample_1d(p, l, k):
    return np.bincount(
        np.random.choice(l, k, p=p), minlength=l).astype(np.float)


def _sample_1d_rho(p, l, k, rho):
    return np.bincount(
        np.where(
            np.random.random(l) > rho,
            np.random.choice(l, k, p=p),
            np.random.choice(l, k)),
        minlength=l).astype(np.float)


def sample(P, k=1, rho=None):
    l = P.shape[2]
    if rho is None:
        A = np.apply_along_axis(lambda x: _sample_1d(x, l, k), 2, P)
    else:
        A = np.apply_along_axis(lambda x: _sample_1d_rho(x, l, k, rho), 2, P)
    return A.astype(np.float)
