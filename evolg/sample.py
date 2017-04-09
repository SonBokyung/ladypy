import numpy as np

__all__ = ['sample_response']


def _sample_1d(pr, l, k):
    return np.bincount(
        np.random.choice(l, k, p=pr), minlength=l).astype(np.float)


def _sample_1d_rho(pr, l, k, rho):
    return np.bincount(
        np.where(
            np.random.random(k) > rho,
            np.random.choice(l, k, p=pr),
            np.random.choice(l, k)),
        minlength=l).astype(np.float)


def sample_response(p, k=1, rho=None):
    """ Sample k responses from the given active matrix p.
    """
    l = p.shape[1]
    if rho is None:
        A = np.apply_along_axis(lambda x: _sample_1d(x, l, k), 1, p)
    else:
        A = np.apply_along_axis(lambda x: _sample_1d_rho(x, l, k, rho), 1, p)
    return A.astype(np.float)
