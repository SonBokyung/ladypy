from .context import np
from .agent import Agent


def sample_from_P_without_rho(P, k):
    n, m = P.shape

    rows = [np.random.choice(m, k, p=P[i, :]) for i in range(n)]

    return np.concatenate([
        np.bincount(row, minlength=m) for row in rows
    ]).reshape(n, m).astype(np.float64)


def sample_from_P(P, k, rho):
    if rho == 0:
        return sample_from_P_without_rho(P, k)

    n, m = P.shape

    rows = [
        np.where(
            np.random.rand(m) > rho,
            np.random.choice(m, m, p=P[i, :]),
            np.random.choice(m, m))
        for i in range(n)
    ]

    return np.concatenate([
        np.bincount(row, minlength=m) for row in rows
    ]).reshape(n, m).astype(np.float64)
