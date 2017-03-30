from typing import List, Tuple
import numpy as np


def derive_P(A: np.array) -> np.array:
    return (A.transpose() / A.sum(axis=1)).transpose()


def derive_Q(A: np.array) -> np.array:
    return (A / A.sum(axis=0)).transpose()


class Agent:
    A: np.array = None
    P: np.array = None
    Q: np.array = None

    def __init__(self, A: np.array):
        self.init(A)

    def init(self, A: np.array):
        self.A = A
        self.P = derive_P(A)
        self.Q = derive_Q(A)


def payoff(A: Agent, B: Agent) -> float:
    return 0.5 * (
        np.einsum('ij,ji->', A.P, B.Q) + np.einsum('ij,ji->', B.P, A.Q))


def total_payoff(A: Agent, Bs: List[Agent]) -> float:
    return sum([payoff(A, B) for B in Bs])


def calc_payoff(As: List[Agent]) -> List[float]:
    return [payoff(a, a) for a in As]


def calc_payoff_total(As: List[Agent]) -> List[float]:
    return [total_payoff(a, As[:i] + As[(i + 1):]) for i, a in enumerate(As)]


def calc_payoff_avg(As: List[Agent]) -> List[float]:
    l = len(As) - 1
    return [val / l for val in calc_payoff_total(As)]


def calc_probs(As: List[Agent]) -> List[float]:
    tp = calc_payoff_total(As)
    return tp / sum(tp)


# Implementation without considering mistakes or random situations as rho.
def sample_from_P(P: np.array, k: int, rho: float=0) -> np.array:
    n, m = P.shape

    return np.concatenate([
        np.bincount(np.random.choice(m, k, p=P[i, :]), minlength=m)
        for i in range(n)
    ]).reshape(n, m).astype(np.float64)


def sample_from_P_with_rho(P: np.array, k: int, rho: float) -> np.array:
    n, m = P.shape

    rows = [
        np.where(
            np.random.rand(m) > rho,
            np.random.choice(m, m, p=P[i, :]), np.random.choice(m, m))
        for i in range(n)
    ]
    return np.concatenate([np.bincount(row, minlength=m)
                           for row in rows]).reshape(n, m).astype(np.float64)


def learn(Bs: List[Agent],
          props: List[float],
          ks: Tuple[int, int, int],
          eps: float,
          rho: float) -> Agent:
    k_prt, k_rol, k_rnd = ks
    n, m = Bs[0].P.shape
    sample = sample_from_P_with_rho if rho == 0 else sample_from_P

    A = np.zeros(Bs[0].P.shape)
    idx = np.arange(len(Bs))

    if k_prt > 0:
        prt = np.random.choice(idx, p=props)
        idx = np.delete(idx, prt)
        A += sample(Bs[prt].P, k_prt, rho)

    if k_rol > 0:
        mdls = np.random.choice(idx, k_rol, p=props, replace=False)
        idx = np.setdiff1d(idx, mdls)
        for mdl in mdls:
            A += sample(Bs[mdl].P, k_rol, rho)

    if k_rnd > 0:
        rnds = np.random.choice(idx, k_rnd, replace=False)
        idx = np.setdiff1d(idx, rnds)
        for rnd in rnds:
            A += sample(Bs[rnd].P, k_rnd, rho)

    A += eps * np.random.rand(n, m)

    return Agent(A)


def generate_random_matrix(n: int, m: int) -> np.array:
    return np.random.rand(n, m)


def generate_agents(N: int, n: int, m: int) -> List[Agent]:
    return [Agent(generate_random_matrix(n, m)) for _ in range(N)]


def simulate(repeat: int,
             generation: int,
             agent: int,
             size: Tuple[int, int],
             ks: Tuple[int, int, int],
             rho: float,
             ax,
             eps: float = 1e-2):
    """ Simulate simple situation with the given parameters.
    """
    for r in range(repeat):
        ps = generate_agents(agent, *size)

        pts = [sum(calc_payoff(ps)) / agent]

        for epoch in range(generation):
            probs = calc_probs(ps)
            ps = [learn(ps, probs, ks, eps, rho) for _ in range(agent)]
            pts.append(sum(calc_payoff(ps)) / agent)

        print('K=%s - %dth simulation done' % (Ks, r))
        if ax is not None:
            ax.plot(pts, ls='-', alpha=0.75)

    if ax is not None:

        for h in range(1, min(size) + 1):
            ax.axhline(h, ls='dotted', c='k', alpha=0.25)

        title = '%d Simulation' % repeat + ('s' if repeat > 1 else "") + \
                ' with %d Agent' % agent + ('s' if agent > 1 else "")
        option = '%dx%d matrix, K=%s, eps=%f, rho=%f' % \
                (size[0], [size[1], ks, eps, rho)

        ax.set_title(title + '\n' + option, fontweight='bold')
        ax.set_xlabel('Generation')
        ax.set_ylabel('Average Payoff')
        ax.axis([0, generation, 0, min(size) + 1])


Ksets= [[(1, 0, 0), (4, 0, 0), (7, 0, 0), (10, 0, 0)],
         [(0, 1, 0), (0, 4, 0), (0, 7, 0), (0, 10, 0)],
         [(0, 0, 1), (0, 0, 4), (0, 0, 7), (0, 0, 10)]]
