from .context import np
from .agent import Agent


def payoff(A, B):
    return 0.5 * (np.einsum('ij,ji->', A.P, B.Q) +
                  np.einsum('ij,ji->', B.P, A.Q))


def total_payoff(A, Bs):
    return sum([payoff(A, B) for B in Bs])


def calc_payoff(As):
    return [payoff(a, a) for a in As]


def calc_payoff_total(As):
    return [total_payoff(a, As[:i] + As[(i + 1):]) for i, a in enumerate(As)]


def calc_payoff_avg(As):
    l = len(As) - 1
    return [val / l for val in calc_payoff_total(As)]


def calc_probs(As):
    tp = calc_payoff_total(As)
    return tp / sum(tp)
