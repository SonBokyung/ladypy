from .context import np
from .agent import Agent
from .calc import calc_payoff, calc_payoff_avg, calc_probs
from .learn import learn


def generate_random_matrix(n, m):
    return np.random.rand(n, m)


def generate_agents(N, size):
    n, m = size
    return [Agent(generate_random_matrix(n, m)) for _ in range(N)]


def simulate(rep, gen, agent, size, ks, rho, eps=1e-2):
    """ Simulate simple situation with the given parameters.
    """
    ret = []

    for r in range(rep):
        ps = generate_agents(agent, size)
        pts = [sum(calc_payoff(ps)) / agent]

        for epoch in range(gen):
            probs = calc_probs(ps)
            ps = [learn(ps, probs, ks, eps, rho) for _ in range(agent)]
            pts.append(sum(calc_payoff(ps)) / agent)

        ret.append(pts)
        print('K=%s - %dth simulation done' % (ks, r))

    return ret
