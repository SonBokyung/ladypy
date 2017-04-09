import numpy as np
from .calc import derive_P_from, derive_Q_from, payoff_PQ
from .sample import sample_response

__all__ = ['ELG']


def initialize_A(n_pep, n_obj, n_sig):
    """ Generate initial association matrices for simulation.
        Works as a wrapper of generating random matrix of `numpy.ndarray`.
    """
    return np.random.random(size=(n_pep, n_obj, n_sig))


class BaseSimulation:
    """ Base class for simulation.
    """
    N_REP = 10
    N_GEN = 100
    N_POP = 100
    N_OBJ = 5
    N_SIG = 5

    def __init__(self, **kargs):
        self.N_REP = kargs.get('rep', 10)
        self.N_GEN = kargs.get('gen', 100)
        self.N_POP = kargs.get('pop', 100)
        self.N_OBJ = kargs.get('obj', 5)
        self.N_SIG = kargs.get('sig', 5)


class ELG(BaseSimulation):
    def __init__(self, **kargs):
        super(ELG, self).__init__(**kargs)

    def _generate_param_K(self, **kargs):
        return {
            'parent': kargs.get('parent', 0),
            'rolemodel': kargs.get('rolemodel', 0),
            'random': kargs.get('random', 0)
        }

    def _evolute(self, P, pr, K, eps, rho):
        A = eps * initialize_A(self.N_POP, self.N_OBJ, self.N_SIG)
        idx = np.arange(self.N_POP)

        if K['parent'] > 0:
            i_prt = np.random.choice(idx, p=pr)
            idx = np.delete(idx, i_prt)
            A += sample_response(P[i_prt], K['parent'], rho)

        if K['rolemodel'] > 0:
            pr_mdl = pr[idx] / pr[idx].sum()
            i_mdls = np.random.choice(
                idx, K['rolemodel'], p=pr_mdl, replace=False)
            idx = np.setdiff1d(idx, i_mdls)
            for i_mdl in i_mdls:
                A += sample_response(P[i_mdl], 1, rho)

        if K['random'] > 0:
            i_rnds = np.random.choice(idx, K['random'], replace=False)
            for i_rnd in i_rnds:
                A += sample_response(P[i_rnd], 1, rho)

        return A

    def run(self, **kargs):
        """ Run the simulation with the given parameters K, eps, rho.
        """
        tracks = []
        K = self._generate_param_K(**kargs)
        eps = kargs.get('eps', 1e-3)
        rho = kargs.get('rho', None)

        for trial in range(self.N_REP):
            track = np.zeros(self.N_GEN + 1)

            A = initialize_A(self.N_POP, self.N_OBJ, self.N_SIG)
            P = derive_P_from(A)
            Q = derive_Q_from(A)

            po = payoff_PQ(P, Q)
            prob = po / po.sum()

            track[0] = po.sum() / self.N_POP

            for epoch in range(1, self.N_GEN + 1):
                A = self._evolute(P, prob, K, eps, rho)
                P = derive_P_from(A)
                Q = derive_Q_from(A)

                po = payoff_PQ(P, Q)
                prob = po / po.sum()

                track[epoch] = po.sum() / self.N_POP

            tracks.append(track)

        return tracks
