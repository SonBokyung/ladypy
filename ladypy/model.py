import numpy as np
from tqdm import tqdm, trange
from .calc import derive_P_from, derive_Q_from, payoff_PQ
from .sample import sample_response

__all__ = ['ELG']


def initialize_A(n_pep, n_obj, n_sig):
    """ Generate initial association matrices for simulation.
        Works as a wrapper of generating random matrix of `numpy.ndarray`.
    """
    return np.random.random(size=(n_pep, n_obj, n_sig))


class ELG:
    def __init__(self, **kargs):
        self.set_config(**kargs)

    def set_config(self, **kargs):
        self.N_GEN = kargs.get('gen', 100)
        self.N_POP = kargs.get('pop', 100)
        self.N_OBJ = kargs.get('obj', 5)
        self.N_SIG = kargs.get('sig', 5)
        self.K = {
            'k_par': kargs.get('k_par', 0),
            'K_rol': kargs.get('K_rol', 0),
            'k_rol': kargs.get('k_rol', 1),
            'K_rnd': kargs.get('K_rnd', 0),
            'k_rnd': kargs.get('k_rnd', 1)
        }
        self.eps = kargs.get('eps', 1e-3)
        self.rho = kargs.get('rho', None)

    def _evolute(self, P, pr):
        A = self.eps * initialize_A(self.N_POP, self.N_OBJ, self.N_SIG)

        for i in range(self.N_POP):
            idx = np.arange(self.N_POP)

            if self.K['k_par'] > 0:
                i_prt = np.random.choice(idx, p=pr)
                idx = np.delete(idx, i_prt)
                A[i] += sample_response(P[i_prt], self.K['k_par'], self.rho)

            if self.K['K_rol'] > 0:
                pr_mdl = pr[idx] / pr[idx].sum()
                i_mdls = np.random.choice(
                    idx, self.K['K_rol'], p=pr_mdl, replace=False)
                idx = np.setdiff1d(idx, i_mdls)
                for i_mdl in i_mdls:
                    A[i] += sample_response(P[i_mdl], self.K['k_rol'],
                                            self.rho)

            if self.K['K_rnd'] > 0:
                i_rnds = np.random.choice(idx, self.K['K_rnd'], replace=False)
                for i_rnd in i_rnds:
                    A[i] += sample_response(P[i_rnd], self.K['k_rnd'],
                                            self.rho)

        return A

    def run(self):
        """ Run the simulation with the given parameters K, eps, rho.
        """
        A = initialize_A(self.N_POP, self.N_OBJ, self.N_SIG)
        P = derive_P_from(A)
        Q = derive_Q_from(A)

        po = payoff_PQ(P, Q)
        prob = po / po.sum()

        track = np.zeros(self.N_GEN + 1)
        track[0] = po.sum() / self.N_POP

        for epoch in trange(1, self.N_GEN + 1, desc='Epoch'):
            A = self._evolute(P, prob)
            P = derive_P_from(A)
            Q = derive_Q_from(A)

            po = payoff_PQ(P, Q)
            prob = po / po.sum()

            track[epoch] = po.sum() / self.N_POP

        return A, track
