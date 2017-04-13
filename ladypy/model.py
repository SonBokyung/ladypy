from __future__ import division, absolute_import
import numpy as np
from .calc import derive_P_from, derive_Q_from, payoff_PQ
from .sample import sample_response

__all__ = ['ELG']


class ELG:
    def __init__(self, **kargs):
        self.set_config(**kargs)
        self.initialize()

    def set_config(self, **kargs):
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
        self.EPS = kargs.get('eps', 1e-3)
        self.RHO = kargs.get('rho', None)

    def initialize(self):
        """ Initialize agents in the model with random values.
        """
        assert isinstance(self.N_POP, int)
        assert isinstance(self.N_OBJ, int)
        assert isinstance(self.N_SIG, int)

        self.A = np.random.random((self.N_POP, self.N_OBJ, self.N_SIG))
        self.P = derive_P_from(self.A)
        self.Q = derive_Q_from(self.A)
        self.payoff = payoff_PQ(self.P, self.Q)

    def evolve(self):
        """ Generate new child epoch with the current epoch.
        """
        pr = self.payoff / self.payoff.sum()

        An = self.EPS * np.random.random((self.N_POP, self.N_OBJ, self.N_SIG))

        for i in range(self.N_POP):
            idx = np.arange(self.N_POP)

            if self.K['k_par'] > 0:
                i_prt = np.random.choice(idx, p=pr)
                idx = np.delete(idx, i_prt)
                An[i] += sample_response(self.P[i_prt], self.K['k_par'],
                                         self.RHO)

            if self.K['K_rol'] > 0:
                pr_mdl = pr[idx] / pr[idx].sum()
                i_mdls = np.random.choice(
                    idx, self.K['K_rol'], p=pr_mdl, replace=False)
                idx = np.setdiff1d(idx, i_mdls)
                for i_mdl in i_mdls:
                    An[i] += sample_response(self.P[i_mdl], self.K['k_rol'],
                                             self.RHO)

            if self.K['K_rnd'] > 0:
                i_rnds = np.random.choice(idx, self.K['K_rnd'], replace=False)
                for i_rnd in i_rnds:
                    An[i] += sample_response(self.P[i_rnd], self.K['k_rnd'],
                                             self.RHO)

        self.A = An
        self.P = derive_P_from(An)
        self.Q = derive_Q_from(An)
        self.payoff = payoff_PQ(self.P, self.Q)

    def fitness(self):
        return self.payoff.sum() / self.N_POP
