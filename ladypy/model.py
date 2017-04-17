from __future__ import division, absolute_import
import numpy as np
from .calc import derive_P_from, derive_Q_from, payoff, payoff_avg
from .sample import sample_response

__all__ = ['ELG']


class ELG:
    def __init__(self,
                 pop=100,
                 obj=5,
                 sig=5,
                 k_par=0,
                 k_rol=0,
                 k_rnd=0,
                 eps=1e-3,
                 rho=0.):
        self.N_POP = pop
        self.N_OBJ = obj
        self.N_SIG = sig
        self.K_PAR = k_par
        self.K_ROL = k_rol
        self.K_RND = k_rnd
        self.EPS = eps
        self.RHO = rho

        self.initialize()

    def initialize(self):
        """ Initialize agents in the model with random values.
        """
        self.A = np.random.random((self.N_POP, self.N_OBJ, self.N_SIG))
        self.P = derive_P_from(self.A)
        self.Q = derive_Q_from(self.A)
        self.payoff = payoff(self.P, self.Q)

    def evolve(self):
        """ Generate new child epoch with the current epoch.
        """
        pr = self.payoff / self.payoff.sum()

        An = self.EPS * np.random.random((self.N_POP, self.N_OBJ, self.N_SIG))

        for i in range(self.N_POP):
            idx = np.arange(self.N_POP)

            if self.K_PAR > 0:
                i_prt = np.random.choice(idx, p=pr)
                idx = np.delete(idx, i_prt)
                An[i] += sample_response(self.P[i_prt], self.N_SIG, self.K_PAR,
                                         self.RHO)

            if self.K_ROL > 0:
                pr_mdl = pr[idx] / pr[idx].sum()
                i_mdls = np.random.choice(
                    idx, self.K_ROL, p=pr_mdl, replace=False)
                idx = np.setdiff1d(idx, i_mdls)
                for i_mdl in i_mdls:
                    An[i] += sample_response(self.P[i_mdl], self.N_SIG, 1,
                                             self.RHO)

            if self.K_RND > 0:
                i_rnds = np.random.choice(idx, self.K_RND, replace=False)
                for i_rnd in i_rnds:
                    An[i] += sample_response(self.P[i_rnd], self.N_SIG, 1,
                                             self.RHO)

        self.A = An
        self.P = derive_P_from(An)
        self.Q = derive_Q_from(An)
        self.payoff = payoff_avg(self.P, self.Q)

    def fitness(self):
        return self.payoff.sum() / self.N_POP


class GLG(ELG):
    def __init__(self, pop=100, obj=5, sig=5, k_obs=0, k_smp=0, rho=0.):
        self.N_POP = pop
        self.N_OBJ = obj
        self.N_SIG = sig
        self.K_OBS = k_obs
        self.K_SMP = k_smp
        self.RHO = rho

        self.initialize()

    def initialize(self):
        """ Initialize agents in the model with random values.
        """
        self.A = np.random.random((self.N_POP, self.N_OBJ, self.N_SIG))
        self.P = derive_P_from(self.A)
        self.Q = derive_Q_from(self.A)
        self.D = np.random.random((self.N_POP, self.N_POP))
        self.payoff = payoff(self.P, self.Q)

    def talk(self):
        """ Generate new child epoch with the current epoch.
        """
        pr = self.payoff / self.payoff.sum()

        An = np.zeros(size=(self.N_POP, self.N_OBJ, self.N_SIG))
        Dn = np.zeros(size=(self.N_POP, self.N_POP))

        for i in range(self.N_POP):
            j_mdls = np.random.choice(self.N_POP, self.K_SMP, False, pr)

            for j in j_mdls:
                tmp = sample_response(self.P[j], self.N_SIG, self.K_OBS,
                                      self.RHO)
                An[i] += tmp * self.D[i, j]
                An[j] += tmp * self.D[j, i]

        self.A += An
        self.P = derive_P_from(self.A)
        self.Q = derive_Q_from(self.A)
        self.D += Dn
        self.payoff = payoff_avg(self.P, self.Q)

    def fitness(self):
        return self.payoff.sum() / self.N_POP
