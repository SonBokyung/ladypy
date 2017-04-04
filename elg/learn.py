from .context import np
from .agent import Agent
from .sample import sample_from_P


def learn(Bs, props, ks, eps, rho):
    k_prt, k_rol, K_rol, k_rnd, K_rnd = ks
    n, m = Bs[0].P.shape

    A = np.zeros(Bs[0].P.shape)
    idx = np.arange(len(Bs))

    if k_prt > 0:
        prt = np.random.choice(idx, p=props)
        idx = np.delete(idx, prt)
        A += sample_from_P(Bs[prt].P, k_prt, rho)

    if k_rol > 0 and K_rol > 0:
        mdls = np.random.choice(idx, K_rol, p=np.array(props)[idx], replace=False)
        idx = np.setdiff1d(idx, mdls)
        for mdl in mdls:
            A += sample_from_P(Bs[mdl].P, k_rol, rho)

    if k_rnd > 0 and K_rnd > 0:
        rnds = np.random.choice(idx, K_rnd, replace=False)
        idx = np.setdiff1d(idx, rnds)
        for rnd in rnds:
            A += sample_from_P(Bs[rnd].P, k_rnd, rho)

    A += eps * np.random.rand(n, m)

    return Agent(A)
