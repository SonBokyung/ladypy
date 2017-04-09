import sys
import os
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
from elgpy.simulate import ELG

N_REP = 20
N_GEN = 100
N_POP = 100
N_OBJ = 5
N_SIG = 5


def draw_simulated_graph(ax, ks, rho):
    sim = ELG(rep=N_REP, gen=N_GEN, pop=N_POP, obj=N_OBJ, sig=N_SIG)
    results = sim.run(parent=ks[0], rolemodel=ks[1], random=ks[2], rho=rho)

    for result in results:
        ax.plot(result, ls='-', alpha=.5)

    for h in range(1, min(N_OBJ, N_SIG) + 1):
        ax.axhline(h, ls='dotted', c='k', alpha=.25)

    title = '{rep} Simulations with {pop} Agents'.format(rep=N_REP, pop=N_POP)
    option = '{}x{} matrix P, K=({}, {}, {}), rho={}'.format(
        N_OBJ, N_SIG, ks[0], ks[1], ks[2], rho)

    ax.set_title(title + '\n' + option, fontweight='bold')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Payoff')
    ax.axis([0, N_GEN, 0, min(N_OBJ, N_SIG) + 1])


if __name__ == '__main__':
    fig, axes = plt.subplots(nrows=2, ncols=2)
    fig.set_size_inches(12, 10)

    for i in trange(len(axes.flat), desc='Chart'):
        draw_simulated_graph(axes.flat[i], [3 * i + 1, 3 * i + 1, 0, 0], 0)

    plt.tight_layout()

    plt.savefig('output_test3.png')
