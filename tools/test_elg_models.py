import sys
import os

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from tqdm import tqdm, trange
from ladypy.model import ELG


def draw_simulated_graph(ax, conf):
    sim = ELG(**conf)

    for rep in trange(conf['rep'], desc='Trial'):
        A, result = sim.run()
        ax.plot(result, ls='-', alpha=.5)

    for h in range(1, min(conf['obj'], conf['sig']) + 1):
        ax.axhline(h, ls='dotted', c='k', alpha=.25)

    title = \
        ('{rep} Simulations with {pop} Agents\n'
         '{obj}x{sig} matrix P, '
         'K=({k_par}, {K_rol}, {k_rol}, {K_rnd}, {k_rnd}), '
         'rho={rho}').format(**conf)

    ax.set_title(title, fontweight='bold')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Payoff')
    ax.axis([0, conf['gen'], 0, min(conf['obj'], conf['sig']) + 1])

    return A


if __name__ == '__main__':
    fig, axes = plt.subplots(nrows=2, ncols=2)
    fig.set_size_inches(12, 10)

    for i, ax in enumerate(axes.flat):
        draw_simulated_graph(ax, {
            'rep': 20,
            'gen': 1000,
            'pop': 100,
            'obj': 5,
            'sig': 5,
            'k_par': 0,
            'K_rol': 3 * i + 1,
            'k_rol': 1,
            'K_rnd': 0,
            'k_rnd': 0,
        })

    plt.tight_layout()
    plt.savefig('output_test3.png')
