from __future__ import print_function, division, unicode_literals
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from tqdm import tqdm, trange
from ladypy.model import ELG


def generate_conf(**kargs):
    return {
        'rep': kargs.get('rep', 20),
        'gen': kargs.get('gen', 1000),
        'pop': kargs.get('pop', 100),
        'obj': kargs.get('obj', 5),
        'sig': kargs.get('sig', 5),
        'k_par': kargs.get('k_par', 0),
        'K_rol': kargs.get('K_rol', 0),
        'k_rol': kargs.get('k_rol', 0),
        'K_rnd': kargs.get('K_rnd', 0),
        'k_rnd': kargs.get('k_rnd', 0),
        'eps': kargs.get('eps', 1e-3),
        'rho': kargs.get('rho', 0)
    }


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
        draw_simulated_graph(ax, generate_conf(gen=100, k_par=3 * i + 1))

    plt.tight_layout()
    plt.savefig('output_1.png')

    fig, axes = plt.subplots(nrows=2, ncols=2)
    fig.set_size_inches(12, 10)

    for i, ax in enumerate(axes.flat):
        draw_simulated_graph(ax, generate_conf(K_rol=3 * i + 1, k_rol=1))

    plt.tight_layout()
    plt.savefig('output_2.png')

    fig, axes = plt.subplots(nrows=2, ncols=2)
    fig.set_size_inches(12, 10)

    for i, ax in enumerate(axes.flat):
        draw_simulated_graph(ax, generate_conf(K_rnd=3 * i + 1, k_rnd=1))

    plt.tight_layout()
    plt.savefig('output_3.png')
