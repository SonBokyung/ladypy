import sys
mport os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib as mpl
mpl.use('Agg')

import matplotlib.pyplot as plt
from elg import simulate
from elg.data import Ksets


def draw_simulated_graph(ax, rep, gen, agent, size, ks, rho):
    pts = simulate(rep, gen, agent, size, ks, rho)

    for pt in pts:
        ax.plot(pt, ls='-', alpha=.75)

    for h in range(1, min(size) + 1):
        ax.axhline(h, ls='dotted', c='k', alpha=0.25)

    title = '%d Simulation' % rep + ('s' if rep > 1 else "") + \
            ' with %d Agent' % agent + ('s' if agent > 1 else "")
    option = '%dx%d matrix, K=%s, eps=%f, rho=%f' % \
        (size[0], size[1], ks, eps, rho)

    ax.set_title(title + '\n' + option, fontweight='bold')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Average Payoff')
    ax.axis([0, generation, 0, min(size) + 1])


if __name__ == '__main__':
    fig, axes = plt.subplots(nrows=2, ncols=2)
    fig.set_size_inches(12, 10)

    for i, ax in enumerate(axes.flat):
        draw_simulated_graph(
            ax, rep=1, gen=100, agent=100, size=(5, 5), ks=Ksets[0][i], rho=0)

    plt.tight_layout()

    output_dir = os.path.join('..', 'out')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.savefig(os.path.join(output_dir, 'output_parental.png'))
