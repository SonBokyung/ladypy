import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ex_elg import simulate


def draw_simulated_graph(ax, rep, gen, agent, size, ks, rho):
    pts = simulate(rep, gen, agent, size, ks, rho)

    for pt in pts:
        ax.plot(pt, ls='-', alpha=.75)

    for h in range(1, min(size) + 1):
        ax.axhline(h, ls='dotted', c='k', alpha=0.25)

    title = '%d Simulation' % rep + ('s' if rep > 1 else "") + \
            ' with %d Agent' % agent + ('s' if agent > 1 else "")
    option = '%dx%d matrix, K=%s, rho=%f' % (size[0], size[1], ks, rho)

    ax.set_title(title + '\n' + option, fontweight='bold')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Average Payoff')
    ax.axis([0, gen, 0, min(size) + 1])
