from __future__ import print_function, division, unicode_literals
import sys
import os

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from tqdm import tqdm, trange
from ladypy.model import ELG


def mk_conf(**kargs):
    return {
        'pop': kargs.get('pop', 100),
        'obj': kargs.get('obj', 5),
        'sig': kargs.get('sig', 5),
        'k_par': kargs.get('k_par', 0),
        'k_rol': kargs.get('k_rol', 0),
        'k_rnd': kargs.get('k_rnd', 0),
        'eps': kargs.get('eps', 1e-3),
        'rho': kargs.get('rho', 0)
    }


def draw_simul(ax, rep, gen, conf, flog=sys.stdout):
    mdl = ELG(**conf)

    for _ in trange(rep, desc='Trial', file=flog):
        mdl.initialize()
        dat = [mdl.fitness()]

        for _ in trange(gen, desc='Epoch', file=flog):
            mdl.evolve()
            dat.append(mdl.fitness())

        ax.plot(dat, ls='-', alpha=.5)

    for h in range(1, min(conf['obj'], conf['sig']) + 1):
        ax.axhline(h, ls='dotted', c='k', alpha=.25)

    title = \
        ('{rep} Simulations with {pop} Agents\n'
         '{obj}x{sig} matrix P, '
         'K=({k_par}, {k_rol}, {k_rnd}), '
         'rho={rho}').format(rep=rep, **conf)

    ax.set_title(title, fontweight='bold')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Payoff')
    ax.axis([0, gen, 0, min(conf['obj'], conf['sig']) + 1])


if len(sys.argv) == 1:
    print('No argument for simulation type.')
    exit(-1)

ans = int(sys.argv[1])

if not (1 <= ans <= 5):
    print('Invalid argument for simulation type.')
    exit(-1)

dir_root = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))

dir_log = os.path.join(dir_root, 'log')
if not os.path.exists(dir_log):
    os.mkdir(dir_log)

fname_log = os.path.join(dir_log, str(ans) + '.log')
flog = open(fname_log, 'w')

fig, axes = plt.subplots(nrows=2, ncols=2)
fig.set_size_inches(12, 10)

for i, ax in enumerate(tqdm(axes.flat, desc='Chart', file=flog)):
    if ans == 1:
        draw_simul(ax, 1, 100, mk_conf(k_par=3 * i + 1), flog)
    elif ans == 2:
        draw_simul(ax, 20, 1000, mk_conf(k_rol=3 * i + 1), flog)
    elif ans == 3:
        draw_simul(ax, 20, 5000, mk_conf(k_rnd=3 * i + 1), flog)
    elif ans == 4:
        draw_simul(ax, 20, 600, mk_conf(k_par=1, rho=(1e-4 * 10**i)), flog)
    else:
        draw_simul(ax, 20, 1000, mk_conf(k_rnd=1, rho=(1e-5 * 10**i)), flog)

plt.tight_layout(True)

dir_out = os.path.join(dir_root, 'out')
if not os.path.exists(dir_out):
    os.mkdir(dir_out)

fname_out = os.path.join(dir_out, str(ans) + '.png')
plt.savefig(fname_out)
flog.close()
