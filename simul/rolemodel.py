import sys
import os
sys.path.insert(0,
                os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib as mpl
mpl.use('Agg')

import matplotlib.pyplot as plt
from elg import simulate, Ksets

fig, axes = plt.subplots(nrows=2, ncols=2)
fig.set_size_inches(12, 10)

for i, ax in enumerate(axes.flat):
    simulate(
        repeat=10,
        generation=1000,
        agent=100,
        size=(5, 5),
        ks=Ksets[1][i],
        rho=0,
        ax=ax)

plt.tight_layout()

output_dir = os.path.join('..', 'out')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

plt.savefig(os.path.join(output_dir, 'output_rolemodel.png'))
