import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib as mpl
mpl.use('Agg')

import matplotlib.pyplot as plt
from elg.data import Ksets
from draw import draw_simulated_graph


if __name__ == '__main__':
    fig, axes = plt.subplots(nrows=2, ncols=2)
    fig.set_size_inches(12, 10)

    for i, ax in enumerate(axes.flat):
        draw_simulated_graph(
            ax, rep=1, gen=100, agent=100, size=(5, 5), ks=Ksets[1][i], rho=0)
