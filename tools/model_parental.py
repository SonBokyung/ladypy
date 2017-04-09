import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from ex_elg.data import Ksets
from draw import draw_simulated_graph
from tqdm import tqdm, trange

if __name__ == '__main__':
    fig, axes = plt.subplots(nrows=2, ncols=2)
    fig.set_size_inches(12, 10)

    for i in trange(len(axes.flat), desc='Chart'):
        draw_simulated_graph(
            axes.flat[i],
            rep=10,
            gen=100,
            agent=100,
            size=(5, 5),
            ks=Ksets[0][i],
            rho=0)

    plt.tight_layout()

    plt.savefig('output_parental.png')
