import numpy as np

__all__ = ['Simulation', 'ELGSimulation']


class Simulation:
    """ Base class for simulation.
    """
    N_REP = 10
    N_GEN = 100
    N_POP = 100
    N_OBJ = 5
    N_SIG = 5

    def __init__(self, **kargs):
        N_REP = kargs.get('rep', 10)
        N_GEN = kargs.get('gen', 100)
        N_POP = kargs.get('pop', 100)
        N_OBJ = kargs.get('obj', 5)
        N_SIG = kargs.get('sig', 5)


class ELGSimulation(Simulation):
    def __init__(self, **kargs):
        super(ELGSimulation, self).__init__(**kargs)

    def run(self):
        pass
