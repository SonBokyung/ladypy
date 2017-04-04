from .context import np

def derive_P(A):
    return (A.transpose() / A.sum(axis=1)).transpose()


def derive_Q(A):
    return (A / A.sum(axis=0)).transpose()


class Agent:
    A = None
    P = None
    Q = None

    def __init__(self, A):
        self.init(A)

    def init(self, A):
        self.A = A
        self.P = derive_P(A)
        self.Q = derive_Q(A)
