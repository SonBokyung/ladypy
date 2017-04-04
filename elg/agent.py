from .context import np
from .learn import derive_P, derive_Q


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
