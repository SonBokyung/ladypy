import numpy as np
#from random import *

Try = 1         # 전체 과정을 반복하는 횟수
Gen = 100       # Generations
Pop = 80        # Populaton
Nobj = 5        # number of objects
Nwrd = 5        # number of words
K = 3           # teaching parameter


def making_f(A):

    F = np.zeros(Pop)
    for i in range(Pop):
        fitness = 0.0
        for u in range(Pop):
            if i == u:
                continue
            P1 = A[i] / A[i].sum(1).reshape(Nobj, 1)
            Q1 = A[i] / A[i].sum(0)
            P2 = A[u] / A[u].sum(1).reshape(Nobj, 1)
            Q2 = A[u] / A[u].sum(0)

            fitness += (P1 * Q2 + P2 * Q1).sum() * 0.5
        F[i] = fitness / (Pop - 1)
    return F


def teaching(A, k):
    new = np.random.random((Pop, Nobj, Nwrd)) / 1000.0
    for n in range(Pop):
        for i in range(Nobj):
            p_wrd = A[n][i] / A[n][i].sum()
            js = np.random.choice(np.arange(Nobj), k, p=p_wrd)
            for j in js:
                new[n, i, j] += 1
    return new


A = np.random.random((Pop, Nobj, Nwrd))

#Original = np.copy(A)
print A.sum(0).round(1)
print
print


for one_try in range(Try):
    print 'Try : %d' % (one_try + 1); print
    # A = np.copy(Original)    # to make every Try has the same begining state
    for one_gen in range(Gen):

        F = making_f(A)  # fitness
        chosen = np.random.choice(
            (np.arange(Pop)), Pop, p=(F / F.sum()))  # compete

        A = A[chosen]
        A = teaching(A, K)  # next generation

        print(F.sum() / Pop).round(2),
    print
    print
    print A.round(2).sum(0)
    print
    print
