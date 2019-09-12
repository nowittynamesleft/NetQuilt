import numpy as np
from scipy import sparse


def RWR(A, alpha=0.95, maxiter=3):
    print('RWR')
    n = A.shape[0]
    with np.errstate(divide='ignore'):
        d = 1.0/A.sum(axis=1)
    d[np.isposinf(d)] = 0

    # normalize adjacency matrices
    d = np.asarray(d).reshape(-1)
    d = sparse.spdiags(d, 0, n, n)
    A = d.dot(A)

    # pagerank
    I = sparse.eye(n, dtype=np.float)
    P = I
    S = sparse.lil_matrix(np.zeros((n, n)), dtype=np.float)
    for ii in range(0, maxiter):
        P = alpha*A.dot(P) + (1.0 - alpha)*I
        S += P
        print ("### iteration %d" % (ii))

    return S


def IsoRank(A1, A2, R_12, alpha=0.5, maxiter=3, rand_init=False, ones_init=False):
    print('Iso rank')
    try:
        A1_is_pos = np.sum(A1 < 0) == 0
        A2_is_pos = np.sum(A2 < 0) == 0
        assert A1_is_pos and A2_is_pos
    except AssertionError:
        A1_is_pos = np.sum(A1 < 0) == 0
        A2_is_pos = np.sum(A2 < 0) == 0
        print(A1_is_pos)
        print(A2_is_pos)
        print('Adjacency matrices not all nonneggative. Exiting.') 
        exit()
    n1 = A1.shape[0]
    n2 = A2.shape[0]
    with np.errstate(divide='ignore'):
        d1 = 1.0/A1.sum(axis=1)
        d2 = 1.0/A2.sum(axis=1)
    d1[np.isposinf(d1)] = 0
    d2[np.isposinf(d2)] = 0

    # normalize adjacency matrices
    d1 = np.asarray(d1).reshape(-1)
    d1 = sparse.spdiags(d1, 0, n1, n1)

    d2 = np.asarray(d2).reshape(-1)
    d2 = sparse.spdiags(d2, 0, n2, n2)

    A1 = d1.dot(A1)
    A2 = d2.dot(A2)

    # IsoRank algorithm
    # R = R_12
    #R = sparse.lil_matrix(np.ones((n1, n2)), dtype=np.float)
    if rand_init:
        R = sparse.lil_matrix(np.random.rand(n1, n2), dtype=np.float)
    elif ones_init:
        R = sparse.lil_matrix(np.ones((n1, n2)), dtype=np.float)
    else:
        R = R_12
    S = R
    for ii in range(0, maxiter):
        R /= R.sum()
        try:
            assert alpha >= 0 and 1 - alpha >= 0
        except AssertionError:
            print(alpha >= 0)
            print(1 - alpha >= 0)
            print('Alpha  or 1 - alpha is negative. Exiting.')
            exit()
        R = alpha*A1.transpose().dot(R.dot(A2)) + (1.0 - alpha)*R_12
        S += R
        print ("### iteration %d" % (ii))
        try:
            R_is_pos = np.sum(R < 0) == 0
            assert R_is_pos
        except AssertionError:
            print(np.sum(R < 0) == 0)
            print('R not all greater than 0. Exiting.') 
            exit()

    return S


if __name__ == "__main__":
    A1 = np.array([[0, 1, 1, 0, 0, 0],
                   [1, 0, 1, 0, 0, 0],
                   [1, 1, 0, 1, 0, 0],
                   [0, 0, 1, 0, 1, 1],
                   [0, 0, 0, 1, 0, 1],
                   [0, 0, 0, 1, 1, 0]], dtype=np.float)
    A2 = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=np.float)
    A1 = sparse.csr_matrix(A1)
    A2 = sparse.csr_matrix(A2)
    R = sparse.lil_matrix(np.ones((6, 3), dtype=np.float))
    print (IsoRank(A1, A2, R, alpha=1.0).todense())
    print (RWR(A1).todense())
