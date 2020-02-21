import numpy as np
from scipy import sparse
from scipy.sparse.linalg import norm


def RWR(A, alpha=0.9, maxiter=3, tol=1e-2):
    print('RWR')
    print('Tolerance ' + str(tol))
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
    for ii in range(0, maxiter):
        prev = P
        P = alpha*A.dot(P) + (1.0 - alpha)*I
        delta = norm(P - prev, ord='fro')/norm(prev, ord='fro')
        print ("### iteration %d with delta=%0.3f" % (ii, delta))
        if delta < tol:
            break

    return P

def row_wise_normalize(mat):
    # row-wise normalization of nxn matrix
    n1 = mat.shape[0]
    with np.errstate(divide='ignore'):
        row_sums_inv = 1.0/mat.sum(axis=1)
    row_sums_inv[np.isposinf(row_sums_inv)] = 0

    row_sums_inv = np.asarray(row_sums_inv).reshape(-1)
    row_sums_inv = sparse.spdiags(row_sums_inv, 0, n1, n1)
    norm_mat = row_sums_inv.dot(mat)

    return norm_mat


def IsoRank(A1, A2, R_12, alpha=0.5, maxiter=100, rand_init=False, ones_init=False, tol=1e-2):
    print('ISORANK')
    print('Tolerance')
    print(tol)
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
    A1 = row_wise_normalize(A1)
    A2 = row_wise_normalize(A2)

    '''
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
    '''

    # ***adding this normalization:
    R_normalized = row_wise_normalize(R_12)
    # IsoRank algorithm
    # R = R_12
    #R = sparse.lil_matrix(np.ones((n1, n2)), dtype=np.float)
    if rand_init:
        S = sparse.lil_matrix(np.random.rand(n1, n2), dtype=np.float)
    elif ones_init:
        S = sparse.lil_matrix(np.ones((n1, n2)), dtype=np.float)
    else:
        print('Using normalized R as initialization')
        S = R_normalized
    #S = S/norm(S, ord=1) # normalizing S
    S = S/(S.shape[0]*S.shape[1])
    for ii in range(0, maxiter):
        S_prev = S
        try:
            assert alpha >= 0 and 1 - alpha >= 0
        except AssertionError:
            print(alpha >= 0)
            print(1 - alpha >= 0)
            print('Alpha  or 1 - alpha is negative. Exiting.')
            exit()
        S = alpha*A1.transpose().dot(S.dot(A2)) + (1.0 - alpha)*R_normalized
        #S = S/norm(S, ord=1)
        #S = S/np.sum(S)
        try:
            S_is_pos = np.sum(S < 0) == 0
            assert S_is_pos
        except AssertionError:
            print(np.sum(S < 0) == 0)
            print('S not all greater than 0. Exiting.') 
            exit()
        delta = norm(S - S_prev, ord='fro')/norm(S_prev, ord='fro')
        print(norm(S_prev, ord='fro'))
        print(norm(S, ord='fro'))
        print ("iteration %d with delta=%f" % (ii, delta))
        if delta < tol:
            print('Converged to less than ' + str(tol))
            print(S)
            break

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
