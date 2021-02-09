import numpy as np
from scipy import sparse
from scipy.sparse.linalg import norm
from numpy.linalg import norm as dense_norm


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


def isorank_leaveout_test(A_1, R_12, alpha):
    S_exact = isorank_leaveout(A_1, R_12, alpha)
    S_iter = IsoRank(A_1, sparse.identity(R_12.shape[1]), R_12, alpha=alpha, ones_init=True, tol=1e-10).todense()
    diff = S_exact - S_iter
    print('S_exact:')
    print(S_exact)
    print('S_iter:')
    print(S_iter)
    print('Difference:')
    print(diff)
    print('Norm of S_exact:')
    print(np.linalg.norm(S_exact))
    print('Norm of S_iter:')
    print(np.linalg.norm(S_iter))
    print('Norm of diff:')
    print(np.linalg.norm(diff))



def isorank_leaveout(A_1, R_12, alpha):
    '''
    Exact solution to isorank problem where one graph is not known (I)
    '''
    print('Exact soln')
    A_1_normed = row_wise_normalize(A_1)
    I_alphaA1 = np.eye(A_1.shape[0]) - alpha*A_1_normed.todense()
    inverse = np.linalg.pinv(I_alphaA1) # trying pseudo inverse to see if that's closer to iterative solution? somehow with numerical stability?
    print('inverse')
    print(inverse)
    print('Matmul, should be identity')
    print(np.matmul(inverse, I_alphaA1))
    #assert np.all(np.matmul(inverse, I_alphaA1) == np.eye(A_1.shape[0])) # not exactly identity, but close enough I guess
    R_normalized = row_wise_normalize(R_12)
    blast =  (1 - alpha)*R_normalized.todense()
    S_12 = np.matmul(inverse, blast)

    return S_12


def IsoRank(A1, A2, R_12, alpha=0.5, maxiter=100, rand_init=False, ones_init=False, tol=1e-2, set_iterations=None):
    # all the normalizations done before actually iterating:
    '''
    A1 = row_wise_normalize(A1)
    A2 = row_wise_normalize(A2)
    R_normalized = row_wise_normalize(R_12)
    Initliazation of S to random, ones, or R_normalized
    '''
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
    print('R_12')
    print(type(R_12))
    R_normalized = row_wise_normalize(R_12)
    print('R normalized type')
    print(type(R_normalized))
    # IsoRank algorithm
    # R = R_12
    #R = sparse.lil_matrix(np.ones((n1, n2)), dtype=np.float)
    if rand_init:
        print('Creating random init matrix')
        S = sparse.csr_matrix(np.random.rand(n1, n2), dtype=np.float)
    elif ones_init:
        print('Creating ones init matrix')
        S = sparse.csr_matrix(np.ones((n1, n2)), dtype=np.float)
    else:
        print('Using normalized R as initialization')
        S = sparse.csr_matrix(R_normalized, dtype=np.float)
    #S = S/norm(S, ord=1) # normalizing S
    print('Dividing by number of entries')

    print('A1 shape ' + str(A1.shape))
    print('A2 shape ' + str(A2.shape))
    sparsity_1 = 1.0 - ( A1.count_nonzero() / float(A1.shape[0]*A1.shape[1]))
    print ("### Sparsity of the first ajacency matrix: ", str(sparsity_1))
    sparsity_2 = 1.0 - ( A2.count_nonzero() / float(A2.shape[0]*A2.shape[1]))
    print ("### Sparsity of the second ajacency matrix: ", str(sparsity_2))
    dense = False
    if sparsity_1 < 0.5 or sparsity_2 < 0.5:
        print('Using dense matrices instead; matrices are not sparse enough')
        dense = True
        #S = S/(S.shape[0]*S.shape[1])
        #S = S.todense()
        A1 = A1.todense()
        A2 = A2.todense()
        R_normalized = R_normalized.todense()
    print(A1)
    print(A2)
    A1_transpose = A1.transpose()
    print('Starting iterations')
    if set_iterations is not None:
        maxiter = set_iterations
        tol = -np.inf
    for ii in range(0, maxiter):
        print('S_prev = S')
        S_prev = S
        try:
            print('assert alpha > 0')
            assert alpha >= 0 and 1 - alpha >= 0
        except AssertionError:
            print(alpha >= 0)
            print(1 - alpha >= 0)
            print('Alpha  or 1 - alpha is negative. Exiting.')
            exit()
        print('S_dot_A2')
        S_dot_A2 = S @ A2
        print('first_term')
        first_term = alpha*A1_transpose @ S_dot_A2
        print('second_term')
        second_term = (1.0 - alpha)*R_normalized
        print('first_term + second_term')
        S = first_term + second_term
        #S = S/norm(S, ord=1)
        #S = S/np.sum(S)
        try:
            print('np.sum(S < 0) == 0 assertion')
            S_is_pos = np.sum(S < 0) == 0
            assert S_is_pos
        except AssertionError:
            print(np.sum(S < 0) == 0)
            print('S not all greater than 0. Exiting.') 
            exit()
        print('delta calculation')
        if dense:
            delta = dense_norm(S - S_prev, ord='fro')/dense_norm(S_prev, ord='fro')
            print(dense_norm(S_prev, ord='fro'))
            print(dense_norm(S, ord='fro'))
        else:
            delta = norm(S - S_prev, ord='fro')/norm(S_prev, ord='fro')
            print(norm(S_prev, ord='fro'))
            print(norm(S, ord='fro'))
            
        print ("iteration %d with delta=%f" % (ii, delta))
        if delta < tol:
            print('Converged to less than ' + str(tol))
            print(S)
            break
    print('IsoRank finished.')
    return sparse.lil_matrix(S)


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
