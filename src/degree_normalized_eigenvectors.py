import numpy as np
from numpy.linalg import norm
import random as random
from scipy.stats import ortho_group
from scipy.sparse import csr_matrix
from graph_class import Graph
import hde as hde
import time as time

def hde_spectral_drawing(G: Graph, p: int, m: int, tol = 1e-8, max_iters = 1000, D_orth = True, prints = False):
    X = hde.hde(G, m, D_orth = D_orth)
    B = hde_matrix(G.laplacian, X)
    U = power_method(B, p = p, tol = tol, max_iters = max_iters)
    if prints: print(U)
    return X@U


def gershgorin_bound(A):
    m = len(A[:,0])
    vec = np.zeros(m)
    for i in range(m):
        vec[i] = A[i,i]
        vec[i] += np.sum(np.abs(A[i,:]))
        vec[i] -= np.abs(A[i,i])
    return np.max(vec)

def power_method(A, p = 2, tol = 1e-8, max_iters = 1000, prints = False):
    n = len(A[0,:])
    U = np.zeros(shape = (n,p))
    if prints: print(U)
    for k in range(p):
        print("Computing eigenvector ", k, "...")
        uk_t = np.random.rand(n)
        uk_t /= norm(uk_t)  # normalization
        iters = 0
        uk = np.zeros(n)
        while np.dot(uk_t, uk) < 1 - tol and iters < max_iters:
            uk = uk_t.copy()
            # orthogonalize against previous eigenvectors
            if k > 0:
                for l in range(k):
                    uk = uk - np.dot(uk, U[:,l]) / (np.linalg.norm(U[:,l])**2) * U[:,l]
            
            # multiply by A
            uk_t = A@uk
            
            # normalize
            uk_t = uk_t / np.linalg.norm(uk_t)
            iters += 1
        
        # save eigenvector
        U[:,k] = uk_t
    return U

def hde_matrix(L, X):
    L_sparse = csr_matrix(L)
    LX = L_sparse @ X
    XLX = X.T @ LX
    m = len(X[0,:])
    mu = gershgorin_bound(XLX)
    B = mu * np.eye(m) -  XLX
    return B

# # Test power method
# b = random.uniform(0.31, 1.99)
# d = [-2, b, 0.3, 0.2, 0.1]
# V = ortho_group.rvs(dim = 5)
# A = V@np.diag(d)@V.T
# p = 2
# U = power_method(A,p)
# for j in range(p):
#     print("Dominant eigenvalue ", j + 1)
#     print("Eigenvalue results: ", A@U[:,j] / U[:,j])
#     print("Expected results  : ", A@V[:,j] / V[:,j])
#     if j != p - 1: print("\n")

# # Test hde matrix
# pwd = "/Users/guifre/cla_project/cla-project-drawing-graphs-by-eigenvectors"
# f = open(pwd + "/data/add20.txt", "r")
# G = Graph(f)
# m = 4
# X = hde.hde(G, m)
# B = hde_matrix(G.laplacian, X)
# print(B)



                    
    
    
    

# # Test case for Gershgorin bound
# # Given matrix
# matrix = np.array([[3, 1, 2, 4],
#                    [0, 6, 2, 1],
#                    [1, 0, 4, 2],
#                    [2, 1, 0, 5]])

# print(gershogorin_bound(matrix))


def degree_normalized_eigenvectors(D, A, p, tol=1e-6, max_iter=2000, matmul = False, prints = False):
    """
    Compute the top non-degenerate eigenvectors of the degree-normalized adjacency matrix of a graph.

    This function uses the power method with D-orthogonalization to iteratively compute the eigenvectors.

    Parameters
    ----------
    D : numpy.ndarray
        The degree matrix of the graph. Should be a square matrix of size n x n.
    A : numpy.ndarray
        The adjacency matrix of the graph. Should be a square matrix of size n x n.
    p : int
        The number of eigenvectors to compute.
    tol : float, optional
        The tolerance for convergence of the power method. The default is 1e-8.
    max_iter : int, optional
        The maximum number of iterations for the power method. The default is 1000.

    Returns
    -------
    U[:, 1:] : numpy.ndarray
        The computed eigenvectors, excluding the first column of the matrix U which is the trivial eigenvector.

    Raises
    ------
    Warning
        If the power method does not converge within the specified number of iterations, a warning is printed.

    """
    n = D.shape[0]
    D_diag = D.diagonal()
    D_inv_diag = np.ones(n) / D_diag
    U = np.zeros((n, p + 1))
    U[:, 0] = 1.0 / np.sqrt(n)
    times = []
    for k in range(1, p + 1):
        print("Finding ", k, "-th eigenvector...")
        uk_t = np.random.rand(n)
        uk_t /= norm(uk_t)  # normalization
        uk = np.zeros(n)
        iter_count = 0
        iter_times = []
        while abs(np.dot(uk_t, uk)) < (1 - tol) and iter_count < max_iter:
            t_iter_0 = time.time()
            uk = uk_t.copy()
            t_for_k_0 = time.time()
            # D-orthogonalize against previous eigenvectors
            for l in range(k):
                ul = U[:, l]
                D_ul = D_diag * ul
                uk = uk - np.dot(uk, D_ul) / np.dot(ul, D_ul) * ul
                # old code:
                # uk = uk - ((uk.T @ D @ ul) / (ul.T @ D @ ul)) * ul
            t_for_k_1 = time.time()

            # multiply with 1/2 * (I + D^-1 A)
            t_matmul_0 = time.time()
            if not matmul:
                for i in range(n):
                    neig = A[i, :] @ uk
                    uk_t[i] = 0.5 * (uk[i] + neig / D[i, i])
            else:
                if prints: print("computing with matrix multiplication")
                uk_t = 0.5 * (uk + (A @ uk) * D_inv_diag)
            t_matmul_1 = time.time()
            # uk_t = 0.5 * (uk + (A @ uk) * D_inv.diagonal()) # vectorized version

            uk_t = uk_t / norm(uk_t)  # normalization

            iter_count += 1
            t_iter_1 = time.time()
            
            iter_times.append([t_iter_1 - t_iter_0, t_for_k_1 - t_for_k_0, t_matmul_1 - t_matmul_0])
        times.append(iter_times)

        if iter_count == max_iter:
            print(f"Warning: Convergence not reached for k = {k}")
            print(f"1 - product = ", 1 - abs(np.dot(uk_t, uk)))
        else:
            print("Convergence reached for eigenvector u^",k + 1)
        U[:, k] = uk_t

    return U[:, 1:], times
