import numpy as np
from numpy.linalg import norm
import random as random
from scipy.stats import ortho_group
from scipy.sparse import csr_matrix
from graph_class import Graph
import hde as hde
import time as time

def rayleigh_quotient(A, x):
    return np.dot(x,A@x) / np.dot(x,x)

def hde_spectral_drawing(G: Graph, p: int, m: int, tol = 1e-8, max_iters = 1000, D_orth = True, prints = False, test = False, test_gershgorin = False, use_gershgorin = False):
    X = hde.hde(G, m, D_orth = D_orth)
    if prints:
        print("prints for matrix Xss! with X shape: ", np.shape(X))
        print(X)
        print("XT D X = ")
        print(X.T@np.diag(G.degs)@X)
        print("Rank of X!", np.linalg.matrix_rank(X))
    B = hde_matrix(G.laplacian, X, use_gershgorin = use_gershgorin, test_gershgorin = test_gershgorin)
    U = power_method(B, p = p, tol = tol, max_iters = max_iters, test = test)
    if prints: print(U)
    return X@U, B, X


def gershgorin_bound(A, test = False):
    m = len(A[:,0])
    vec = np.zeros(m)
    for i in range(m):
        vec[i] = A[i,i]
        vec[i] += np.sum(np.abs(A[i,:]))
        vec[i] -= np.abs(A[i,i])
    mu = np.max(vec)
    if test:
       U = power_method(A, 1)
       max_eigvalue = rayleigh_quotient(A, U[:,0])
       print("max eig. value with PM: ", max_eigvalue)
       print("Gershgorin bound: ", mu)
       assert(mu >= np.abs(max_eigvalue) and "Gershgorin bound not bounding max. eig. value for A!")
    return np.max(vec)

def power_method(A, p = 2, tol = 1e-8, max_iters = 1000, prints = False, test = False):
    n = len(A[0,:])
    U = np.zeros(shape = (n,p))
    if prints: print(U)
    for k in range(p):
        print("Computing eigenvector ", k, "...")
        uk_t = np.random.normal(0,1,n)
        uk_t /= norm(uk_t)  # normalization
        iters = 0
        uk = np.zeros(n)
        while np.linalg.norm(uk_t - uk) > tol and iters < max_iters:
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
        
    if test:
        u, s, vt = np.linalg.svd(A)
        print("Test results: ")
        print("SVD decomposition of matrix A = ")
        print(A)
        print(":")
        print("u = ", u)
        print("s = ", s)
        print("vt = ", vt)
        print("First ", p, " dominant eigenvectors: ")
        print(u[:,:p])
        print("Power method results: ")
        print(U)
        print("Quotients power method / SVD: ")
        for j in range(p):
            print("Dominant eigenvalue ", j + 1)
            print("PM results: ", rayleigh_quotient(A,U[:,j]))
            print("SVD results  : ", rayleigh_quotient(A,u[:,j]))
            if j != p - 1: print("\n")
    return U

def hde_matrix(L, X, use_sparse = True, use_gershgorin = False, test_gershgorin = False):
    if use_sparse:
        L_sparse = csr_matrix(L)
        LX = L_sparse @ X
    else:
        LX = L @ X

    XLX = X.T @ LX
    m = len(X[0,:])
    mu = 0
    if use_gershgorin:
        mu = gershgorin_bound(XLX, test = test_gershgorin)
    else:
        epsilon = 1e-6
        V = power_method(XLX, 1)
        mu = np.abs(rayleigh_quotient(XLX, V[:,0])) * (1 + epsilon)
    B = 0.5 * (np.eye(m) -  XLX / mu)
    return B

# Test power method
b = random.uniform(0.31, 1.99)
d = [-2, b, 0.3, 0.2, 0.1]
V = ortho_group.rvs(dim = 5)
A = V@np.diag(d)@V.T
p = 2
U = power_method(A,p, test = True)
print("\n")
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
    U[:, 0] = np.ones(n) / np.sqrt(n)
    times = []
    for k in range(1, p + 1):
        print("Finding ", k, "-th eigenvector...")
        uk_t = np.random.normal(0,1,n)
        uk_t /= norm(uk_t)  # normalization
        uk = np.zeros(n)
        iter_count = 0
        iter_times = []
        while np.linalg.norm(uk - uk_t) >= tol and iter_count < max_iter:
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
