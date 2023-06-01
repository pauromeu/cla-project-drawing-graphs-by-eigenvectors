import mpmath as mp
import csv
from scipy.sparse import identity
from scipy.sparse.linalg import splu
from graph_plot import *
import numpy as np
from numpy.linalg import norm
import random as random
from scipy.stats import ortho_group
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs
from graph_class import Graph
import hde as hde
import time as time
from pm import *
<< << << < HEAD
== == == =
>>>>>> > c108cbacb9b15a5a1d78b13580ecc6b978cadcb9

# Set the desired precision
# mp.dps = 100


def rayleigh_quotient(A, x):
    return np.dot(x, A@x) / np.dot(x, x)


<< << << < HEAD


def hde_spectral_drawing(G: Graph, p: int, m: int, tol=1e-8, max_iters=1000, D_orth=True, prints=False, test=False, test_gershgorin=False, use_gershgorin=False):
    X = hde.hde(G, m, D_orth=D_orth)


== == == =


def sparse_rayleigh_quotient(A_sparse, x):
    xT = x[np.newaxis, :]
    return (xT @ A_sparse @ x) / np.dot(x, x)


def hde_spectral_drawing(G: Graph, p: int, m: int, tol=1e-8, max_iters=1000, D_orth=True, prints=False, test=False, test_gershgorin=False, use_gershgorin=False):
    X = hde.hde(G, m, D_orth=D_orth)


>>>>>> > c108cbacb9b15a5a1d78b13580ecc6b978cadcb9
   if prints:
        print("prints for matrix Xss! with X shape: ", np.shape(X))
        print(X)
        print("XT D X = ")
        print(X.T@np.diag(G.degs)@X)
        print("Rank of X!", np.linalg.matrix_rank(X))
    B = hde_matrix(G.laplacian, X, use_gershgorin=use_gershgorin,
                   test_gershgorin=test_gershgorin)
    U = power_method(B, p=p, tol=tol, max_iters=max_iters, test=test)
    if prints:
        print(U)
    return X@U, B, X


def gershgorin_bound(A, test=False):
    m = len(A[:, 0])
    vec = np.zeros(m)
    for i in range(m):
        vec[i] = A[i, i]
        vec[i] += np.sum(np.abs(A[i, :]))
        vec[i] -= np.abs(A[i, i])
    mu = np.max(vec)
    if test:
        U = power_method(A, 1)
        max_eigvalue = rayleigh_quotient(A, U[:, 0])
        print("max eig. value with PM: ", max_eigvalue)
        print("Gershgorin bound: ", mu)
        assert (mu >= np.abs(max_eigvalue)
                and "Gershgorin bound not bounding max. eig. value for A!")
    return np.max(vec)


def test_power_method(A, U, p):
    """ Tests power method results using SVD decomposition

    Args:
        A: matrix for which power method has been applied
        U: results of power method
        p: number of eigenvectors retrieved using power method
    """
    u, s, vt = np.linalg.svd(A)
    print("Test results: ")
    print("SVD decomposition of matrix A = ")
    print(A)
    print(":")
    print("u = ", u)
    print("s = ", s)
    print("vt = ", vt)
    print("First ", p, " dominant eigenvectors: ")
    print(u[:, :p])
    print("Power method results: ")
    print(U)
    print("Quotients power method / SVD: ")
    for j in range(p):
        print("Dominant eigenvalue ", j + 1)
        print("PM results: ", rayleigh_quotient(A, U[:, j]))
        print("SVD results  : ", rayleigh_quotient(A, u[:, j]))
        if j != p - 1:
            print("\n")


def stopping_criteria_pm(x, xprev, A, tol, iters, mode=0):
    """Computes residual for different stopping criteria for the power method
    """
    if iters != 0:
        residual = 0
        if mode == 0:
            residual = 1. - np.dot(x, xprev)
        elif mode == 1:  # computing the residual
            residual = norm(A@x - rayleigh_quotient(A, x)*x)
        elif mode == 2:
            residual = norm(x - xprev)
        return residual
    else:
        return 99999


def power_method(A, p=2, tol=1e-8, max_iters=1000, prints=False, test=False, mode=0):
    """ Basic implementation of power method to find the first p dominant eigenvectors of matrix A

    Args:
        A: matrix for which we want to compute the first p dominant eigenvectors
        p (int, optional): Defaults to 2.
        tol (_type_, optional): tolerance for the stopping criteria of the power method. Defaults to 1e-8.
        max_iters (int, optional): maximum number of iterations. Defaults to 1000.
        prints (bool, optional): if True provides useful prints for debugging. Defaults to False.
        test (bool, optional): if True, tests immediately the results of the power method comparing with SVD results. Defaults to False.
        mode: residual mode (see stopping_criteria_pm function for more info.). Defaults to 0.

    Returns:
        U: n x p matrix containing the first p dominant eigenvectors of matrix A
    """
    n = len(A[0, :])
    U = np.zeros(shape=(n, p))
    if prints:
        print(U)
    for k in range(p):
        print("Computing dominant eigenvector ", k + 1, "...")
        # Set initial
        uk_t = np.random.normal(0, 1, n)
        uk_t /= norm(uk_t)  # normalization
        iters = 0
        uk = np.zeros(n)
        residual = stopping_criteria_pm(uk, uk_t, A, tol, iters, mode=mode)
        while residual > tol and iters < max_iters:
            uk = uk_t.copy()
            # orthogonalize against previous eigenvectors
            if k > 0:
                for l in range(k):
                    ul = U[:, l]
                    uk = uk - np.dot(uk, ul) / (np.dot(ul, ul)) * ul
            # multiply by A
            uk_t = A@uk
            # normalize
            uk_t = uk_t / norm(uk_t)
            iters += 1

        # save eigenvector
        U[:, k] = uk_t

    if test:
        test_power_method(A, U, p)

    return U


def hde_matrix(L, X, use_sparse=True, use_gershgorin=False, test_gershgorin=False):
    """Computes matrix from which the eigenvectors for the p-dimensional layout should be extracted.
    In the reference [2] this matrix is just mu * In - X^T L X, where X represents the high-dimensional
    embedding of the graph. mu in the previous expression should be a bound on max_i |lambda_i|, where 
    lambda_i are the eigenvalues of X^T L X. This bound can be computed using the so-called Gershgorin
    bound, or by using the power-method on X^T L X to get the dominant eigenvalue.

    Args:
        L: the laplacian of the graph
        X: the matrix encoding the high-dimensional embedding
        use_sparse (bool, optional): if True, exploits the sparsity of L to compute the product LX. Defaults to True.
        use_gershgorin (bool, optional): if True, we set mu equal to the Gershgorin bound for X^T L X. Defaults to False.
        test_gershgorin (bool, optional): if True, tests if the Gershgorin bound actually bounds the dominant eigenvector of X^T L X. Defaults to False.

    Returns:
        B: matrix mu * In - X^T L X
    """
    if use_sparse:
        L_sparse = csr_matrix(L)
        LX = L_sparse @ X
    else:
        LX = L @ X

    XLX = X.T @ LX
    m = len(X[0, :])
    mu = 0
    if use_gershgorin:
        mu = gershgorin_bound(XLX, test=test_gershgorin)
    else:
        epsilon = 1e-6
        V = power_method(XLX, 1)
        mu = np.abs(rayleigh_quotient(XLX, V[:, 0])) * (1 + epsilon)
    B = 0.5 * (np.eye(m) - XLX / mu)
    return B


def degree_normalized_eigenvectors(G, p, tol=1e-6, max_iter=2000, matmul=True, prints=False, mode=0):
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
    A = G.adj_matrix
    D = np.diag(G.degs)
    n = D.shape[0]
    adj_list = G.adj_list
    degs = G.degs
    D_diag = D.diagonal()
    D_inv_diag = np.ones(n) / D_diag
    D_inv_sparse = csr_matrix(np.diag(D_inv_diag))
    A_sparse = csr_matrix(A)
    D_inv_A_sparse = D_inv_sparse@A_sparse
    B_sparse = 0.5 * (csr_matrix(np.eye(G.n_nodes)) + D_inv_A_sparse)
    U = np.zeros((n, p + 1))
    U[:, 0] = np.ones(n) / np.sqrt(n)
    B = 0.5 * (np.eye(G.n_nodes) + np.diag(D_inv_diag)@G.adj_matrix)
    times = []
    for k in range(1, p + 1):
        print("Finding ", k, "-th eigenvector...")
        uk_t = np.random.normal(0, 1, n)
        uk_t /= norm(uk_t)  # normalization
        uk = np.zeros(n)
        iter_count = 0
        iter_times = []
        residual = stopping_criteria_pm(
            uk, uk_t, B, tol, iter_count, mode=mode)
        while residual >= tol and iter_count < max_iter:
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
                # print("Not matmul!")
                for i in range(n):
                    uk_t[i] = 0.5 * (uk[i] + np.sum(uk[adj_list[i]]) / degs[i])
                # uk_t = 0.5 * (uk_t + uk)
                # for i in range(n):
                #     adj_list_i = adj_list[i]
                #     neig = np.dot(A[i,adj_list_i], uk[adj_list_i])
                #     uk_t[i] = 0.5 * (uk[i] + neig / D[i, i])
            else:
                if prints:
                    print("computing with matrix multiplication")
                uk_t = B @ uk
            t_matmul_1 = time.time()
            # uk_t = 0.5 * (uk + (A @ uk) * D_inv.diagonal()) # vectorized version

            uk_t = uk_t / norm(uk_t)  # normalization
            residual = stopping_criteria_pm(
                uk, uk_t, B, tol, iter_count, mode=mode)
            iter_count += 1
            t_iter_1 = time.time()

            iter_times.append(
                [t_iter_1 - t_iter_0, t_for_k_1 - t_for_k_0, t_matmul_1 - t_matmul_0])
        times.append(iter_times)

        if iter_count == max_iter:
            print(f"Warning: Convergence not reached for k = {k}")
            print(f"last residual = ", residual)
        else:
            print("Convergence reached for eigenvector u^", k + 1)

        # D-orthogonalize against previous eigenvectors
        for l in range(k):
            ul = U[:, l]
            D_ul = D_diag * ul
            uk_t = uk_t - np.dot(uk_t, D_ul) / np.dot(ul, D_ul) * ul
        uk_t = uk_t / norm(uk_t)  # normalize again
        U[:, k] = uk_t

    return U[:, 1:], times, B_sparse


grid_index = 0
axis_index = 1
label_index = 2
title_index = 3
ticks_index = 4
n_plot_params = 5


def power_method_sparse(B_sparse, p, max_iter=1000, tol=1e-6, D = None):
    n = B_sparse.shape[0]
    all_ones = np.ones(n) / np.sqrt(n)
    U = np.zeros(shape= (n, n))
    if D.all() == None:
        D = csr_matrix(np.eye(n))
    U[:, 0] = all_ones.copy()
    for k in range(1, p + 1):
        x = np.random.rand(n)
        if k > 0:
            for l in range(k - 1):
                ul = D@U[:, l]
                x -= np.dot(x, ul) / np.dot(ul, ul) * ul
        x /= norm(x)
        for i in range(max_iter):
            xprev = x
            y = B_sparse@x
            if k > 0:
                for l in range(k):
                    ul = U[:, l]
                    y -= np.dot(y, ul) / np.dot(ul, ul) * ul
            x = y / norm(y)
            if stopping_criteria_pm(x, xprev, B_sparse, tol, i, mode= 0) < tol:
                break
        if i == max_iter - 1:
            print("Warning: convergence not reached!")
            print("residual = ", stopping_criteria_pm(x, xprev, B_sparse, tol, i, mode= 0))

        xvecT = x[np.newaxis, :]
        xvec = x[:, np.newaxis]
        mu = sparse_rayleigh_quotient(B_sparse, x)
        B_sparse -= csr_matrix(mu * (xvec @ xvecT))
        U[:, k] = x
    return U[:, 1:]


    print(n)
    return 0

    return eigenvectors


def rayleigh_iteration(A, p, tol = 1e-8, max_iter = 1000, D = None):
    n = A.shape[0]
    id_csr = identity(n, format= 'csr')
    U = np.zeros(shape = (n, n))
    for k in range(p):
        x = np.random.rand(n)
        x /= norm(x)
        x_prev = x
        x = np.zeros(n)
        iters = 0
        residual = stopping_criteria_pm(x, x_prev, A, tol, iters, mode= 0)
        while residual > tol and iters < max_iter:
            # Orthogonalize w.r.t. previous eigenvectors
            for l in range(k):
                ul = U[:, l]
                D_ul = np.diagonal(D)*ul
                x -= np.dot(x, D_ul) / np.dot(ul, D_ul) * ul

            sigma = np.dot(x_prev, A@x_prev)
            shift = A - id_csr
            lu = splu(shift)
            y = lu.solve(x_prev)
            x = y / norm(y)
            iters += 1
            residual = stopping_criteria_pm(x, x_prev, A, tol, iters, mode= 0)
        U[:, k] = x
        # Deflate original matrix A
    return U





def draw(G: Graph, p = 2, tol = 1e-8, max_iter = 1000, node_size = 0.01, edge_width = 0.1, figsize = (3, 3), dpi = 200, mode = 0, plot_params = [False for _ in range(n_plot_params)], numbering = -1, reference = False):
    # #Degree normalized eigenvectors
    if numbering != -1:
        G.set_num_name(G.name + "_" + str(numbering))
    # U, times, B_sparse = degree_normalized_eigenvectors(G, p, tol = tol, max_iter = max_iter, matmul = True, mode = mode)

    # # Save file with eigenvalue information
    # filename = "files/" + G.num_name + "eigenvectors.csv"
    # with open(filename, 'w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(["u^" + str(i + 2) for i in range(p)])
    #     for i in range(G.n_nodes):
    #         writer.writerow([U[i,j] for j in range(p)])
    D_inv_A_sparse = csr_matrix(
        np.diag(np.ones(G.n_nodes) / G.degs) @ G.adj_matrix)
    B_sparse = 0.5 * (csr_matrix(np.eye(G.n_nodes)) + D_inv_A_sparse)
    # U = power_method_sparse(B_sparse, p, tol = tol, max_iter = max_iter, D = np.diag(G.degs))
    # approximate_eigenvalues = np.array([sparse_rayleigh_quotient(B_sparse, U[:, i]) for i in range(p)])
    # filename = "files/" + G.num_name + "eigenvalues.csv"
    # with open(filename, 'w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(["lambda_" + str(i + 2) for i in range(p)])
    #     writer.writerow(approximate_eigenvalues)

    # Eigenvalues and eigenvectors with reference method
    filename = "files/" + G.num_name + "eigenvectors.csv"
    if reference:
        eigenvalues, eigenvectors = eigs(B_sparse, k= p + 1, which='LM')
        with open(filename, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(eigenvalues[1:])

        x_coord = eigenvectors[:, 1]
        y_coord = eigenvectors[:, 2]
        plot_name = G.num_name + "_ref"
        graph_plot(G, x_coord, y_coord, node_size= node_size, figsize = figsize, dpi = dpi, add_labels= False, edge_width = edge_width, plot_params = plot_params, plot_name = plot_name)

    # x_coord = U[:, 0]
    # y_coord = U[:, 1]
    # graph_plot(G, x_coord, y_coord, node_size = node_size, figsize = figsize, dpi = dpi, add_labels= False, edge_width = edge_width, plot_params = plot_params)
    # return 1


def draw_from_dict(main_args):
    draw(**{key: value for arg in main_args for key, value in main_args.items()})

def draw_n(G: Graph, n: int, p = 2, tol = 1e-8, max_iter = 1000, node_size = 0.01, edge_width = 0.1, figsize = (3, 3), dpi = 200, mode = 0, plot_params = [False for _ in range(n_plot_params)], reference = False):
    saved_args = locals()
    main_args = {}
    for key in saved_args.keys():
        if key != 'n':
            main_args[key] = saved_args[key]

    for i in range(n):
        main_args['numbering'] = i + 1
        draw_from_dict(main_args)
