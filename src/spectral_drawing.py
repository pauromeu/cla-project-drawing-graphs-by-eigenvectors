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
from hde import *
import time as time
from pm import *

# Indices for plot params
grid_index = 0
axis_index = 1
label_index = 2
title_index = 3
ticks_index = 4
n_plot_params = 5

def hde_spectral_drawing(G: Graph, p: int, m: int, tol=1e-8, max_iters=1000, D_orth=True, prints=False, test=False, test_gershgorin=False, use_gershgorin=False):
    X = hde.hde(G, m, D_orth=D_orth)
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
        residual = stopping_criteria(
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
            residual = stopping_criteria(
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

def power_method_sparse(B_sparse, p, max_iter=1000, tol=1e-6, D=None):
    n = B_sparse.shape[0]
    all_ones = np.ones(n) / np.sqrt(n)
    U = np.zeros(shape=(n, n))
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
            if stopping_criteria(x, xprev, B_sparse, tol, i, mode=0) < tol:
                break
        if i == max_iter - 1:
            print("Warning: convergence not reached!")
            print("residual = ", stopping_criteria(
                x, xprev, B_sparse, tol, i, mode=0))

        xvecT = x[np.newaxis, :]
        xvec = x[:, np.newaxis]
        mu = sparse_rayleigh_quotient(B_sparse, x)
        B_sparse -= csr_matrix(mu * (xvec @ xvecT))
        U[:, k] = x
    return U


def rayleigh_iteration(A, p, tol = 1e-8, max_iter = 1000, D = None, prints = False):
    n = A.shape[0]
    id_csr = identity(n, format = 'csr')
    D = id_csr
    U = np.zeros(shape = (n,n))
    for k in range(p):
        print("Computing eigenvector ", k + 1)
        x = np.random.rand(n)
        x /= norm(x)
        x_prev = x.copy()
        if k == 0: 
            sigma = 1.0
        else:
            sigma -= 1e-4
        sigma_prev = sigma
        iters = 0
        residual = stopping_criteria(x, x_prev, A, tol, iters, mode = 0)
        while residual > tol and iters < max_iter:
            x_prev = x
            sigma_prev = sigma
            # Orthogonalize w.r.t. previous eigenvectors
            for l in range(k):
                ul = U[:,l]
                D_ul = ul.copy()
                x = x - (np.dot(x,D_ul) / np.dot(ul, D_ul)) * ul
                
            shift = A - sigma_prev * id_csr
            lu = splu(shift)
            y = lu.solve(x_prev)
            x = y / norm(y)
            sigma = np.dot(x, A@x)
            
            for l in range(k):
                ul = U[:,l]
                D_ul = ul.copy()
                x = x - (np.dot(x,D_ul) / np.dot(ul, D_ul)) * ul
            
            iters += 1
            residual = stopping_criteria(x, x_prev, A, tol, iters, mode = 0)
            
            if prints:
                if iters <= 100:
                    print("Iteration ", iters, " for eigenvector ", k + 1, " | residual = ", residual, " sigma = ", sigma)
                    print("Scalar products")
                    print(U[:,0])
                    print(x)
                    print(np.dot(x,U[:,0]))
        U[:,k] = x
    return U

def draw(G: Graph, p=2, method = "rayleigh", tol=1e-8, max_iter=1000, node_size=0.01, edge_width=0.1, figsize=(3, 3), dpi=200, mode=0, plot_params=[False for _ in range(n_plot_params)], numbering=-1, reference=False):
    # #Degree normalized eigenvectors
    if numbering != -1:
        G.set_num_name(G.name + "_" + method + "_" + str(numbering))
        
    # Select method with which we compute the drawing
    if method == "original":
        U, times, B_sparse = degree_normalized_eigenvectors(
            G, p, tol=tol, max_iter=max_iter, matmul=True, mode=mode)
    elif method == "rayleigh":
        # Construct B sparse matrix to compute eigenvectors with reference method
        D_inv_A_sparse = csr_matrix(
        np.diag(np.ones(G.n_nodes) / G.degs) @ G.adj_matrix)
        B_sparse = 0.5 * (csr_matrix(np.eye(G.n_nodes)) + D_inv_A_sparse)
        U = rayleigh_iteration(B_sparse, p + 1, tol = tol, max_iter = max_iter)
    elif method == "pm":
        # Construct B sparse matrix to compute eigenvectors with reference method
        D_inv_A_sparse = csr_matrix(
        np.diag(np.ones(G.n_nodes) / G.degs) @ G.adj_matrix)
        B_sparse = 0.5 * (csr_matrix(np.eye(G.n_nodes)) + D_inv_A_sparse)
        U = power_method_sparse(B_sparse, p, tol = tol, max_iter = max_iter, D = np.diag(G.degs))
    elif method == "ref":
        # Construct B sparse matrix to compute eigenvectors with reference method
        D_inv_A_sparse = csr_matrix(
        np.diag(np.ones(G.n_nodes) / G.degs) @ G.adj_matrix)
        B_sparse = 0.5 * (csr_matrix(np.eye(G.n_nodes)) + D_inv_A_sparse)
        eigenvalues, eigenvectors = eigs(B_sparse, k = p + 1, which='LM')
        U = eigenvectors[:,1:]

    # Save file with eigenvalue information
    filename = "files/" + G.num_name + "eigenvectors.csv"
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["u^" + str(i + 2) for i in range(p)])
        for i in range(G.n_nodes):
            writer.writerow([U[i,j] for j in range(p)])
        
    # Generate and save plot
    if method == "rayleigh" or method == "pm":
        x_coord = U[:, 1]
        y_coord = U[:, 2]
    else:    
        x_coord = U[:, 0]
        y_coord = U[:, 1]
    graph_plot(G, x_coord, y_coord, node_size = node_size, figsize = figsize, dpi = dpi, add_labels= False, edge_width = edge_width, plot_params = plot_params)
    return 1
    
def draw_from_dict(main_args):
    draw(**{key: value for arg in main_args for key, value in main_args.items()})

def draw_n(G: Graph, n: int, p=2, method = "rayleigh", tol=1e-8, max_iter=1000, node_size=0.01, edge_width=0.1, figsize=(3, 3), dpi=200, mode=0, plot_params=[False for _ in range(n_plot_params)], reference=False):
    saved_args = locals()
    main_args = {}
    for key in saved_args.keys():
        if key != 'n':
            main_args[key] = saved_args[key]

    for i in range(n):
        main_args['numbering'] = i + 1
        draw_from_dict(main_args)
