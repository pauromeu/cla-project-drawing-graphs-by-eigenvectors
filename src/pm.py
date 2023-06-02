import numpy as np
from numpy.linalg import norm
import random as random
from scipy.stats import ortho_group

def rayleigh_quotient(A, x):
    """Computes the Rayleigh quotient for the pair (A,x) where A is an n x n matrix and x is a n x 1 vector.
    """
    return np.dot(x,A@x) / np.dot(x,x)

def sparse_rayleigh_quotient(A_sparse, x):
    xT = x[np.newaxis, :]
    return (xT @ A_sparse @ x) / np.dot(x, x)

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
    print(u[:,:p])
    print("Power method results: ")
    print(U)
    print("Quotients power method / SVD: ")
    for j in range(p):
        print("Dominant eigenvalue ", j + 1)
        print("PM results: ", rayleigh_quotient(A,U[:,j]))
        print("SVD results  : ", rayleigh_quotient(A,u[:,j]))
        if j != p - 1: print("\n")



def stopping_criteria(x, xprev, A, tol, iters, mode=0):
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

def power_method(A, p = 2, tol = 1e-8, max_iters = 1000, prints = False, test = False, mode = 0):
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
    n = len(A[0,:])
    U = np.zeros(shape = (n,p))
    if prints: print(U)
    for k in range(p):
        print("Computing dominant eigenvector ", k + 1, "...")
        # Set initial 
        uk_t = np.random.normal(0,1,n)
        uk_t /= norm(uk_t)  # normalization
        iters = 0
        uk = np.zeros(n)
        residual = stopping_criteria(uk, uk_t, A, tol, iters, mode = mode)
        while residual > tol and iters < max_iters:
            uk = uk_t.copy()
            # orthogonalize against previous eigenvectors
            if k > 0:
                for l in range(k):
                    ul = U[:,l]
                    uk = uk - np.dot(uk, ul) / (np.dot(ul,ul)) * ul
            # multiply by A
            uk_t = A@uk
            # normalize
            uk_t = uk_t / norm(uk_t)
            iters += 1
        
        # save eigenvector
        U[:,k] = uk_t
        
    if test:
        test_power_method(A, U, p)
        
    return U
