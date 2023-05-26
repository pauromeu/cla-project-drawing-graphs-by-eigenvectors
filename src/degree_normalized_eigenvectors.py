import numpy as np
from numpy.linalg import norm


def degree_normalized_eigenvectors(D, A, p, tol=1e-8, max_iter=1000, prints = False):
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
    U = np.zeros((n, p + 1))
    U[:, 0] = 1.0 / np.sqrt(n)

    for k in range(1, p + 1):
        uk_t = np.random.rand(n)
        uk_t /= norm(uk_t)  # normalization
        uk = np.zeros(n)
        iter_count = 0

        while abs(np.dot(uk_t, uk)) < (1 - tol) and iter_count < max_iter:
            uk = uk_t.copy()

            # D-orthogonalize against previous eigenvectors
            for l in range(k):
                ul = U[:, l]
                uk = uk - ((uk.T @ D @ ul) / (ul.T @ D @ ul)) * ul

            # multiply with 1/2 * (I + D^-1 A)
            for i in range(n):
                neig = A[i, :] @ uk
                uk_t[i] = 0.5 * (uk[i] + neig / D[i, i])
            # uk_t = 0.5 * (uk + (A @ uk) * D_inv.diagonal()) # vectorized version

            uk_t = uk_t / norm(uk_t)  # normalization

            iter_count += 1

        if iter_count == max_iter:
            print(f"Warning: Convergence not reached for k = {k}")

        if prints: 
            print("Convergence reached for eigenvector u^",k + 1)
        U[:, k] = uk_t

    return U[:, 1:]
