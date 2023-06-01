import numpy as np
from scipy.sparse.linalg import norm


def lanczos(A, k):
    n = A.shape[0]
    Q = np.zeros((n, k+1))
    alpha = np.zeros(k)
    beta = np.zeros(k+1)

    Q[:, 0] = np.random.rand(n)  # Random initial vector
    Q[:, 0] /= norm(Q[:, 0])  # Normalize

    for j in range(k):
        Q[:, j+1] = A @ Q[:, j]
        alpha[j] = np.dot(Q[:, j], Q[:, j+1])
        Q[:, j+1] -= beta[j] * Q[:, j]
        Q[:, j+1] -= alpha[j] * Q[:, j]

        beta[j+1] = norm(Q[:, j+1])
        Q[:, j+1] /= beta[j+1]

    T = np.diag(alpha) + np.diag(beta[:-1], 1) + np.diag(beta[:-1], -1)

    return T, Q


def implicitly_restarted_lanczos(A, k, restarts=20):
    n = A.shape[0]
    V = np.zeros((n, k * restarts))
    T = np.zeros((k * restarts, k * restarts))

    for r in range(restarts):
        Q = lanczos(A, k)
        V[:, r * k: (r + 1) * k] = Q[1]
        T[r * k: (r + 1) * k, r * k: (r + 1) * k] = Q[0]

        if r < restarts - 1:
            A = A - Q[1][:, -1].reshape(-1, 1) @ Q[0][-1, :].reshape(1, -1)

    eigenvalues, eigenvectors = np.linalg.eig(T)
    sorted_indices = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[sorted_indices][:k]
    eigenvectors = V @ eigenvectors[:, sorted_indices][:, :k]

    return eigenvalues, eigenvectors
