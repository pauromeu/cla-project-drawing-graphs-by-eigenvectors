import numpy as np
from scipy.linalg import eigh


def lanczos(A, m):
    n = A.shape[0]
    V = np.zeros((n, m+1))
    T = np.zeros((m+1, m+1))

    # Initial vector
    V[:, 0] = np.random.rand(n)
    V[:, 0] = V[:, 0] / np.linalg.norm(V[:, 0])

    for j in range(m):
        V[:, j+1] = A.dot(V[:, j])
        T[j, j] = np.dot(V[:, j], V[:, j+1])
        V[:, j+1] = V[:, j+1] - T[j, j] * V[:, j]

        if j+1 < m:
            T[j+1, j] = np.linalg.norm(V[:, j+1])
            T[j, j+1] = T[j+1, j]
            if T[j+1, j] != 0:
                V[:, j+1] = V[:, j+1] / T[j+1, j]

    T = T[:m, :m]
    V = V[:, :m]

    evals_small, evecs_small = eigh(T)
    evecs = V.dot(evecs_small)

    return evals_small, evecs


# Usage
np.random.seed(32)
A = np.random.rand(5, 5)
A = A + A.T
m = 3
evals, evecs = lanczos(A, m)
print('Eigenvalues:', evals)
print('Eigenvectors:', evecs)
