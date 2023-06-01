import numpy as np

import numpy as np


def gram_schmidt(matrix, v):
    # Convert the matrix and vector to numpy arrays
    matrix = np.array(matrix)
    v = np.array(v)

    # Ensure matrix has 2 dimensions
    if matrix.ndim == 1:
        matrix = np.expand_dims(matrix, axis=1)

    # Check if columns are orthogonal
    for i in range(matrix.shape[1]):
        for j in range(i):
            assert np.abs(np.dot(matrix[:, i], matrix[:, j])
                          ) < 1e-8, "Columns are not orthogonal"

    # Check if p >= n
    assert matrix.shape[1] <= matrix.shape[0], "p should be lower than or equal to n"

    # Perform Gram-Schmidt orthogonalization
    for i in range(matrix.shape[1]):
        vi = matrix[:, i]
        v = v - np.dot(vi, v) / np.dot(vi, vi) * vi

    return v


# Example usage
matrix = np.array([[1, 0, 0], [0, 1, 0]]).T
v = np.array([2, 3, 4])
result = gram_schmidt(matrix, v)
print(result)
