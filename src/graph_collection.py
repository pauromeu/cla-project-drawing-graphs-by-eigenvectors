import numpy as np
from graph_class import Graph

def cyclic_graph(n):
    """Returns cyclic graph with n nodes
    """
    A = np.zeros(shape = (n,n))
    for i in range(n - 1):
        A[i, i + 1] = 1
    A = A + A.T
    # Connect endpoints!
    A[0, n - 1] = 1
    A[n - 1, 0] = 1
    G = Graph()
    G.set_adj_matrix(A)
    name = "cyclic_" + str(n)
    G.set_name(name)
    return G

def regular_graph(n, k):
    """Returns regular graph with n nodes and degree k
    """
    A = np.zeros(shape = (n,n))
    for i in range(n):
        if k % 2 == 0:
            for j in range(1, int(k / 2) + 1):
                A[i, (i + j) % n] = 1
                A[i, (i - j) % n] = 1
        else:
            for j in range(1, int((k - 1) / 2) + 1):
                A[i, (i + j) % n] = 1
                A[i, (i - j) % n] = 1
            A[i, (i + int(n / 2)) % n] = 1
    print(A)
    G = Graph()
    G.set_adj_matrix(A)
    name = "regular_n" + str(n) + "_k" + str(k)
    G.set_name(name)
    return G

def bipartite(p, q):
    """Bipartite graph with 2 groups of nodes, with p and q nodes respectively.
    """
    B = np.ones(shape = (p,q))
    L = np.zeros(shape = (p,p))
    R = np.zeros(shape = (q,q))
    A = np.concatenate((np.concatenate((L, B), axis = 1), np.concatenate((B.T, R), axis = 1)), axis = 0)
    G = Graph()
    print(A)
    G.set_adj_matrix(A)
    name = "bipartite_p" + str(p) + "_q" + str(q) 
    G.set_name(name)
    return G