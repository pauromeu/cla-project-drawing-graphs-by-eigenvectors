import networkx as nx
import matplotlib.pyplot as plt
from graph_plot import *
from graph_class import Graph
from spectral_drawing import *


def barabasi_albert_graph(n, m, seed=None):
    # Parameters for the Barabási–Albert graph
    # n is the number of nodes
    # m is the number of edges to attach from a new node to existing nodes
    Gnx = nx.barabasi_albert_graph(n, m, seed=seed)
    A = nx.adjacency_matrix(Gnx).toarray().astype(np.int32)

    G = Graph()
    G.set_adj_matrix(A)
    G.set_degs()
    G.set_laplacian()

    return G


n = 100
m = 2

G = barabasi_albert_graph(n, m, seed=32)

U, times = degree_normalized_eigenvectors(
    np.diag(G.degs), G.laplacian, 2, tol=1e-6, max_iter=2000, matmul=False, prints=False)

x_coord = U[:, 0]
y_coord = U[:, 1]

graph_plot(G.adj_matrix, x_coord, y_coord, add_labels=False)
