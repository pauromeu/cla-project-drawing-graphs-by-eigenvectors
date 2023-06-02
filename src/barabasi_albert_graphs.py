import networkx as nx
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
