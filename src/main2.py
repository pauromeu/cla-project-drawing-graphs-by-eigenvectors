from graph_plot import *
from spectral_drawing import *
from graph_class import Graph
from graph_collection import *
import os
from sbm import gen_sbm_graph
from load_sbm_array import *

# Plot parameters
plot_params = [False for _ in range(n_plot_params)]
plot_params[title_index] = False
plot_params[label_index] = True
plot_params[axis_index] = True
plot_params[grid_index] = True

def generate_basic_plots():
    # Examples from the graph_collection module
    graphs = []
    G = cyclic_graph(5)
    graphs.append(G)

    G = regular_graph(100,50)
    graphs.append(G)

    G = bipartite(50,50)
    graphs.append(G)

    for G in graphs:
        draw_n(G, 1, method = "original", p = 2, tol = 1e-8, max_iter = 1000, plot_params = plot_params, mode = 0, reference = False)

def generate_walshau_plots():
    # Plots from Walshaw's collection
    graphs = []
    pwd = os.getcwd()
    f = open(pwd + '/data/3elt.graph')
    G = Graph(f)
    graphs.append(G)
    print(G)

    f = open(pwd + '/data/crack.graph')
    G = Graph(f)
    graphs.append(G)
    print(G)

    f = open(pwd + '/data/add20.txt')
    G = Graph(f)
    graphs.append(G)
    print(G)

    f = open(pwd + '/data/uk.graph')
    G = Graph(f)
    graphs.append(G)
    print(G)

    f = open(pwd + '/data/4elt.graph')
    G = Graph(f)
    graphs.append(G)
    print(G)

    for G in graphs:
        draw_n(G, 1, p = 2, tol = 1e-8, max_iter = 1000, plot_params = plot_params, mode = 0, reference = False)

def generate_sbm_plots():
    # SBM plots
    tol = 1e-10
    max_iter = 10000

    # Base parameters
    alpha = 0.99
    lbda = 0.99
    n = 500
    K = 5

    graphs = []
    G = gen_sbm_graph(alpha, lbda, n, K)
    graphs.append(G)

    Ks = [1, 2, 4, 8, 16]
    alphas = np.linspace(0.1, 0.99, num = 5)
    lbdas = np.linspace(0.9, 0.99, num = 5)

    # Increasing lbda
    for lbda in lbdas:
        G = gen_sbm_graph(alpha, lbda, n, K)
        graphs.append(G)

    # Reset base parameters
    alpha = 0.99
    lbda = 0.99
    n = 500
    K = 5

    # Increasing alpha
    for alpha in alphas:
        G = gen_sbm_graph(alpha, lbda, n, K)
        graphs.append(G)

    # Reset base parameters
    alpha = 0.99
    lbda = 0.99
    n = 500
    K = 5

    # Increasing K
    for K in Ks:
        G = gen_sbm_graph(alpha, lbda, n, K)
        graphs.append(G)

    # Saved SBM graph with 500 nodes, 5 communities and alpha, lambda = 0.99
    G = Graph()
    G.set_adj_matrix(sbm_500_5_99_99)
    G.set_name("sbm_500_5_99_99")
    graphs.append(G)

    for G in graphs:
        U = draw_n(G, 1, p = 2, method = "original", tol = 1e-8, max_iter = 2000, plot_params = plot_params, mode = 0)
        

# Uncomment any of the following functions to generate the corresponding graph drawings
# generate_basic_plots()
generate_sbm_plots()
# generate_walshau_plots()

