from graph_plot import *
from spectral_drawing import *
from graph_class import Graph
from graph_collection import *
import os
from sbm import gen_sbm_graph
from load_sbm_array import *
plot_params = [False for _ in range(n_plot_params)]
plot_params[title_index] = False
plot_params[label_index] = True
plot_params[axis_index] = True
plot_params[grid_index] = True
# pwd = "/Users/guifre/cla_project/cla-project-drawing-graphs-by-eigenvectors"
# f = open(pwd + '/data/crack.graph')
# G = Graph(f)
# print(G)
graphs = []
# G = cyclic_graph(5)
# graphs.append(G)

# G = regular_graph(100,50)
# graphs.append(G)

# G = bipartite(50,50)
# graphs.append(G)



# Plots Walshaw collection
# pwd = os.getcwd()
# f = open(pwd + '/data/3elt.graph')
# G = Graph(f)
# graphs.append(G)
# print(G)

# f = open(pwd + '/data/crack.graph')
# G = Graph(f)
# graphs.append(G)
# print(G)

# f = open(pwd + '/data/add20.txt')
# G = Graph(f)
# graphs.append(G)
# print(G)

# f = open(pwd + '/data/uk.graph')
# G = Graph(f)
# graphs.append(G)
# print(G)

f = open(pwd + '/data/4elt.txt')
G = Graph(f)
graphs.append(G)
print(G)

for G in graphs:
    draw_n(G, 1, p = 2, tol = 1e-8, max_iter = 1000, plot_params = plot_params, mode = 0, reference = False)

# # SBM graph
# tol = 1e-10
# max_iter = 10000

# alpha = 0.99
# lbda = 0.99
# n = 500
# K = 5
# G = gen_sbm_graph(alpha, lbda, n, K)
# graphs.append(G)

# # Plots SBMs
# alpha = 0.99
# lbda = 0.99
# Ks = [1, 2, 4, 8, 16]
# alphas = np.linspace(0.1, 0.99, num = 5)
# lbdas = np.linspace(0.9, 0.99, num = 5)
# n = 500
# K = 5
# # Increasing lbda
# graphs = []
# for lbda in lbdas:
#     G = gen_sbm_graph(alpha, lbda, n, K)
#     graphs.append(G)
# for G in graphs:
#      U = draw(G, tol = tol, max_iter = max_iter, plot_params = plot_params, mode = 0, reference = True)

# # Increasing alpha
# graphs = []
# for alpha in alphas:
#     G = gen_sbm_graph(alpha, lbda, n, K)
#     graphs.append(G)
# for G in graphs:
#      U = draw(G, tol = tol, max_iter = max_iter, plot_params = plot_params, mode = 0, reference = True)

# # Increasing K
# graphs = []
# for K in Ks:
#     G = gen_sbm_graph(alpha, lbda, n, K)
#     graphs.append(G)
# for G in graphs:
#      U = draw(G, tol = tol, max_iter = max_iter, plot_params = plot_params, mode = 0, reference = True)

# # Saved SBM graph with 500 nodes, 5 communities and alpha, lambda = 0.99
# G = Graph()
# G.set_adj_matrix(sbm_500_5_99_99)
# G.set_name("sbm_500_5_99_99")

# for G in graphs:
#     U = draw(G, tol = 1e-8, max_iter = 2000, plot_params = plot_params, mode = 0)


# #Degree normalized eigenvectors
# U, times = degree_normalized_eigenvectors(np.diag(G.degs), G.adj_matrix, 2, tol = 1e-6, max_iter = 2000, matmul = True)

# #B = 0.5 * (np.eye(G.n_nodes) + np.diag(1. / G.degs)@G.adj_matrix)
# #2D graph plot coordinates
# x_coord = U[:, 0]
# y_coord = U[:, 1]
# graph_plot(G.adj_matrix, x_coord, y_coord, node_size = 0.01, figsize = (3,3), dpi = 200, add_labels= False)
