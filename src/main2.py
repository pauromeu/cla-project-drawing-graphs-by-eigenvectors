from graph_plot import *
from spectral_drawing import *
from graph_class import Graph
from graph_collection import *
import os
plot_params = [False for _ in range(n_plot_params)]
plot_params[title_index] = True
plot_params[label_index] = True
plot_params[axis_index] = True
plot_params[grid_index] = True 
# pwd = "/Users/guifre/cla_project/cla-project-drawing-graphs-by-eigenvectors"
# f = open(pwd + '/data/crack.graph')
# G = Graph(f)
# print(G)
graphs = []
G = cyclic_graph(5)
graphs.append(G)

G = regular_graph(100,50)
graphs.append(G)

G = bipartite(50,50)
graphs.append(G)

pwd = os.getcwd()
f = open(pwd + '/data/3elt.graph')
G = Graph(f)
graphs.append(G)

for G in graphs:
    U = draw(G, max_iter = 2000, plot_params = plot_params)




# #Degree normalized eigenvectors
# U, times = degree_normalized_eigenvectors(np.diag(G.degs), G.adj_matrix, 2, tol = 1e-6, max_iter = 2000, matmul = True)

# #B = 0.5 * (np.eye(G.n_nodes) + np.diag(1. / G.degs)@G.adj_matrix)
# #2D graph plot coordinates
# x_coord = U[:, 0]
# y_coord = U[:, 1]
# graph_plot(G.adj_matrix, x_coord, y_coord, node_size = 0.01, figsize = (3,3), dpi = 200, add_labels= False)