from graph_plot import *
from degree_normalized_eigenvectors import *
from graph_class import Graph
pwd = "/Users/guifre/cla_project/cla-project-drawing-graphs-by-eigenvectors"
f = open(pwd + '/data/crack.graph')
G = Graph(f)
print(G)

#Degree normalized eigenvectors
U, times = degree_normalized_eigenvectors(np.diag(G.degs), G.adj_matrix, 2, tol = 1e-6, max_iter = 2000, matmul = True)

#B = 0.5 * (np.eye(G.n_nodes) + np.diag(1. / G.degs)@G.adj_matrix)
#2D graph plot coordinates
x_coord = U[:, 0]
y_coord = U[:, 1]
graph_plot(G.adj_matrix, x_coord, y_coord, node_size = 0.01, figsize = (3,3), dpi = 200, add_labels= False)