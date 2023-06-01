from graph_collection import regular_graph
from graph_class import Graph
from graph_plot import *
from spectral_drawing import *
from lanczos import implicitly_restarted_lanczos
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
import os

graphs = []


plot_params = [False for _ in range(n_plot_params)]
plot_params[title_index] = False
plot_params[label_index] = True
plot_params[axis_index] = True
plot_params[grid_index] = True


G = regular_graph(6, 3)

draw_n(G, 4, plot_params=plot_params)

k = 5
num_regulars = 1
G = regular_graph(10, k)
draw_n(G, num_regulars, plot_params=plot_params)

G = regular_graph(20, k)
draw_n(G, num_regulars, plot_params=plot_params)

G = regular_graph(100, k)
draw_n(G, num_regulars, plot_params=plot_params)

G = regular_graph(200, k)
draw_n(G, num_regulars, plot_params=plot_params)

N_nodes = 100
G = regular_graph(N_nodes, 5)
draw_n(G, num_regulars, plot_params=plot_params)

G = regular_graph(N_nodes, 10)
draw_n(G, num_regulars, plot_params=plot_params)

G = regular_graph(N_nodes, 15)
draw_n(G, num_regulars, plot_params=plot_params)

G = regular_graph(N_nodes, 20)
draw_n(G, num_regulars, plot_params=plot_params)
