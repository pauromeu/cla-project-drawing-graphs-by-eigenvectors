import hde
import numpy as np
from pm import *
from scipy.stats import ortho_group
from graph_class import *
from sbm import *

# Test bfs distance function
# Example graph represented as a list of NumPy arrays
graph = [
    np.array([1, 2]),    # Node 0, neighbors: 1, 2
    np.array([0, 3, 4]), # Node 1, neighbors: 0, 3, 4
    np.array([0, 5]),    # Node 2, neighbors: 0, 5
    np.array([1]),       # Node 3, neighbor: 1
    np.array([1, 5]),    # Node 4, neighbors: 1, 5
    np.array([2, 4])     # Node 5, neighbors: 2, 4
]

# Call BFS to compute shortest distance between nodes 0 and 5
shortest_distance = hde.bfs_shortest_distance(graph, 0, 5)
print("Shortest distance between nodes 0 and 5:", shortest_distance)

# Test power method
min_eigv = 0
max_eigv = 10
dim = 5
d = np.random.uniform(min_eigv, max_eigv, dim)
V = ortho_group.rvs(dim = dim)
A = V@np.diag(d)@V.T
p = 2
print("Original eigenvectors: ")
print(np.sort(d))
U = power_method(A,p, tol = 1e-7, test = True, mode = 1)

# Test shortest path distances
adj_list = [
  [1],
  [0,2],
  [1,3,5],
  [2,6],
  [5,7],
  [4,2],
  [3],
  [4]
]

# Perform BFS starting from node 0 and get the shortest path distances
j = 7
print(len(adj_list))
shortest_distances = bfs_shortest_path_distances(adj_list, j)
print(shortest_distances)

# Print the shortest path distances
for i in range(len(adj_list)):
    print("Dist(", j, ",", i,") = ", shortest_distances[i])
    
# Testing SBM
ex22 = example22(1.0, 0.5, 5)
print(ex22)
G = gen_sbm_graph(1.0, 0.5, 5)
print(G.adj_matrix)
print(G.degs)
print(G.laplacian)
print(G.adj_list)