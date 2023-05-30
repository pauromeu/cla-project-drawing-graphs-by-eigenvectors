import numpy as np
from numpy.linalg import norm
from graph_class import Graph
import random as rand
from queue import Queue
from collections import deque
import time as time

max_dist = float('inf')

def hde(G: Graph, m: int, prints = False, D_orth = True):
    """ Finds m-dimensional high-dimensional embedding of the graph G

    Args:
        G (Graph): the graph.
        m (int): the dimension of the embedding subspace.
        prints: if True provides useful prints for debugging. Defaults False. 
        D_orth: if True, returns matrix X D-orthogonalized, i.e. X^T D X = In
        
    Returns:
        X: matrix encoding the m-dimensional HDE of the graph G
    """
    n = G.n_nodes
    p = np.zeros(m)
    X = np.zeros(shape = (n,  m))
    d = np.full(n, max_dist)
    # Choose first pivot node at random
    p[0] = rand.randint(0,n - 1) # includes both extrems when computing random integer
    adj_list = G.adj_list
    for i in range(m):
        if i > 0 and p[i] == p[i - 1] and 0:
            p[i] = rand.randint(0,n - 1)
        if prints: print("Computing X^", i, "...")
        
        t0 = time.time()
        pivot = int(p[i])
        X[:,i] = G.bfs_distances(pivot)
        if prints: print("finished computing bfs distances")
        
        d = np.minimum(d, X[:,i])
        t1 = time.time()
        if prints: print("time elapsed 2: ", t1 - t0)
        
        if i + 1 <= m - 1: p[i + 1] = np.argmax(d)
        
    if prints:
        print("Rank of X before orthonormalize: ", np.linalg.matrix_rank(X))
        print("p:",p)
        print("d: ",d)
        print("X: ")
        print(X)
    return gram_schmidt_all_ones(G, X, D_orth = D_orth)
    
def gram_schmidt_all_ones(G, U, prints = False, D_orth = True):
    """ Gram-Schmidt orthogonalization (or D-orthogonalization if D_orth = True) of the vectors in U, 
    which, at the same time, orthogonalizes (or D-orthogonalizes) against the all ones vector. 

    Args:
        G: graph that contains the degrees matrix D
        U: 
        prints (bool, optional): _description_. Defaults to False.
        D_orth (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    dim = len(U[:,0])
    m = len(U[0,:])
    degs = G.degs
    u0 = np.full(dim, 1)
    u0 = u0 / np.sqrt(np.dot(u0, degs* u0))
    U = np.concatenate((u0[:,np.newaxis], U), axis = 1)
    if prints:
        print(U)
        print(np.shape(U))
    epsilon = 1e-3
    for i in range(1, m + 1):
        if prints: print(i)
        for j in range(i):
            U[:,i] = U[:,i] - np.dot(U[:,i],degs * U[:,j]) / (np.dot(U[:,j], degs * U[:,j]))* U[:,j]
        
        norm_ui = np.sqrt(np.dot(U[:,i], degs * U[:,i]))
        if norm_ui < epsilon:
            U[:,i] = np.zeros(dim)
        else:
            U[:,i] = U[:,i] / norm_ui
    
    result = U[:,1:]
    if prints:
        print(result)
        print(result.T@result)
    return result
       

def bfs_shortest_path_distances(adj_list, start_node):
    # Number of vertices in the graph
    n = len(adj_list)
    
    # List to keep track of visited nodes
    visited = [False] * n
    
    # Queue to store the nodes to visit
    queue = deque()
    
    # Distance dictionary to store the shortest path distances
    distance = {start_node: 0}
    
    # Mark the start node as visited and enqueue it
    visited[start_node] = True
    queue.append(start_node)
    
    # Perform BFS
    while queue:
        # Dequeue a node from the front of the queue
        current_node = queue.popleft()
        
        # Visit all adjacent nodes of the current node
        for adjacent_node in adj_list[current_node]:
            # If the adjacent node hasn't been visited, mark it as visited, enqueue it,
            # and update the distance as the distance to the current node + 1
            if not visited[adjacent_node]:
                visited[adjacent_node] = True
                queue.append(adjacent_node)
                distance[adjacent_node] = distance[current_node] + 1
    
    np_distances = []
    for node, dist in distance.items():
        np_distances.append(dist)
    np_distances = np.array(np_distances)
    return np_distances
     
       
# # Test shortest path distances
# # Example graph represented as an adjacency list
# adj_list = [[1, 2], [0, 2, 3], [0, 1, 3], [1, 2]]
# adj_list = [
#   [1, 2, 3],
#   [0, 4],
#   [0, 5, 6],
#   [0, 7],
#   [1, 8],
#   [2, 9],
#   [2],
#   [3],
#   [4],
#   [5]
# ]

# # Perform BFS starting from node 0 and get the shortest path distances
# shortest_distances = bfs_shortest_path_distances(adj_list, 0)

# # Print the shortest path distances
# print(shortest_distances)


# pwd = "/Users/guifre/cla_project/cla-project-drawing-graphs-by-eigenvectors"
# f = open(pwd + "/data/add20.txt", "r")
# G = Graph(f)

# hde(G, 50)






    