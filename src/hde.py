import numpy as np
from numpy.linalg import norm
from graph_class import Graph
import random as rand
from queue import Queue
from collections import deque
import time as time


def bfs_shortest_distance(graph, start_node, target_node):
    visited = set()  # Set to keep track of visited nodes
    queue = Queue()  # Queue to store nodes for traversal
    distances = {start_node: 0}  # Dictionary to store distances from start node

    # Add the start node to the queue and mark it as visited
    queue.put(start_node)
    visited.add(start_node)

    while not queue.empty():
        current_node = queue.get()

        if current_node == target_node:
            return distances[current_node]  # Return the shortest distance if target node is found

        # Explore neighbors of the current node
        neighbors = graph[current_node]
        for neighbor in neighbors:
            if neighbor not in visited:
                queue.put(neighbor)
                visited.add(neighbor)
                distances[neighbor] = distances[current_node] + 1  # Update the distance to the neighbor

    return float('inf')  # Return infinity if there is no path between the start and target nodes
                

def hde(G: Graph, m: int, prints = False, D_orth = True):
    """ Finds m-dimensional high-dimensional embedding of the graph G

    Args:
        G (Graph): the graph.
        m (int): the dimension of the embedding subspace.
    """
    n = G.n_nodes
    p = np.zeros(m)
    X = np.zeros(shape = (n,  m))
    d = np.zeros(n)
    d[:] = float('inf')
    # Choose first pivot node at random
    p[0] = rand.randint(0,n - 1)
    # print(p[0], n)
    adj_list = G.chaco_array
    # test_vec =  np.array(np.vectorize(G.bfs_distance)([1, 2, 3, 4, 5, 6],[120, 12, 34, 30, 1, 114]))
    # print("distance", np.vectorize(G.bfs_distance)([1, 2, 3, 4, 5, 6],[120, 12, 34, 30, 1, 114]))
    # print(test_vec)
    # print(np.minimum(d[:6],test_vec))
    for i in range(m):
        if prints: print("Computing X^", i, "...")
        t0 = time.time()
        t1 = time.time()
        if prints: print("time elapsed 1: ", t1 - t0)
        t0 = time.time()
        pivot = int(p[i])
        X[:,i] = G.bfs_distances(pivot)
        if prints: print("finished computing bfs distances")
        d = np.minimum(d, X[:,i])
        t1 = time.time()
        if prints: print("time elapsed 2: ", t1 - t0)
        # choose next pivot
        if i + 1 <= m - 1: p[i + 1] = np.argmax(d)
        
    return gram_schmidt_all_ones(X, D_orth = D_orth)
    
def gram_schmidt_all_ones(U, prints = False, D_orth = True):
    # Remains to implement D-orthogonalized version of this... 
    dim = len(U[:,0])
    m = len(U[0,:])
    u0 = np.full(dim, 1)
    u0 = u0 / np.sqrt(np.linalg.norm(u0))
    U = np.concatenate((u0[:,np.newaxis], U), axis = 1)
    if prints:
        print(U)
        print(np.shape(U))
    epsilon = 1e-4
    for i in range(1, m + 1):
        if prints: print(i)
        for j in range(i):
            U[:,i] = U[:,i] - np.dot(U[:,i],U[:,j]) * U[:,j]
        norm_ui = np.linalg.norm(U[:,i])
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






    