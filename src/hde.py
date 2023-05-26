import numpy as np
from numpy.linalg import norm
from graph_class import Graph
import random as rand
from queue import Queue


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
                

def hde(G: Graph, m: int):
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
    print(p[0])
    adj_list = G.chaco_array
    print("distance", np.vectorize(G.bfs_distance)([55,44, 43],[120, 12, 34]))
    for i in range(0):
        print("Computing X^", i, "...")
        for j in range(n):
            X[j,i] = bfs_shortest_distance(adj_list, int(p[i]), j)
            d[j] = min(d[j], X[j,i])
        # choose next pivot
        if i <= m - 2: p[i + 1] = np.argmax(d)
        
    return X
    

pwd = "/Users/guifre/cla_project/cla-project-drawing-graphs-by-eigenvectors"
f = open(pwd + "/data/add20.txt", "r")
G = Graph(f)

hde(G, 20)






    