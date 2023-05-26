import numpy as np
from queue import Queue

def bfs_shortest_distance(graph, start_node, target_node):
    snode = start_node
    tnode = target_node
    visited = set()  # Set to keep track of visited nodes
    queue = Queue()  # Queue to store nodes for traversal
    distances = {start_node: 0}  # Dictionary to store distances from start node

    # Add the start node to the queue and mark it as visited
    queue.put(start_node)
    visited.add(start_node)

    while not queue.empty():
        current_node = queue.get()

        if current_node == target_node:
            print("Shortest distance ", snode, ", ", tnode, " = ", distances[current_node])
            return distances[current_node]  # Return the shortest distance if target node is found

        # Explore neighbors of the current node
        neighbors = graph[current_node]
        for neighbor in neighbors:
            if neighbor not in visited:
                queue.put(neighbor)
                visited.add(neighbor)
                distances[neighbor] = distances[current_node] + 1  # Update the distance to the neighbor

    return float('inf')  # Return infinity if there is no path between the start and target nodes

class Graph:
    def bfs_distance(self, start_node, target_node):
        return bfs_shortest_distance(self.chaco_array, start_node, target_node)
    def vec_bfs_distance(self, node_list):
        vec_bfs_dist = np.vectorize(Graph.bfs_distance)
        return vec_bfs_dist(node_list)

    def get_nodes_edges_from_file(chaco_file):
        # get number of nodes and edges from chaco file first line
        n_nodes = 0
        n_edges = 0
        graph_info = []
        f = chaco_file
        for word in f.readline().split():
            graph_info.append(int(word))
        n_nodes, n_edges = graph_info[0], graph_info[1]
        return n_nodes, n_edges
    
    def get_chaco_array_from_file(chaco_file):
        # get chaco array from graph chaco file
        # chaco_array[i] = np array containing indices for nodes that are neighbours with node i.
        chaco_array = []
        f = chaco_file
        for line in f:
            nbhs = []
            for word in line.split():
                nbhs.append(int(word) - 1) # nodes in original chaco file are indexed starting at 1 and not 0
            nbhs = np.array(nbhs)
            chaco_array.append(nbhs)
        return chaco_array
        
    def __init__(self, chaco_file = None):
        if chaco_file != None:
            self.n_nodes, self.n_edges = Graph.get_nodes_edges_from_file(chaco_file)
            self.chaco_array = Graph.get_chaco_array_from_file(chaco_file)
            self.adj_matrix = Graph.get_adj_matrix_from_chaco_array(self.chaco_array)
        else:
            self.n_nodes = -1
            self.n_edges = -1
            self.chaco_array = None
            self.adj_matrix = None
    
    def get_adj_matrix_from_chaco_array(chaco_array, prints = False):
        n_nodes = len(chaco_array)
        adj_matrix = np.zeros(shape = [n_nodes, n_nodes])
        for n in range(n_nodes):
            adj_matrix[n, chaco_array[n]] = 1
        if prints: 
            print(adj_matrix)
        return adj_matrix
    
    def __str__(self):
        return f"[graph object] nodes = {self.n_nodes}; edges = {self.n_edges}"