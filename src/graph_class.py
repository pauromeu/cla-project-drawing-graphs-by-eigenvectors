import numpy as np
from queue import Queue
from collections import deque

def bfs_shortest_path_distances(adj_list, start_node):
    """Simple function to compute shortest path distances from start_node to all other nodes in the graph, 
    which is given via an adjacency list adj_list, using Breadth First Search.
    """
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
    
    np_distances = np.zeros(n)
    for node, dist in distance.items():
        np_distances[node] = dist

    return np_distances

class Graph:
    """Graph class. Its attributes are:
    adj_list: adjacency list for all the nodes in the graph.
    n_nodes, n_edges: number of nodes and edges, respectively.
    adj_matrix: adjacency matrix of the graph.
    laplacian: laplacian matrix of the graph. 
    degs: array of degrees of the graph.
    """
    def bfs_distances(self, start_node):
        return bfs_shortest_path_distances(self.adj_list, start_node)
    
    def vec_bfs_distances(self, start_node_list):
        return np.vectorize(Graph.bfs_distances)(start_node_list)

    def get_nodes_edges_from_file(chaco_file):
        # get number of nodes and edges from chaco file first line
        n_nodes = 0
        n_edges = 0
        graph_info = []
        f = chaco_file
        i = 0
        for word in f.readline().split():
            graph_info.append(int(word))
        n_nodes, n_edges = graph_info[0], graph_info[1]
        return n_nodes, n_edges
    
    def get_chaco_array_from_file(chaco_file):
        # get chaco array from graph chaco file
        # chaco_array[i] = np array containing indices for nodes that are neighbours with node i.
        chaco_array = []
        f = chaco_file
        i = 0
        for line in f:
            if i == 0:
                print(line)
            nbhs = []
            for word in line.split():
                nbhs.append(int(word) - 1) # nodes in original chaco file are indexed starting at 1 and not 0
            nbhs = np.array(nbhs)
            chaco_array.append(nbhs)
            i += 1
        return chaco_array
    
    def set_degs(self, prints = False):
        if prints: print("Setting degrees")
        self.degs = np.array([np.sum(self.adj_matrix[i,:]) for i in range(self.n_nodes)])
        
    def __init__(self, chaco_file = None):
        if chaco_file != None:
            self.n_nodes, self.n_edges = Graph.get_nodes_edges_from_file(chaco_file)
            self.adj_list = Graph.get_chaco_array_from_file(chaco_file)
            self.adj_matrix = Graph.get_adj_matrix_from_chaco_array(self.adj_list)
            Graph.set_degs(self)
            Graph.set_laplacian(self)
            
            # Set name of graph as name of file without extension
            filename = chaco_file.name
            name_len = len(filename)
            name = filename[name_len - filename[::-1].find('/'):filename.find('.')]
            self.name = name
        else:
            self.n_nodes = None
            self.n_edges = None
            self.adj_list = None
            self.adj_matrix = None
            self.degs = None
            self.laplacian = None
            self.name = None
    
    def set_adj_list(self):
        if self.adj_matrix.all() != None and self.adj_list == None:
            self.adj_list = []
            for i in range(self.n_nodes):
                adj_list_node_i = []
                for j in range(self.n_nodes):
                    if self.adj_matrix[i,j] == 1:
                        adj_list_node_i.append(j)
                self.adj_list.append(adj_list_node_i)
                
    def set_name(self, name: str):
        self.name = name        
   
    def set_adj_matrix(self, A):
        self.adj_matrix = A
        self.n_nodes = len(A[0,:])
        self.n_edges = np.sum(A) / 2
        self.set_degs()
        self.set_adj_list()
        self.set_laplacian()
    
    def get_adj_matrix_from_chaco_array(chaco_array, prints = False):
        n_nodes = len(chaco_array)
        adj_matrix = np.zeros(shape = [n_nodes, n_nodes])
        for n in range(n_nodes):
            adj_matrix[n, chaco_array[n]] = 1
        if prints: 
            print(adj_matrix)
        return adj_matrix
    
    def set_laplacian(self):
        self.laplacian = self.adj_matrix - np.diag(self.degs)

    def __str__(self):
        return f"[graph object] nodes = {self.n_nodes}; edges = {self.n_edges}"
    
    

# # Test shortest path distances
# adj_list = [
#   [1],
#   [0,2],
#   [1,3,5],
#   [2,6],
#   [5,7],
#   [4,2],
#   [3],
#   [4]
# ]

# # Perform BFS starting from node 0 and get the shortest path distances
# j = 7
# print(len(adj_list))
# shortest_distances = bfs_shortest_path_distances(adj_list, j)
# print(shortest_distances)

# # Print the shortest path distances
# for i in range(len(adj_list)):
#     print("Dist(", j, ",", i,") = ", shortest_distances[i])

# v1 = np.array([1,2,3,4])
# v2 = np.array([1,-1,1,-1])

# print(v1*v2)
 

