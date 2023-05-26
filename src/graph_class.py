import numpy as np
class Graph:
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
    
    def __init__(self, n_nodes, n_edges, chaco_array):
        self.n_nodes = n_nodes
        self.n_edges = n_edges
        self.chaco_array = chaco_array
        self.adj_matrix = Graph.get_adj_matrix_from_chaco_array(self.chaco_array)
        
    def __init__(self, chaco_file):
        self.n_nodes, self.n_edges = Graph.get_nodes_edges_from_file(chaco_file)
        self.chaco_array = Graph.get_chaco_array_from_file(chaco_file)
        self.adj_matrix = Graph.get_adj_matrix_from_chaco_array(self.chaco_array)
    
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