import numpy as np
import random as rand
import graph_class as graph

def sbm_adj_matrix(Theta, B, prints = False):
    n_nodes = len(Theta[:,0])
    g = []
    for i in range(n_nodes):
        g.append(np.where(Theta[i,:] == 1)[0][0])
    g = np.array(g)
    if prints: print("node's communities: ", g)

    adj_matrix = np.zeros(shape = (n_nodes, n_nodes))
    for j in range(1, n_nodes + 1):
        for i in range(1, j + 1):
            if i < j:
                adj_matrix[i - 1,j - 1] = np.random.binomial(1, B[g[i - 1], g[j - 1]])
            elif i == j:
                adj_matrix[i - 1,j - 1] = 0
                
    for i in range(1, n_nodes + 1):
        for j in range(1, i):
            adj_matrix[i - 1,j - 1] = adj_matrix[j - 1,i - 1]
    return adj_matrix

def gen_sbm_graph(alpha, lbda, n_nodes):
    G = graph.Graph()
    G.set_adj_matrix(example22(alpha,lbda,n_nodes))   
    return G 

def example22(alpha, lbda, n_nodes, prints = False):
    all_ones = np.array([np.ones(n_nodes)]).T # To create column vectors that can be interpreted as matrices...
    B0 = lbda * np.eye(n_nodes) + (1 - lbda) * (all_ones@all_ones.T)
    B = alpha * B0
    Theta = np.eye(n_nodes)
    if prints: 
        print(Theta)
        print(alpha * Theta)
        print(all_ones)
        print(B)
    return sbm_adj_matrix(Theta, B)


# # Testing
# ex22 = example22(1.0, 0.5, 5)
# print(ex22)
# G = gen_sbm_graph(1.0, 0.5, 5)
# print(G.adj_matrix)
# print(G.degs)
# print(G.laplacian)
# print(G.adj_list)
            