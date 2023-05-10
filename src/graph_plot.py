import networkx as nx
import matplotlib.pyplot as plt


def graph_plot(adj_matrix, x_coord, y_coord, title=None):
    """
    Plot a graph with nodes and edges based on provided adjacency matrix and node coordinates.

    Parameters
    ----------
    adj_matrix : numpy.ndarray
        The adjacency matrix of the graph. Should be a square matrix of size n x n.
    x_coord : list or numpy.ndarray
        The x-coordinates of the nodes. Should be a list or array of length n.
    y_coord : list or numpy.ndarray
        The y-coordinates of the nodes. Should be a list or array of length n.

    Notes
    -----
    This function uses networkx to create a graph from the adjacency matrix and matplotlib 
    to plot the graph. It plots nodes at the provided (x, y) coordinates and draws edges 
    between nodes based on the adjacency matrix. It also adds a cross at (0, 0).

    """
    # create networkx graph from adjacency matrix
    graph = nx.from_numpy_array(adj_matrix)

    # create dictionary of node positions using (x, y) coordinates
    pos = {i: (x_coord[i], y_coord[i]) for i in range(len(x_coord))}

    # draw nodes and edges
    node_size = 200
    nx.draw_networkx_nodes(graph, pos, node_color='blue', node_size=node_size)
    nx.draw_networkx_edges(graph, pos, edge_color='red', width=1)

    # add text labels to identify each vertex
    for i in range(len(x_coord)):
        plt.text(x_coord[i], y_coord[i], str(i+1), ha='center',
                 va='center', color='white', weight='bold')

    # add a grey cross at (0,0)
    cross_size = 0.05
    plt.plot([-cross_size, cross_size], [0, 0], color='grey',
             linewidth=0.5, markersize=node_size)
    plt.plot([0, 0], [-cross_size, cross_size], color='grey',
             linewidth=0.5, markersize=node_size)

    # add labels to edges with weights from adjacency matrix
    # edge_labels = {(i, j): adj_matrix[i][j] for i, j in graph.edges()}
    # nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_color='black')

    # set axis labels and title
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Graph')

    # set equal scaling for both axes
    plt.axis('equal')

    # display plot
    plt.show()
