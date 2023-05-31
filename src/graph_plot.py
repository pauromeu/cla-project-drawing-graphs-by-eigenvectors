import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from graph_class import Graph
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams.update({'font.size': 11})
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['mathtext.fontset'] = 'cm'
main_colors = ["r","b","c", "g", "m", "k"]

# How to put math characters in plot
# "grad_norms": r"$\Vert\nabla \ell\Vert$",
#     "costs": "Negative log-likelihood"

grid_index = 0
axis_index = 1
label_index = 2
title_index = 3
ticks_index = 4

def graph_plot(G, x_coord, y_coord, title="Graph", node_size=1, edge_width=0.1, cross=False, figsize=(7, 7), dpi=200, add_labels=True, plot_params = [False, False, False, False, False]):
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
    adj_matrix = G.adj_matrix
    graph = nx.from_numpy_array(adj_matrix)

    # create dictionary of node positions using (x, y) coordinates
    pos = {i: (x_coord[i], y_coord[i]) for i in range(len(x_coord))}

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(1, 1, 1)


    # draw nodes and edges
    node_size = node_size
    nx.draw_networkx_nodes(graph, pos, node_color='blue', node_size=node_size)
    nx.draw_networkx_edges(graph, pos, edge_color='red', width=edge_width)

    # add text labels to identify each vertex
    if add_labels:
        for i in range(len(x_coord)):
            plt.text(x_coord[i], y_coord[i], str(i+1), ha='center',
                     va='center', color='black', weight='bold')

    # add a grey cross at (0,0)
    if cross:
        cross_size = 0.05
        plt.plot([-cross_size, cross_size], [0, 0], color='grey',
                 linewidth=0.5, markersize=node_size)
        plt.plot([0, 0], [-cross_size, cross_size], color='grey',
                 linewidth=0.5, markersize=node_size)

    # if add_labels:
    #     # add labels to edges with weights from adjacency matrix
    #     edge_labels = {(i, j): adj_matrix[i][j] for i, j in graph.edges()}
    #     nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_color='black')

    # set axis labels and title
    if plot_params[label_index]:
        plt.xlabel(r"$x$")
        plt.ylabel(r"$y$")
    if plot_params[title_index]:
        plt.title("awesome graph :P")
    if plot_params[grid_index]: 
        plt.grid(True, linewidth = 0.05)
        
    # PROBLEM WITH TICKS, TRY TO FIX LATER
    if plot_params[ticks_index]:
        xticks = np.arange(-1,1,0.2)    
        ax.set_xticks(xticks)
        ax.set_xticklabels([xticks[i] for i in range(len(xticks))])

    # set equal scaling for both axes
    plt.axis('equal')
    if plot_params[axis_index]:
        for axis in ax.spines.keys():
            ax.spines[axis].set_linewidth(0.5)
    else:
        for axis in ax.spines.keys():
            ax.spines[axis].set_linewidth(0)
        
    filename = G.name
    plt.savefig("plots/" + filename)

    # display plot
    plt.show()
    plt.close()
