# Drawing Graphs by Eigenvectors

This project focused on spectral graph drawing methods, which construct the layout of a graph using eigenvectors of certain matrices. One possible approach is the force-directed strategy [1], which defines an energy function, with the minimum determining a drawing that is optimal in a certain sense. Another approach using the generalized eigenvectors of the graph Laplacian was proposed in [2].

# Generate graph drawings

To obtain graph drawings for the different graphs stored in the ```/data``` folder, as well as other artificially generated graphs from the ```graph_collection```, ```sbm``` and ```barabasi_albert``` modules, run the file ```main.py```, uncommenting one or more of the functions that are included there, namely: ```generate_sbm_plots```, ```generate_walshaw_plots```, ```generate_basic_plots```. The code is clear enough to adapt it in case users want to plot their own graphs, which should be uploaded as files in the format explained here: https://chriswalshaw.co.uk/jostle/jostle-exe.pdf. Alternatively, users can upload adjacency matrices and initialize the corresponding graph by running:

```
A = np.load('file_containing_adj_matrix')
G = Graph()
G.set_adj_matrix(A)
```

# Explanation

The main function to be taken into account to draw graphs is the function ```draw_n``` from the ```spectral_drawing module```:

```
draw_n(G: Graph, n: int, p=2, method = "rayleigh", tol=1e-8, max_iter=1000, node_size=0.01, edge_width=0.1, figsize=(3, 3), dpi=200, mode=0, plot_params=[False for _ in range(n_plot_params)], reference=False):
```

The different options and parameters of this function are explained in its dosctring, which we include here for completeness:

```
Draw a graph using eigenvectors and save the plot.

Parameters:
    G (Graph): The input graph.
    p (int): Number of eigenvectors to use for drawing. Default is 2.
    method (str): Method to compute the drawing. Available options: "original", "rayleigh", "pm", "ref". Default is "rayleigh".
    tol (float): Tolerance for convergence. Default is 1e-8.
    max_iter (int): Maximum number of iterations. Default is 1000.
    node_size (float): Size of the graph nodes in the plot. Default is 0.01.
    edge_width (float): Width of the graph edges in the plot. Default is 0.1.
    figsize (tuple): Figure size (width, height) in inches. Default is (3, 3).
    dpi (int): Dots per inch for the figure. Default is 200.
    mode (int): residual mode (0 - <x, x_prev>, 1 - norm(A x - rayleigh_quotient(A,x)x), 2 - norm(x - x_prev)). Default is 0. 
    plot_params (list): List of plot parameters. See main for more information.
    numbering (int): Numbering for the graph. Default is -1.

Returns:
    True if the drawing was successful
```

The matrix for which we compute eigenvectors is, as suggested in reference [2], B := 0.5 * (I_n + D^{-1} A), where D, A are the degree and adjacency matrices of the graph G.
Regarding the method to compute corresponding eigenvectors, the following should be highlighted:
- ```rayleigh```: here we use Rayleigh quotient iteration on the matrix B, together with successive orthonormalization of the current eigenvector to be computed with respect to already computed eigenvectors. We start with an initial guess for sigma = 1, and, at eatch eigenvector generation step, we set as initial guess the last obtained sigma for the previous eigenvector minus a small quantity, e.g. 1e-4, which can depend on the graph to be drawn. The LU decomposition with pivoting of the shift A - sigma * I is computed using the ```splu``` function from ```scipy.sparse.linalg```.
- ```original```: we follow the algorithm presented in Figure 3 of reference [2]
- ```pm```: we use a basic implementation of the power method on the matrix B.
- ```ref```: we use the reference method provided by the function ``egis`` again from ```scipy.sparse.linalg``` applied to the matrix B.

## References

[1] Giuseppe Di Battista, Peter Eades, Roberto Tamassia, and Ioannis G Tollis. Graph drawing:
algorithms for the visualization of graphs. Prentice Hall PTR, 1998.

[2] Y. Koren. Drawing graphs by eigenvectors: theory and practice. Comput. Math. Appl.,
49(11-12):1867–1888, 2005.

[3] Jing Lei and Alessandro Rinaldo. Consistency of spectral clustering in stochastic block
models. Ann. Statist., 43(1):215–237, 2015.
