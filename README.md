# Drawing Graphs by Eigenvectors

This project focused on spectral graph drawing methods, which construct the layout of a graph using eigenvectors of certain matrices. One possible approach is the force-directed strategy [1], which defines an energy function, with the minimum determining a drawing that is optimal in a certain sense. Another approach using the generalized eigenvectors of the graph Laplacian was proposed in [2].

# Generate graph drawings

To obtain graph drawings for the different graphs stored in the /data folder, as well as other artificially generated graphs from the graph_collection, sbm and barabasi_albert modules, run the file main.py, uncommenting one or more of the functions that are included there, namely: generate_sbm_plots, generate_walshaw_plots, generate_basic_plots. The code is clear enough to adapt it in case users want to plot their own graphs, which should be uploaded as file in the format explained here: https://chriswalshaw.co.uk/jostle/jostle-exe.pdf

# Explanation

The main function to be taken into account to draw graphs is the function ```draw_n``` from the ```spectral_drawing module```. 

```
draw_n(G: Graph, n: int, p=2, method = "rayleigh", tol=1e-8, max_iter=1000, node_size=0.01, edge_width=0.1, figsize=(3, 3), dpi=200, mode=0, plot_params=[False for _ in range(n_plot_params)], reference=False):
```

## References

[1] Giuseppe Di Battista, Peter Eades, Roberto Tamassia, and Ioannis G Tollis. Graph drawing:
algorithms for the visualization of graphs. Prentice Hall PTR, 1998.

[2] Y. Koren. Drawing graphs by eigenvectors: theory and practice. Comput. Math. Appl.,
49(11-12):1867–1888, 2005.

[3] Jing Lei and Alessandro Rinaldo. Consistency of spectral clustering in stochastic block
models. Ann. Statist., 43(1):215–237, 2015.